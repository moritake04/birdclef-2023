import argparse
import ast
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import wandb
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pathlib import Path

from classifier import bird2023classifier


def upsample_data(df, thr=20, seed=42):
    # get the class distribution
    class_dist = df["primary_label"].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df


def padded_cmap(y_true, y_score, padding_factor=5):
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(y_true.shape[1])])
    padded_y_true = np.concatenate([y_true, new_rows])
    padded_y_score = np.concatenate([y_score, new_rows])
    score = sklearn.metrics.average_precision_score(
        padded_y_true, padded_y_score, average="macro"
    )
    return score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    parser.add_argument("-r", "--resume_train", action="store_true")
    args = parser.parse_args()
    return args


def wandb_start(cfg):
    wandb.init(
        project=cfg["general"]["project_name"],
        name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
        group=f"{cfg['general']['save_name']}_cv" if cfg["general"]["cv"] else "all",
        job_type=cfg["job_type"],
        mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
        config=cfg,
    )


def train_and_predict(cfg, train_X, train_y, valid_X=None, valid_y=None):
    if cfg["model"]["pretrained_path"] is not None:
        print("Using pretrained model (past bird-clef comp)")
        print(cfg["model"]["pretrained_path"])
        tmp_num_classes = cfg["model"]["num_classes"]
        cfg["model"]["num_classes"] = cfg["model"]["pretrained_classes"]
        model = bird2023classifier.Bird2023Classifier(
            cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y
        )
        model.load_weight(cfg["model"]["pretrained_path"])
        model.model.set_head(tmp_num_classes)
    else:
        model = bird2023classifier.Bird2023Classifier(
            cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y
        )   
    model.train(weight_path=cfg["ckpt_path"])

    if valid_X is None:
        del model
        torch.cuda.empty_cache()
        return
    else:
        valid_preds = model.predict(valid_X)
        del model
        torch.cuda.empty_cache()
        return valid_preds


def one_fold(skf, cfg, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    
    # Ensures all classses exist on train fold
    train_y_minor = train_y[~train_X["cv"]].reset_index(drop=True)
    train_X_minor = train_X[~train_X["cv"]].reset_index(drop=True)
    train_y = train_y[train_X["cv"]]
    train_X = train_X[train_X["cv"]]
    train_y = train_y.reset_index(drop=True)
    train_X = train_X.reset_index(drop=True)
    
    train_indices, valid_indices = list(skf.split(train_X, train_X["primary_label"]))[
        fold_n
    ]
    train_X_cv, train_y_cv = (
        train_X.iloc[train_indices].reset_index(drop=True),
        train_y.iloc[train_indices].reset_index(drop=True),
    )
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    train_X_cv = pd.concat([train_X_cv, train_X_minor], axis=0).reset_index(drop=True)
    train_y_cv = pd.concat([train_y_cv, train_y_minor], axis=0).reset_index(drop=True)
    print(train_X_cv["primary_label"].unique().__len__())
    print(len(train_X_cv))
    
    if cfg["oversampling"] is not None:
        train_cv = pd.concat([train_X_cv, train_y_cv], axis=1).reset_index(drop=True)
        train_cv = upsample_data(train_cv, thr=cfg["oversampling"], seed=cfg["general"]["seed"])
        train_X_cv, train_y_cv = train_cv.iloc[:, :-264], train_cv.iloc[:, -264:]
        print(len(train_X_cv))
        
    # Add nocall data
    #"""
    f"{cfg['general']['input_path']}/aicrowd2020_noise_30sec/noise_30sec"
    datadir = Path(f"{cfg['general']['input_path']}/aicrowd2020_noise_30sec/noise_30sec/")
    all_audios = list(datadir.glob("*.ogg"))
    aicrowd2020 = ["aicrowd2020_noise_30sec/noise_30sec/" + ogg_name.name for ogg_name in all_audios]
    datadir = Path(f"{cfg['general']['input_path']}/ff1010bird_nocall/nocall/")
    all_audios = list(datadir.glob("*.ogg"))
    ff1010bird_nocall = ["ff1010bird_nocall/nocall/" + ogg_name.name for ogg_name in all_audios]
    datadir = Path(f"{cfg['general']['input_path']}/train_soundscapes/nocall/")
    all_audios = list(datadir.glob("*.ogg"))
    train_soundscapes = ["train_soundscapes/nocall/" + ogg_name.name for ogg_name in all_audios]
    df_X = pd.DataFrame(columns=train_X_cv.columns)
    df_X["filename"] = aicrowd2020 + ff1010bird_nocall + train_soundscapes
    df_y = pd.DataFrame(columns=train_y_cv.columns, data=np.zeros((len(df_X), 264)))
    train_X_cv = pd.concat([train_X_cv, df_X], axis=0).reset_index(drop=True)
    train_y_cv = pd.concat([train_y_cv, df_y], axis=0).reset_index(drop=True)
    print(len(train_X_cv))
    #"""
    
    valid_preds = train_and_predict(cfg, train_X_cv, train_y_cv, valid_X_cv, valid_y_cv)

    print(f"[fold_{fold_n}]")
    if cfg["audio"]["second_label"]:
        valid_y_cv[valid_y_cv < 1.0] = 0.0
    padded_cmap_score = padded_cmap(valid_y_cv.values, valid_preds)
    wandb.log({"padded_cmap_score": padded_cmap_score})

    torch.cuda.empty_cache()
    wandb.finish()

    return valid_preds, padded_cmap_score


def all_train(cfg, train_X, train_y):
    print("[all_train] start")
    seed_everything(cfg["general"]["seed"], workers=True)

    # train
    train_and_predict(cfg, train_X, train_y)

    return


def main():
    # Read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")

    # Set jobtype for wandb
    cfg["job_type"] = "train"

    # Set random seed
    seed_everything(cfg["general"]["seed"], workers=True)

    # Read csv
    train = pd.read_csv(f"{cfg['general']['input_path']}/train_metadata.csv")
    train["filename"] = "train_audio/" + train["filename"]
    train["secondary_labels"] = train["secondary_labels"].apply(ast.literal_eval)
    sample_submission = pd.read_csv(
        f"{cfg['general']['input_path']}/sample_submission.csv"
    )
    # Retrieve target
    birds = sample_submission.columns[1:]
    # Label encoding
    train = pd.concat([train, pd.get_dummies(train["primary_label"])], axis=1)
    # Fix order
    new_columns = list(train.columns.difference(birds)) + list(birds)
    train = train.reindex(columns=new_columns)

    # Search minor classes
    counts = train.primary_label.value_counts()
    # Condition that selects classes with less than `thr` samples
    cond = train.primary_label.isin(counts[counts<5].index.tolist())
    # Retrieve included train index
    no_cv = train[cond]["primary_label"].duplicated()
    no_cv_idx = train[cond][~no_cv].index
    # Add a new column to select samples for cross validation
    train.insert(0, "cv", True)
    # Set cv = False for those class where there is samples less than thr
    train.loc[no_cv_idx, "cv"] = False

    # Split X/y
    train_X = train.iloc[:, :-264]
    # Secondary label
    if cfg["audio"]["second_label"]:
        # Secondary label encoding
        for idx, each_secondary_labels in enumerate(train["secondary_labels"]):
            for secondary_label in each_secondary_labels:
                for bird in birds:
                    if secondary_label == bird:
                        train.loc[idx, bird] = 0.5
    train_y = train.iloc[:, -264:]
    cfg["model"]["num_classes"] = len(train_y.columns)

    if cfg["general"]["cv"]:
        skf = StratifiedKFold(
            n_splits=cfg["general"]["n_splits"],
            shuffle=True,
            random_state=cfg["general"]["seed"],
        )
        valid_cmap_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n

            if args.resume_train and os.path.isfile(
                f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}.ckpt"
            ):
                print("resume train")
                cfg[
                    "ckpt_path"
                ] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}.ckpt"
            else:
                cfg["ckpt_path"] = None

            _, valid_cmap = one_fold(skf, cfg, train_X, train_y, fold_n)
            valid_cmap_list.append(valid_cmap)

        valid_cmap_mean = np.mean(valid_cmap_list, axis=0)
        print(f"cv mean pfbeta:{valid_cmap_mean}")
    else:
        # train all data
        cfg["fold_n"] = "all"

        if args.resume_train and os.path.isfile(
            f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch.ckpt"
        ):
            print("resume train")
            cfg[
                "ckpt_path"
            ] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch.ckpt"
        else:
            cfg["ckpt_path"] = None

        all_train(cfg, train_X, train_y)


if __name__ == "__main__":
    main()
