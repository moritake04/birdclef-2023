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

from classifier import bird2023classifier


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
    xeno_am = pd.read_csv(
        f"{cfg['general']['input_path']}/xenocanto/train_extended.csv"
    )[:14684]
    xeno_nz = pd.read_csv(
        f"{cfg['general']['input_path']}/xenocanto/train_extended.csv"
    )[14685:]
    bird2020 = pd.read_csv(f"{cfg['general']['input_path']}/bird2020/train.csv")
    bird2021 = pd.read_csv(
        f"{cfg['general']['input_path']}/bird2021/train_metadata.csv"
    )
    bird2022 = pd.read_csv(
        f"{cfg['general']['input_path']}/bird2022/train_metadata.csv"
    )
    bird2023 = pd.read_csv(f"{cfg['general']['input_path']}/train_metadata.csv")
    sample_submission = pd.read_csv(
        f"{cfg['general']['input_path']}/sample_submission.csv"
    )
    # Set common columns
    xeno_am["birdclef"] = "xeno"
    xeno_nz["birdclef"] = "xeno"
    bird2020["birdclef"] = "2020"
    bird2021["birdclef"] = "2021"
    bird2022["birdclef"] = "2022"
    xeno_am["primary_label"] = xeno_am["ebird_code"]
    xeno_nz["primary_label"] = xeno_nz["ebird_code"]
    bird2020["primary_label"] = bird2020["ebird_code"]
    xeno_am["filepath"] = xeno_am["primary_label"] + "/" + xeno_am["filename"]
    xeno_nz["filepath"] = xeno_nz["primary_label"] + "/" + xeno_nz["filename"]
    bird2020["filepath"] = bird2020["primary_label"] + "/" + bird2020["filename"]
    bird2021["filepath"] = bird2021["primary_label"] + "/" + bird2021["filename"]
    bird2022["filepath"] = bird2022["filename"]
    bird2023["filepath"] = bird2023["filename"]
    # Set filiname and concat
    bird2023["filename"] = bird2023["filepath"].map(
        lambda x: x.split("/")[-1].split(".")[0]
    )
    train = pd.concat(
        [xeno_am, xeno_nz, bird2020, bird2021, bird2022], axis=0
    ).reset_index(drop=True)
    train["filepath"] = train["filepath"].map(lambda x: x.split(".")[0])
    train["filename"] = train["filepath"].map(lambda x: x.split("/")[-1])
    # Drop duplicates and set columns
    nodup_idx = train[["filename", "primary_label", "author"]].drop_duplicates().index
    train = train.loc[nodup_idx].reset_index(drop=True)
    train = train[~train["filename"].isin(bird2023["filename"])].reset_index(drop=True)
    train = train[
        ["filepath", "filename", "primary_label", "secondary_labels", "birdclef"]
    ]
    train["secondary_labels"] = train["secondary_labels"].apply(ast.literal_eval)
    # Retrieve target
    birds = sample_submission.columns[1:]
    # Label encoding
    train = pd.concat([train, pd.get_dummies(train["primary_label"])], axis=1)
    # Fix order
    new_columns = list(train.columns.difference(birds)) + list(birds)
    train = train.reindex(columns=new_columns)

    # Split X/y
    train_X = train.iloc[:, :-264]
    # Secondary label
    if cfg["audio"]["second_label"]:
        # Secondary label encoding
        for idx, each_secondary_labels in enumerate(train["secondary_labels"]):
            for secondary_label in each_secondary_labels:
                for bird in birds:
                    if secondary_label == bird:
                        train.loc[idx, bird] = 1
    train_y = train.iloc[:, -264:]
    cfg["model"]["num_classes"] = len(train_y.columns)

    # train all data
    cfg["fold_n"] = "all"

    if args.resume_train and os.path.isfile(
        f"{cfg['general']['output_path']}/pretrain/weights/cfg['general']['save_name']}/last_epoch.ckpt"
    ):
        print("resume train")
        cfg[
            "ckpt_path"
        ] = f"{cfg['general']['output_path']}/pretrain/weights/{cfg['general']['save_name']}/last_epoch.ckpt"
    else:
        cfg["ckpt_path"] = None

    all_train(cfg, train_X, train_y)


if __name__ == "__main__":
    main()
