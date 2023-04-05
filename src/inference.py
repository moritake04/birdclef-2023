import argparse
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import wandb
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

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
    parser.add_argument("mode", type=str, help="valid or test")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    parser.add_argument(
        "-s",
        "--save_preds",
        action="store_true",
        help="Whether to save the predicted value or not.",
    )
    parser.add_argument("-a", "--amp", action="store_false")
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


def one_fold_valid(skf, cfg, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(skf.split(train_X, train_X["primary_label"]))[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    model = bird2023classifier.Bird2023ClassifierInference(
        cfg, f"{cfg['ckpt_path']}.ckpt"
    )
    valid_preds = model.predict(valid_X_cv)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if cfg["save_preds"]:
        print("save_preds!")
        joblib.dump(
            valid_preds,
            f"{cfg['general']['output_path']}/preds/valid_{cfg['general']['seed']}_{cfg['general']['save_name']}_{fold_n}.preds",
            compress=3,
        )

    print(f"[fold_{fold_n}]")
    padded_cmap_score = padded_cmap(valid_y_cv.values, valid_preds)

    torch.cuda.empty_cache()
    wandb.finish()

    return valid_preds, padded_cmap_score


def one_fold_test(cfg, fold_n, ogg_name_list, sample_submission):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)

    model = bird2023classifier.Bird2023ClassifierInference(
        cfg, f"{cfg['ckpt_path']}.ckpt"
    )
    test_preds = model.test_predict(ogg_name_list, sample_submission)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return test_preds


def all_train_test(cfg, test_X):
    print("[all_train]")

    seed_everything(cfg["general"]["seed"], workers=True)

    model = bird2023classifier.Bird2023ClassifierInference(
        cfg, f"{cfg['ckpt_path']}.ckpt"
    )
    test_preds = model.predict(test_X)

    return test_preds


def main():
    # Read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")
    cfg["mode"] = args.mode
    cfg["save_preds"] = args.save_preds

    # Set amp
    if args.amp:
        print("fp16")
        cfg["pl_params"]["precision"] = 16
    else:
        print("fp32")
        cfg["pl_params"]["precision"] = 32

    # Set random seed
    seed_everything(cfg["general"]["seed"], workers=True)

    # Read csv
    train = pd.read_csv(f"{cfg['general']['input_path']}/train_metadata.csv")
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

    # Split X/y
    train_X = train.iloc[:, :-264]
    train_y = train.iloc[:, -264:]
    cfg["model"]["num_classes"] = len(train_y.columns)

    if cfg["general"]["cv"]:
        if cfg["mode"] == "valid":
            skf = StratifiedKFold(
                n_splits=cfg["general"]["n_splits"],
                shuffle=True,
                random_state=cfg["general"]["seed"],
            )
            valid_cmap_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                cfg[
                    "ckpt_path"
                ] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                _, valid_cmap = one_fold_valid(skf, cfg, train_X, train_y, fold_n)
                valid_cmap_list.append(valid_cmap)

            valid_cmap_mean = np.mean(valid_cmap_list, axis=0)
            print(f"cv mean pfbeta:{valid_cmap_mean}")

        elif cfg["mode"] == "test":
            datadir = Path(f"{cfg['general']['input_path']}/test_soundscapes/")
            all_audios = list(datadir.glob("*.ogg"))
            ogg_name_list = [ogg_name.name for ogg_name in all_audios]

            test_preds_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                cfg[
                    "ckpt_path"
                ] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                test_preds = one_fold_test(
                    cfg, fold_n, ogg_name_list, sample_submission
                )
                test_preds_list.append(test_preds)
                print(test_preds)

            final_test_preds = np.mean(test_preds_list, axis=0)
            print(final_test_preds)

    else:
        cfg["fold_n"] = "all"
        cfg[
            "ckpt_path"
        ] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch"
        final_test_preds = all_train_test(cfg, test_X)
        print(final_test_preds)


if __name__ == "__main__":
    main()
