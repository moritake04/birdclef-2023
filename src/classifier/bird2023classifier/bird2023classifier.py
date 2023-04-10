import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from . import bird2023model, bird2023sedmodel, dataset


class Bird2023Classifier:
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        # Create datasets
        train_dataset = dataset.Bird2023Dataset(
            cfg, train_X, train_y, augmentation=True
        )

        # Create Data loader
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            **cfg["train_loader"],
        )

        # Create dataset and dataloader for valid
        if valid_X is None:
            self.valid_dataloader = None
        else:
            valid_dataset = dataset.Bird2023Dataset(
                cfg, valid_X, valid_y, augmentation=False
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                **cfg["valid_loader"],
            )

        # Create callback list for pytorch-lightning
        callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step")]

        # Define Early Stopping
        if cfg["model"]["early_stopping_patience"] is not None:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss",
                    patience=cfg["model"]["early_stopping_patience"],
                )
            )

        # Define model save method
        if cfg["model"]["model_save"]:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}",
                    filename=f"last_epoch_fold{cfg['fold_n']}"
                    if cfg["general"]["cv"]
                    else "last_epoch",
                    save_weights_only=cfg["model"]["save_weights_only"],
                )
            )

        # Create training logger
        logger = WandbLogger(
            project=cfg["general"]["project_name"],
            name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
            group=f"{cfg['general']['save_name']}_cv"
            if cfg["general"]["cv"]
            else f"{cfg['general']['save_name']}_all",
            job_type=cfg["job_type"],
            mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
            config=cfg,
        )

        # Create model and trainer
        if cfg["model"]["sed"]:
            self.model = bird2023sedmodel.Bird2023SEDModel(cfg)
        else:
            self.model = bird2023model.Bird2023Model(cfg)
        self.trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            **cfg["pl_params"],
        )

        self.cfg = cfg

    def train(self, weight_path=None):
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader,
            ckpt_path=weight_path,
        )

    def predict(self, test_X, weight_path=None):
        preds = []
        test_dataset = dataset.Bird2023Dataset(self.cfg, test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=weight_path
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds

    def load_weight(self, weight_path):
        self.model = self.model.load_from_checkpoint(
            checkpoint_path=weight_path,
            cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")


class Bird2023ClassifierInference:
    def __init__(self, cfg, weight_path=None):
        self.weight_path = weight_path
        self.cfg = cfg
        
        if cfg["model"]["sed"]:
            self.model = bird2023sedmodel.Bird2023SEDModel(self.cfg)
        else:
            self.model = bird2023model.Bird2023Model(self.cfg)
        self.trainer = Trainer(**self.cfg["pl_params"])

    def load_weight(self, weight_path):
        self.model = self.model.load_from_checkpoint(
            checkpoint_path=weight_path,
            cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")

    def predict(self, test_X):
        test_dataset = dataset.Bird2023Dataset(self.cfg, test_X, augmentation=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=self.weight_path
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()

        return preds

    def test_predict(self, ogg_name_list, sample_submission):
        sub_df = pd.DataFrame(columns=sample_submission.columns)
        test_dataset = dataset.Bird2023TestDataset(self.cfg, ogg_name_list)

        self.load_weight(self.weight_path)
        self.model.to("cpu")
        self.model.eval()

        for i, data in enumerate(test_dataset):
            preds = []
            ogg_name = ogg_name_list[i][:-4]
            for start in tqdm(
                range(0, len(data), self.cfg["test_loader"]["batch_size"])
            ):
                with torch.no_grad():
                    pred = self.model(
                        data[start : start + self.cfg["test_loader"]["batch_size"]]
                    )
                    pred = torch.sigmoid(pred).to("cpu")
                preds.append(pred)
            preds = torch.cat(preds)
            preds = preds.cpu().detach().numpy()
            row_ids = [f"{ogg_name}_{(i+1)*5}" for i in range(len(preds))]
            df = pd.DataFrame(columns=sample_submission.columns)
            df["row_id"] = row_ids
            df[df.columns[1:]] = preds
            sub_df = pd.concat([sub_df, df]).reset_index(drop=True)

        return preds
