import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import bird2023model, bird2023sedmodel, dataset

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


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth_eps=0.0025, weight=None, reduction="mean"):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smooth_eps = smooth_eps
        self.weight = weight
        self.reduction = reduction
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(weight=self.weight, reduction=self.reduction)

    def forward(self, input, target):
        target_smooth = torch.clamp(target.float(), self.smooth_eps, 1.0 - self.smooth_eps)
        target_smooth = target_smooth + (self.smooth_eps / target.size(1))
        return self.bce_with_logits_loss(input, target_smooth)


class Bird2023Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        if cfg["model"]["pretrained_path"]:
            tmp_num_classes = cfg["model"]["num_classes"]
            cfg["model"]["num_classes"] = cfg["model"]["pretrained_classes"]
            sed_model = bird2023sedmodel.Bird2023SEDModel(cfg)
            sed_model = sed_model.load_from_checkpoint(
                checkpoint_path=cfg["model"]["pretrained_path"],
                cfg=cfg
            )
            sed_model.set_head(tmp_num_classes)
            cfg["model"]["num_classes"] = tmp_num_classes
            
            base_model = timm.create_model(
                model_name=cfg["model"]["model_name"],
                pretrained=cfg["model"]["pretrained"],
                in_chans=cfg["model"]["in_chans"],
                num_classes=cfg["model"]["num_classes"],
                drop_rate=cfg["model"]["drop_rate"],
                drop_path_rate=cfg["model"]["drop_path_rate"],
            )
            
            if "eca_nfnet" in self.cfg["model"]["model_name"]:
                if cfg["model"]["avg_and_max"]:
                    in_features = base_model.head.fc.in_features
                    fc = nn.Linear(in_features, cfg["model"]["num_classes"], bias=True)
                    drop = nn.Dropout(p=cfg["model"]["drop_rate"])
                    self.classifier = nn.Sequential(fc, drop)
                    layers = list(sed_model.encoder)
                else:
                    layers = list(sed_model.encoder) + list(base_model.children())[-1:]
            else:
                if cfg["model"]["avg_and_max"]:
                    in_features = base_model.classifier.in_features
                    fc = nn.Linear(in_features, cfg["model"]["num_classes"], bias=True)
                    drop = nn.Dropout(p=cfg["model"]["drop_rate"])
                    self.classifier = nn.Sequential(fc, drop)
                    layers = list(sed_model.encoder)
                else:
                    layers = list(sed_model.encoder) + list(base_model.children())[-2:]
            
            self.model = nn.Sequential(*layers)
            del sed_model, base_model
            torch.cuda.empty_cache()
        else:
            self.model = timm.create_model(
                model_name=cfg["model"]["model_name"],
                pretrained=cfg["model"]["pretrained"],
                in_chans=cfg["model"]["in_chans"],
                num_classes=cfg["model"]["num_classes"],
                drop_rate=cfg["model"]["drop_rate"],
                drop_path_rate=cfg["model"]["drop_path_rate"],
            )
            # avg_and_max は未実装

        if cfg["model"]["criterion"] == "bce_smooth":
            self.criterion = LabelSmoothingBCEWithLogitsLoss()
        else:
            self.criterion = nn.__dict__[cfg["model"]["criterion"]]()

        if cfg["model"]["grad_checkpointing"]:
            print("grad_checkpointing true")
            self.model.set_grad_checkpointing(enable=True)
               

    def forward(self, X):
        outputs = self.model(X)
        
        if self.cfg["model"]["avg_and_max"]:
            outputs = F.adaptive_max_pool2d(outputs, 1) + F.adaptive_avg_pool2d(outputs, 1)
            outputs = outputs[:, :, 0, 0]
            outputs = self.classifier(outputs)
        
        return outputs

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, x, y, alpha=1.0):
        indices = torch.randperm(x.size(0))
        shuffled_data = x[indices]
        shuffled_target = y[indices]

        lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        new_data = x.clone()
        new_data[:, :, bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        return new_data, y, shuffled_target, lam

    def mixup_data(self, x, y, alpha=1.0, return_index=False):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        if return_index:
            return mixed_x, y_a, y_b, lam, index
        else:
            return mixed_x, y_a, y_b, lam

    def mix_criterion(self, pred, y_a, y_b, lam, criterion="default"):
        if criterion == "default":
            criterion = self.criterion
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def training_step(self, batch, batch_idx):
        if self.cfg["model"]["train_2nd"] and self.current_epoch >= (
            self.cfg["pl_params"]["max_epochs"] - self.cfg["model"]["epoch_2nd"]
        ):
            # 最後だけaugmentation切るとかする用
            self.cfg["model"]["aug_mix"] = False
        if self.cfg["model"]["aug_mix"] and torch.rand(1) < 0.5:
            X, y = batch
            if torch.rand(1) >= 0.5:
                mixed_X, y_a, y_b, lam = self.mixup_data(X, y, alpha=self.cfg["model"]["mixup_alpha"])
            else:
                mixed_X, y_a, y_b, lam = self.cutmix_data(X, y, alpha=self.cfg["model"]["cutmix_alpha"])
            # mixed_X, y_a, y_b, lam = self.mixup_data(X, y)
            pred_y = self.forward(mixed_X)
            loss = self.mix_criterion(pred_y, y_a, y_b, lam)
        else:
            X, y = batch
            pred_y = self.forward(X)
            loss = self.criterion(pred_y, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        loss_list = [x["loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X)
        loss = self.criterion(pred_y, y)
        pred_y = torch.sigmoid(pred_y)
        pred_y = torch.nan_to_num(pred_y)
        return {"valid_loss": loss, "preds": pred_y, "targets": y}

    def validation_epoch_end(self, outputs):
        loss_list = [x["valid_loss"] for x in outputs]
        preds = torch.cat([x["preds"] for x in outputs], dim=0).cpu().detach().numpy()
        targets = (
            torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
        )
        if self.cfg["audio"]["second_label"]:
            targets[targets < 1.0] = 0.0
        avg_loss = torch.stack(loss_list).mean()
        padded_cmap_score = padded_cmap(targets, preds)
        self.log("valid_avg_loss", avg_loss, prog_bar=True)
        self.log("valid_padded_cmap_score", padded_cmap_score, prog_bar=True)

        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        pred_y = self.forward(X)
        pred_y = torch.sigmoid(pred_y)
        pred_y = torch.nan_to_num(pred_y)

        return pred_y

    def configure_optimizers(self):
        optimizer = optim.__dict__[self.cfg["model"]["optimizer"]["name"]](
            self.parameters(), **self.cfg["model"]["optimizer"]["params"]
        )
        if self.cfg["model"]["scheduler"] is None:
            return [optimizer]
        else:
            if self.cfg["model"]["scheduler"]["name"] == "OneCycleLR":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
                    **self.cfg["model"]["scheduler"]["params"],
                )
                scheduler = {"scheduler": scheduler, "interval": "step"}
            else:
                scheduler = optim.lr_scheduler.__dict__[
                    self.cfg["model"]["scheduler"]["name"]
                ](optimizer, **self.cfg["model"]["scheduler"]["params"])
            return [optimizer], [scheduler]
