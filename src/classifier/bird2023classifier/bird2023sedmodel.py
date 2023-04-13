import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCE2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], base_loss="focal"):
        super().__init__()

        if base_loss == "focal":
            self.base_loss = BCEFocalLoss()
        elif base_loss == "bce":
            self.base_loss = nn.__dict__["BCEWithLogitsLoss"]()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.base_loss(input_, target)
        aux_loss = self.base_loss(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)

    return output


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class Bird2023SEDModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        base_model = timm.create_model(
            model_name=cfg["model"]["model_name"],
            pretrained=cfg["model"]["pretrained"],
            in_chans=cfg["model"]["in_chans"],
            # num_classes=cfg["model"]["num_classes"],
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
        )

        if cfg["model"]["grad_checkpointing"]:
            print("grad_checkpointing true")
            base_model.set_grad_checkpointing(enable=True)

        self.bn0 = nn.BatchNorm2d(cfg["mel_specgram"]["n_mels"])
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, cfg["model"]["num_classes"], activation="sigmoid"
        )
        self.init_weight()

        self.criterion = BCE2WayLoss(base_loss=cfg["model"]["base_criterion"])

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, X):
        # X: (batch_size, 3, mel_bins, time_steps)
        frames_num = X.shape[3]

        X = X.transpose(1, 2)
        X = self.bn0(X)
        X = X.transpose(1, 2)
        X = self.encoder(X)

        # Aggregate in frequency axis
        X = torch.mean(X, dim=2)

        X1 = F.max_pool1d(X, kernel_size=3, stride=1, padding=1)
        X2 = F.avg_pool1d(X, kernel_size=3, stride=1, padding=1)
        X = X1 + X2

        X = F.dropout(X, p=self.cfg["model"]["drop_rate"], training=self.training)
        X = X.transpose(1, 2)
        X = F.relu_(self.fc1(X))
        X = X.transpose(1, 2)
        X = F.dropout(X, p=self.cfg["model"]["drop_rate"], training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(X)
        logit = torch.sum(norm_att * self.att_block.cla(X), dim=2)
        segmentwise_logit = self.att_block.cla(X).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            "framewise_output": framewise_output,
            "segmentwise_output": segmentwise_output,
            "clipwise_output": clipwise_output,
            "logit": logit,
            "framewise_logit": framewise_logit,
        }

        return output_dict

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
        
        if self.cfg["model"]["pred_mode"] == "clipwise":
            pred_y = pred_y["clipwise_output"]
        elif self.cfg["model"]["pred_mode"] == "framewise_max":
            pred_y, _ = torch.max(pred_y["framewise_output"], dim=1)
        elif self.cfg["model"]["pred_mode"] == "framewise_mean":
            pred_y = torch.mean(pred_y["framewise_output"], dim=1)
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
        
        if self.cfg["model"]["pred_mode"] == "clipwise":
            pred_y = pred_y["clipwise_output"]
        elif self.cfg["model"]["pred_mode"] == "framewise_max":
            pred_y, _ = torch.max(pred_y["framewise_output"], dim=1)
        elif self.cfg["model"]["pred_mode"] == "framewise_mean":
            pred_y = torch.mean(pred_y["framewise_output"], dim=1)
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
