# -*- coding: utf-8 -*-
# @Time    : 2025/6/4 19:36
# @Author  : HaiqingSun
# @OriginalFileName: pl_module
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from network.tab_network import TabNet
from utils.utils import mae, cal_r2_score, ci, visualize_and_save_masks


class TabNetLightningModule(pl.LightningModule):
    def __init__(self, input_dim, graph_hidden_dim, graph_output_dim, lambda_sparse, lr, loss_fn, graph_pe_encoding,
                 clip_value=1):
        super().__init__()
        self.model = TabNet(input_dim, 1,
                            n_d=28, n_a=22, n_steps=3, gamma=1.1,
                            n_independent=4, n_shared=5, mask_type='entmax',
                            graph_branch=False,
                            hidden_dim=graph_hidden_dim,
                            graph_output_dim=graph_output_dim,
                            graph_pe_encoding=graph_pe_encoding)
        self.lambda_sparse = lambda_sparse
        self.loss_fn = loss_fn
        self.lr = lr
        self.clip_value = clip_value

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        labels = batch.y
        preds, M_loss = self(batch)
        loss = self.loss_fn(preds.squeeze(), labels)
        loss = loss - self.lambda_sparse * M_loss
        if self.clip_value:
            clip_grad_norm_(self.model.parameters(), self.clip_value)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_start(self):
        self.val_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        labels = batch.y
        preds, _ = self(batch)
        self.val_preds.append(preds.detach().cpu())
        self.val_labels.append(labels.detach().cpu())

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy().flatten()
        labels = torch.cat(self.val_labels).numpy().flatten()
        val_mae = mae(labels, preds)
        self.log("val_mae", val_mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_test_start(self):
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        labels = batch.y
        preds, _ = self(batch)
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).numpy().flatten()
        labels = torch.cat(self.test_labels).numpy().flatten()
        test_mae = mae(labels, preds)
        test_r2 = cal_r2_score(labels, preds)[0]
        test_ci = ci(labels, preds)
        self.log("test_mae", test_mae, prog_bar=False)
        self.log("test_r2", test_r2, prog_bar=False)
        self.log("test_ci", test_ci, prog_bar=False)

    def on_predict_start(self):
        self.predict_preds = []

    def predict_step(self, batch, batch_idx):
        preds, _ = self(batch)
        # with torch.no_grad():
        #     M_explain, masks = self.model.forward_masks(batch)
        #     visualize_and_save_masks(M_explain, masks)
        self.predict_preds.append(preds.detach().cpu())

    def on_predict_end(self):
        preds = torch.cat(self.predict_preds).numpy().flatten()
        np.savetxt("predictions_out.txt", preds, fmt="%.3f")