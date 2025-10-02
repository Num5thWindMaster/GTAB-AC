# -*- coding: utf-8 -*-
# @Time    : 2025/6/4 19:36
# @Author  : HaiqingSun
# @OriginalFileName: pl_trainer
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import os
import random

import numpy as np
import pytorch_lightning
import torch
import torch_geometric
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split, KFold
from torch import nn
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

from pl_module import GTABACLightningModule
from dataset import SpeciesNodePETabularDataset
from utils.utils import cal_r2_score, ci, mae

base_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed=6):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pytorch_lightning.seed_everything(seed)
    np.random.seed(seed)
    torch_geometric.seed_everything(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


batch_size = 32
seed = 16
set_seed(seed)


def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)


graph_pe_encoding_dim = 32
transform = AddRandomWalkPE(walk_length=graph_pe_encoding_dim, attr_name='pe_enc')
# dataset = SpeciesNodePETabularDataset(os.path.join(base_dir, 'data_ext'), pre_transform=transform)
dataset = SpeciesNodePETabularDataset(
    age_label='age',
    id_label='uid',
    root=os.path.join(base_dir, 'data_ext'),
    mode='all',
    edge_acc_emb=True,
    p_threshold=0.1,
    n_threshold=-0.1,
    pre_transform=transform,
)
indices = list(range(len(dataset)))

train_valid_idx, test_idx = train_test_split(
    indices, test_size=0.15, random_state=seed, shuffle=True
)

test_dataset = dataset[test_idx]
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
input_dim = dataset[0].abundance.shape[0]
num_features_xnode = 200
graph_output_dim = 800
fold_epochs = 80
LOG_INTERVAL = 10
clip_value = 1  # 1 0.5
lambda_sparse = 9e-4
lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
all_fold_maes = []
current_fold = 0
best_test_mae = best_test_r2 = best_test_ci = 0
saved_models_path = 'saved_models'
if not os.path.exists(saved_models_path):
    os.mkdir(saved_models_path)
model_file_name = 'saved_models/tabnet_model_exp_e80l1e-4s16'

model = GTABACLightningModule(input_dim=input_dim,
                              num_features_xnode=num_features_xnode,
                              graph_output_dim=graph_output_dim,
                              lambda_sparse=lambda_sparse,
                              lr=lr,
                              loss_fn=nn.SmoothL1Loss(beta=1.0),
                              graph_pe_encoding=graph_pe_encoding_dim)
checkpoint_callback = None
for train_idx, valid_idx in kf.split(train_valid_idx):
    current_fold += 1
    print(f"=== Fold {current_fold} ===")
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
    checkpoint_callback = ModelCheckpoint(monitor="val_mae", mode="min",
                                          save_top_k=1, save_weights_only=True,
                                          filename="best-checkpoint")
    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        mode='min',
        patience=20,
        verbose=True,
    )
    logger = CSVLogger("lightning_logs", name=f"fold_{current_fold}")

    trainer = Trainer(
        max_epochs=fold_epochs,
        accelerator="gpu",
        devices=1,
        logger=False,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

torch.save(model.state_dict(), model_file_name + '.pth')

# For no cross validation training
# best_model = TabNetLightningModule.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path,
#                                                         input_dim=input_dim,
#                                                         graph_hidden_dim=graph_hidden_dim,
#                                                         graph_output_dim=graph_output_dim,
#                                                         lambda_sparse=lambda_sparse,
#                                                         lr=lr,
#                                                         loss_fn=nn.SmoothL1Loss(beta=1.0),
#                                                         graph_pe_encoding=graph_pe_encoding_dim
#                                                         )
# best_model.eval()
# total_preds, total_labels = torch.Tensor(), torch.Tensor()
# total_test_preds, total_test_labels = torch.Tensor(), torch.Tensor()
#
# with torch.no_grad():
#     for batch in test_loader:
#         best_model = best_model.to(device)
#         batch = batch.to(device)
#         preds, _ = best_model(batch)
#         total_preds = torch.cat((total_preds, preds.cpu().squeeze()), 0)
#         total_labels = torch.cat((total_labels, batch.y.cpu().squeeze()), 0)
#
# P, G = total_preds.numpy().flatten(), total_labels.numpy().flatten()
# val_metric = mae(G, P)
# best_test_mae = val_metric
# best_test_r2 = cal_r2_score(G, P)[0]
# best_test_ci = ci(G, P)
# torch.save(best_model.state_dict(), model_file_name + '_best.pth')
#
# print('\n========== Cross-Validation Result ==========')
# print('Best test MAE:', best_test_mae)
# print('Best test R2:', best_test_r2)
# print('Best test CI:', best_test_ci)
