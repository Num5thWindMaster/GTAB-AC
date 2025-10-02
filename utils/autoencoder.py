# -*- coding: utf-8 -*-
# @Time    : 2025/9/16 13:00
# @Author  : HaiqingSun
# @OriginalFileName: step7_4_1_autoencoder
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

ab_seq_path = "../data_ext/raw/dataset_4_seq.csv"
ab_seq_df = pd.read_csv(ab_seq_path)
seq_ids = ab_seq_df.columns.tolist()[1:]
pathway_path = "../data_ext/raw/KO_predicted.tsv"
pathway_df = pd.read_csv(pathway_path, sep="\t")
keys = pathway_df['sequence'].tolist()
values = pathway_df.iloc[:, 1:]
id_col = ab_seq_df.columns[0]
ab_cols_set = set(ab_seq_df.columns[1:])
keys_set = set(keys)
cols_to_drop = ab_cols_set - keys_set
ab_seq_filtered = ab_seq_df.drop(columns=cols_to_drop)
ordered_cols = [id_col] + [k for k in keys if k in ab_seq_filtered.columns]
ab_seq_filtered = ab_seq_filtered[ordered_cols]
ab_seq_filtered.to_csv("../data_ext/raw/abundance_4_seq.csv", index=False)

pt_dict = {}
for seq_id, row in zip(keys, values.itertuples(index=False, name=None)):
    pt_dict[seq_id] = torch.tensor(row, dtype=torch.int8)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
raw_emb_name = sys.argv[1]
torch.save(pt_dict, raw_emb_name)
class Autoencoder(nn.Module):
    def __init__(self, input_dim=10000, latent_dim=300, dropout=0.2):
        super(Autoencoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1024, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 1024),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

emb_path = Path(raw_emb_name)
pathway_emb_dict = torch.load(emb_path)
seq_ids = list(pathway_emb_dict.keys())
features = torch.stack([pathway_emb_dict[k] for k in seq_ids]).float()  # shape (N, input_dim)
features = features.to(device)
input_dim = features.shape[1]
latent_dim = 200

# -------------------------------
# 3. Constructing
# -------------------------------
model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim, dropout=0.2).to(device)

# -------------------------------
# 4. Training configs
# -------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
batch_size = 32

dataset = torch.utils.data.TensorDataset(features)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# 5. Autoencoder
# -------------------------------
model.train()
for epoch in range(100):
    epoch_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        x_recon, _ = model(x_batch)
        loss = criterion(x_recon, x_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataset):.6f}")

# -------------------------------
# 6. Save low dim embedding
# -------------------------------
model.eval()
with torch.no_grad():
    _, z = model(features)
    pathway_emb_lowdim = {seq_id: z[i].cpu() for i, seq_id in enumerate(seq_ids)}
output_name = raw_emb_name.replace(".pt", "_lowdim.pt")
state_dict_name = raw_emb_name.replace(".pt", "_autoencoder.pt")
torch.save(pathway_emb_lowdim, output_name)
torch.save(model.state_dict(), state_dict_name)
print("Low dim embedding saved to: xxx_lowdim.pt")