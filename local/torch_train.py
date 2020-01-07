#!/usr/bin/env python3
# %%
import os
import math
import random
import torch
from torch.utils.data import DataLoader
from KaldiDataset import ReadMatrix
from torch_model import Xvector

random.seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"#### Train with ( {device} )")


def _random_chunk_size(min_size, max_size, num_iter):
    return random.sample(range(min_size, max_size), num_iter)


datadir = 'data/swbd_cell_no_sil'
mdldir = 'torch_models/'
os.makedirs(mdldir, exist_ok=True)

feat_dim = 23
num_targets = 643

num_iter = 20
num_chunk_per_feat = 10
num_feat_per_step = 5
batchsize = 8

initial_lr = 0.01
final_lr = 0.001
gamma = math.exp(math.log(final_lr / initial_lr) / num_iter)

min_chunk_size = 200
max_chunk_size = 400

chunk_list = _random_chunk_size(min_chunk_size, max_chunk_size, num_iter)

model = Xvector(feat_dim, num_targets).to(device=device)
model = torch.nn.DataParallel(model)
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


for i, chunk_size in enumerate(chunk_list):
    iter_loss = 0
    iter_acc = 0

    train_data = ReadMatrix(datadir, chunk_size,
                            num_chunk_per_feat, num_feat_per_step)
    train_data = DataLoader(train_data, batch_size=batchsize, shuffle=True)

    for j, data in enumerate(train_data):
        optimizer.zero_grad()

        train_x = data[0].to(device).view(-1, chunk_size, feat_dim)
        train_x = train_x.transpose(1, 2)
        train_y = data[1].view(-1).to(device)

        logit = model(train_x)
        loss = criterion(logit, train_y)
        pred = logit.max(dim=-1)[1]
        loss.backward()
        optimizer.step()

        e_loss = loss.item()
        e_acc = torch.eq(train_y, pred).sum()

        iter_loss += e_loss
        iter_acc += e_acc

        if j % 10 == 0:
            print(
                f"{j:3d}/{len(train_data)}] loss: {e_loss:.4f}, acc {e_acc}")
    print(f"[{i}]Train Loss:{iter_loss:.5f}, Train Acc:{iter_acc:5d}\n")

    scheduler.step()
    torch.save(model, f'{mdldir}/model_{i + 1}.pth')
