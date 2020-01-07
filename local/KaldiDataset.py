#!/usr/bin/env python3
# %%
import os
import sys
import subprocess
import torch
from kaldi_io import read_mat_scp
from torch.utils.data import Dataset


class ReadMatrix(Dataset):
    def __init__(self, feats):
        if not os.path.isfile(feats):
            sys.exit(f"Can't find the {feats}")

        self.feats = feats
        self.scp = iter(read_mat_scp(self.feats))

    def __len__(self):
        return int(subprocess.check_output(
            f'cat {self.feats} | wc -l', shell=True, encoding='utf8'))

    def __getitem__(self, idx):
        key, np_arr = next(self.scp)
        return key, torch.from_numpy(np_arr)
