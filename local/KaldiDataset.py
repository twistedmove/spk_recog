#!/usr/bin/env python3
# %%
import random
import subprocess
import torch
from kaldi_io import read_mat_scp
from torch.utils.data import Dataset


def _sel_offset(len_frame, chunk_size, num_chunk):
    if len(range(len_frame-chunk_size)) < num_chunk:
        print(len_frame, chunk_size, num_chunk)
    return random.sample(range(len_frame-chunk_size), num_chunk)


class ReadMatrix(Dataset):
    def __init__(self, datadir, chunk_size, num_chunk, batchsize):

        self.feats = f'{datadir}/feats.scp'
        self.utt2spk = {}
        self.spks = []
        with open(f'{datadir}/utt2spk', 'rt') as fp:
            for line in fp.readlines():
                parts = line.split()
                self.utt2spk[parts[0]] = parts[1]
                if parts[1] not in self.spks:
                    self.spks.append(parts[1])
        self.scp = iter(read_mat_scp(self.feats))

        self.chunk_size = chunk_size
        self.num_chunk = num_chunk
        self.batchsize = batchsize

    def __len__(self):
        length = int(subprocess.check_output(
            f'cat {self.feats} | wc -l', shell=True, encoding='utf8'))
        return length // self.batchsize

    def __getitem__(self, idx):
        features = []
        labels = []
        for i in range(self.batchsize):
            uttid, np_arr = next(self.scp)

            if np_arr.shape[0] < self.chunk_size * (self.num_chunk // 2):
                uttid, np_arr = next(self.scp)

            offsets = _sel_offset(
                np_arr.shape[0], self.chunk_size, self.num_chunk)

            for j in offsets:
                features.append(torch.from_numpy(
                    np_arr[j: j + self.chunk_size]))
                labels.append(self.spks.index(self.utt2spk[uttid]))
        data = list(zip(features, labels))
        random.shuffle(data)
        x, y = zip(*data)

        return torch.stack(x), torch.tensor(y)
