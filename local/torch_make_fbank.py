#!/usr/bin/env python3
#%%
import os
import sys
import torch
import torchaudio
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)        # 1.3.0
print(torchaudio.__version__)   # 0.3.1
print(device)                   # cuda

# %%
data = sys.argv[1]
fbank_dir = sys.argv[2]
home = os.path.expanduser("~")
# data = 'data/swbd_cell'
# fbank_dir = f'{home}/Workspace/spk_recog/fbank'

os.makedirs(f'{fbank_dir}/log', exist_ok=True)

name = os.path.basename(data)
scp  = os.path.join(data, 'wav.scp')

sample_frequency=8000
frame_length=20 # the default is 25.
low_freq=20 # the default.
high_freq=3700 # the default is zero meaning use the Nyquist (4k in this case).

feats = open(f'{data}/feats.scp', 'wt')

with open(scp, 'rt') as fp:
    for line in fp.readlines():
        parts = line.split()
        wav, _ = torchaudio.load_wav(parts[1])
        torch.save(
            torchaudio.compliance.kaldi.fbank(wav, sample_frequency=sample_frequency, 
                low_freq=low_freq, high_freq=high_freq), 
            f'{fbank_dir}/{parts[0]}.pt')
        feats.write(f'{parts[0]} {fbank_dir}/{parts[0]}.pt\n')

feats.close()
print("Done")