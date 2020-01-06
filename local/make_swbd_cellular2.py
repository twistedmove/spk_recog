#!/usr/bin/env python3
# Based on kaldi/egs/sre08/v1/local/make_swbd_cellular2.pl

import os
import shutil
import sys
import subprocess


if len(sys.argv) != 3:
    raise Exception("Usage: {0} <path-to-LDC2004S07> <path-to-output>".format(os.path.basename(__file__)))

db_base, outdir = str(sys.argv[1]), str(sys.argv[2])

subprocess.call('mkdir -p ' + outdir, shell=True)

tmp_dir = 'data/tmp_cell2'
subprocess.call('mkdir -p ' + tmp_dir, shell=True)
subprocess.call('find {0} -name \'*.sph\' > {1}/sph.list'.format(db_base, tmp_dir), shell=True)

TRIALS = open(tmp_dir + '/sph.list', 'r')

line = TRIALS.readlines()
wav_dict = {}
for i in range(len(line)):
    raw_basename = line[i].split('/')[-1][:-1].replace('.sph', '')
    wav_dict[raw_basename] = line[i][:-1]
TRIALS.close()

CS = open(db_base + '/swbcell2_01/docs/swb_callstats.tbl', 'r')
GNDR = open(outdir + '/spk2gender', 'w')
SPKR = open(outdir + '/utt2spk', 'w')
WAV = open(outdir + '/wav.scp', 'w')

badAudio = ['45024', '40022']

s2g_dict = {}
line = CS.readlines()
for i in range(len(line)):
    A = line[i][:-1].split(',')
    if A[0] in badAudio:
        continue
    else:
        wav = 'sw_' + A[0]
        spk1 = 'swc' + A[1]
        spk2 = 'swc' + A[2]
        gender1 = A[3].lower()
        gender2 = A[4].lower()
        if gender1 not in ['f', 'm']:
            print('unknown gender', A[3])
            continue
        if gender2 not in ['f', 'm']:
            print('unknown gender', A[4])
            continue

        if wav in wav_dict:
            uttid = spk1 + '-swbdc_' + wav + '_1'
            if not spk1 in s2g_dict:
                s2g_dict[spk1] = gender1
                GNDR.write('{0} {1}\n'.format(spk1, gender1))
            WAV.write('{0} sph2pipe -f wav -p -c 1 {1} |\n'.format(uttid, wav_dict[wav]))
            SPKR.write('{0} {1}\n'.format(uttid, spk1))

            uttid = spk2 + '-swbdc_' + wav + '_2'
            if not spk2 in s2g_dict:
                s2g_dict[spk2] = gender2
                GNDR.write('{0} {1}\n'.format(spk2, gender2))
            WAV.write('{0} sph2pipe -f wav -p -c 1 {1} |\n'.format(uttid, wav_dict[wav]))
            SPKR.write('{0} {1}\n'.format(uttid, spk2))
        else:
            print('Missing ', wav_dict[wav])
CS.close()
GNDR.close()
SPKR.close()
WAV.close()

subprocess.call('utils/utt2spk_to_spk2utt.pl {0}/utt2spk >{0}/spk2utt'.format(outdir), shell=True)
subprocess.call('utils/fix_data_dir.sh {0}'.format(outdir), shell=True)
subprocess.call('utils/validate_data_dir.sh --no-text --no-feats {0}'.format(outdir), shell=True)
shutil.rmtree(tmp_dir)
#
