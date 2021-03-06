#!/bin/bash
. ./path.sh
stage=0

if [ $stage -le 0 ]; then
  local/make_swbd_cellular1.py \
    ~/Database/Switchboard/SWBDcell1_LDC2001S13 data/swbd_cellular1_train
  local/make_swbd_cellular2.py \
    ~/Database/Switchboard/SWBDcell2_LDC2004S07 data/swbd_cellular2_train
  
  for dir in swbd_cellular1_train swbd_cellular2_train; do
    utils/utt2spk_to_spk2utt.pl data/$dir/utt2spk > data/$dir/spk2utt
    utils/fix_data_dir.sh data/$dir
    utils/validate_data_dir.sh --no-text --no-feats data/$dir
  done

  utils/combine_data.sh data/swbd_cell data/swbd_cellular*_train
  utils/fix_data_dir.sh data/swbd_cell
fi

if [ $stage -le 1 ]; then
  # local/torch_make_fbank.py data/swbd_cell ~/Workspace/spk_recog/fbank
  export PATH=../../kaldi/tools/sph2pipe_v2.5:$PATH
  dir=~/Workspace/spk_recog/fbank
  mkdir -p $dir/log
  # steps/make_fbank.sh --nj 10 data/swbd_cell $dir/log $dir
  local/compute_vad_decision.sh --nj 10 data/swbd_cell $dir/log $dir
fi

if [ $stage -le 2 ]; then
  local/prepare_feats_for_egs.sh --nj 40 \
    data/swbd_cell data/swbd_cell_no_sil ~/Workspace/spk_recog/swbd_cell_no_sil
  utils/fix_data_dir.sh data/swbd_cell_no_sil
  if [ ! -e data/swbd_cell_no_sil/.nosil.scp ]; then
    mv data/swbd_cell_no_sil/feats.scp data/swbd_cell_no_sil/.nosil.scp
  fi
  utils/shuffle_list.pl --srand 123 data/swbd_cell_no_sil/.nosil.scp \
    > data/swbd_cell_no_sil/feats.scp
fi
exit 0;