#!/bin/bash
. ./path.sh

local/make_swbd_cellular1.py \
  ~/Database/Switchboard/SWBDcell1_LDC2001S13 data/swbd_cellular1_train
local/make_swbd_cellular2.py \
  ~/Database/Switchboard/SWBDcell2_LDC2004S07 data/swbd_cellular2_train

utils/combine_data.sh data/swbd_cell data/swbd_cellular*_train
utils/fix_data_dir.sh data/swbd_cell

exit 0;
