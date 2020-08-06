#!/bin/bash
python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --load_model ../exp/ctdet/coco_resdcn18/model_last.pth
