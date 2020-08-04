#!/bin/bash
python test.py ctdet --exp_id coco_hg --arch hourglass --fix_res --load_model ../models/ctdet_coco_hg.pth --debug 2 --not_prefetch_test
