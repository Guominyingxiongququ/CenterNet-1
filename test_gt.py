import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import pdb
import cv2
import os
annot_path="/home/user/home/user/Xinyuan/work/CenterNet-1/data/coco/annotations/instances_val2017.json"
coco=coco.COCO(annot_path)
save_dir="/home/user/home/user/Xinyuan/work/CenterNet-1/exp/ctdet/coco_hg/"
coco_dets=coco.loadRes('{}/results.json'.format(save_dir))
coco_eval=COCOeval(coco, coco_dets, "bbox")
print(dir(coco_eval))
save_path = "src/cache/gt/"
p=coco_eval.params
p.imgIds = list(np.unique(p.imgIds))
p.maxDets = sorted(p.maxDets)
coco_eval.params=p
coco_eval._prepare()
cat_dict = coco.cats
# print(cat_dict)
catIds = coco_eval.params.catIds
maxDet = p.maxDets[-1]
for imgId in p.imgIds:
    img_path = "/home/user/home/user/Xinyuan/work/CenterNet-1/data/coco/val2017/"
    fullImgId = str(imgId).zfill(12)
    img_path = os.path.join(img_path, fullImgId+'.jpg')
    print(img_path)
    img = cv2.imread(img_path)
    for catId in catIds:
        cat = int(catId)
        txt = cat_dict[cat]['name']
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        areaRng = p.areaRng[0]
        # pdb.set_trace()
        print(imgId)
        gts = coco_eval._gts[imgId,catId]
        gtIgnore = [g['ignore']==0 for g in gts]
        gtBox = [g['bbox'] for g in gts]
        gtBox = np.array(gtBox)
        gtBox = gtBox[gtIgnore]
        print(gtBox.shape)
        c = [0, 255, 0]
        for gt in gtBox:
            print(gt)
            bbox = np.array(gt, dtype=np.int32)
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), c, 2)
            cv2.rectangle(img,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    # saveImage
    save_path1 = os.path.join(save_path, fullImgId+'.jpg')
    print("save_path: "+save_path1)
    cv2.imwrite(save_path1, img)