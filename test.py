import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import pdb
import cv2
import os
names = [
     'person', 'bicycle', 'car', 'motorcycle', 'airplane',
     'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
     'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
     'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
     'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
     'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
annot_path="/home/user/home/user/Xinyuan/work/CenterNet-1/data/coco/annotations/instances_val2017.json"
coco=coco.COCO(annot_path)
save_dir="/home/user/home/user/Xinyuan/work/CenterNet-1/exp/ctdet/coco_hg/"
coco_dets=coco.loadRes('{}/results.json'.format(save_dir))
coco_eval=COCOeval(coco, coco_dets, "bbox")
print(dir(coco_eval))
save_path = "src/cache/fail/"
coco_dets = coco.loadRes('{}/results.json'.format(save_dir))
coco_eval = COCOeval(coco, coco_dets, "bbox")
p=coco_eval.params
p.imgIds = list(np.unique(p.imgIds))
p.maxDets = sorted(p.maxDets)
coco_eval.params=p
coco_eval._prepare()
cat_dict = coco.cats
# print(cat_dict)
catIds = coco_eval.params.catIds
computeIoU = coco_eval.computeIoU
coco_eval.ious = {(imgId, catId): coco_eval.computeIoU(imgId, catId) \
            for imgId in p.imgIds
            for catId in catIds}
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
        # result = coco_eval.evaluateImg(imgId, catId, areaRng, maxDet)
        # pdb.set_trace()
        result = coco_eval.getImageFailCase(imgId, catId, areaRng, maxDet)
        if result is None:
            continue
        print(imgId)

        gts = result['gtFailBox']
        dts = result['dtFailBox']
        # pdb.set_trace()
        c = [255, 0, 0]
        for dt in dts:
            bbox = np.array(dt, dtype=np.int32)
            cv2.rectangle(
                img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), c, 2)
            cv2.rectangle(img,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
            cv2.putText(img, txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        c = [0, 255, 0]
        for gt in gts:
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