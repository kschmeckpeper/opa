# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
import pycocotools
import argparse


#######################
# Script for pretraining Mask RCNN on the pseudo-gt
# Adapted from: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
#######################


def get_cubes_dicts(img_dir):
    annot_file = os.path.join(img_dir, "annot.npy")
    annot = np.load(annot_file, allow_pickle=True).item()

    dataset_dicts = []
    for k in annot.keys():
        v = annot[k]
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = v["image_id"]
        record['width'] = width
        record['height'] = height

        bboxes = v['bboxes']
        masks = v['masks']

        objs=[]
        for i in range(bboxes.shape[0]):
            box = bboxes[i,:] # x1,y1,x2,y2
            mask = masks[i,:,:] # need to convert mask
            x1,y1,x2,y2 = box
            
            # either convert mask to appropriate format or get vertices of box for polygon annotation
            mask_dict = pycocotools.mask.encode(np.asarray(mask.copy(), order="F"))
            
            obj = {
                "bbox": [x1,y1,x2,y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": mask_dict,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['eval', 'train'])
args = parser.parse_args()

data_path = 'data/cubes/'

for d in ["train", "val"]:
    DatasetCatalog.register("cubes_" + d, lambda d=d: get_cubes_dicts(data_path + d))
    MetadataCatalog.get("cubes_" + d).set(thing_classes=["cubes"])
cubes_metadata = MetadataCatalog.get("cubes_train")

dataset_dicts = get_cubes_dicts(data_path+"train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cubes_metadata, scale=2)
    out = visualizer.draw_dataset_dict(d)

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cubes_train",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cube). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.INPUT.MASK_FORMAT = 'bitmask'

if args.mode=="train":
    # do training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

else:
    # do inference on val set
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.DATASETS.TEST = ('cubes_val')
    model_path = "../configs/"
    cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    count_test=0
    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_cubes_dicts(data_path+"val")
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=cubes_metadata, 
                    scale=2, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out.save("test_"+str(count_test)+".png")
        count_test+=1