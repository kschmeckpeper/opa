import torch
import torch.nn as nn
import numpy as np
import math

from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import detector_postprocess

# Using Detectron2 model for InstanceSegmentation
class InstanceSegmentation(nn.Module):
    def __init__(self, model_params, model_weights):
        super(InstanceSegmentation, self).__init__()
        confidence_threshold = 0.05 # overwritten in forward
        cfg = self.setup_cfg(confidence_threshold, model_params, model_weights)
        self.model = build_model(cfg)
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)


    def setup_cfg(self, confidence_threshold, model_params, model_weights):
        cfg = get_cfg()
        cfg.merge_from_file(model_params)
        cfg.MODEL.WEIGHTS = model_weights
        # Set score_threshold for builtin models
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cube).
        cfg.freeze()
        return cfg


    def forward(self, input_imgs, gt_instances=None):
        # Input:
        #    input_imgs: Numpy array type np.uint8, Batch x H x W x C
        # Output:
        #    predictions: List of tuples [(feats, masks), (), ...], size Batch
        
        images, height, width = self.preprocess_imgs(imgs=input_imgs)
        features = self.model.backbone(images.tensor)
        proposals, rpn_losses = self.model.proposal_generator(images, features, gt_instances) # rpn_losses is empty when gt_instances=None

        if gt_instances is not None: # gt is available!
            roi_proposals, roi_losses = self.model.roi_heads(images, features, proposals, gt_instances)
            mask_losses = self.model.roi_heads._forward_mask_updated(features, roi_proposals)
            mask_rcnn_losses = {**rpn_losses, **roi_losses, **mask_losses}
        
        # run again inference only on fast-rcnn and mask heads to get the predictions
        # decrease the threshold if no instances were predicted, overwrites the config test_score_thresh
        for test_score_thresh in [0.7, 0.5, 0.3, 0.05, 0.0]:  
            #instances, _ = self.model.roi_heads(images, features, proposals, test_score_thresh=test_score_thresh)
            #init_res, mask_features = self.model.roi_heads._forward_mask_updated(features, instances)
            init_res, mask_features = self.model.roi_heads(images, features, proposals, test_score_thresh=test_score_thresh)
            no_pred = False
            for j in range(len(init_res)): # if any batch did not have predictions then no_pred=true
                if len(init_res[j])==0: # number of predicted instances
                    no_pred = True
            if no_pred is False:
                break

        predictions = []
        index = 0
        for i in range(len(init_res)): # batch_size
            num_instances = len(init_res[i])
            # keep the raw output of mask rcnn (float) N x 1 x 28 x 28
            raw_masks = init_res[i].pred_masks

            # normalize the raw_masks with max value -- to ensure that masks always have values > 0.5
            # affects the binary masks to be used later for the calculation of pseudo-gt
            raw_masks_norm = torch.zeros(raw_masks.size(), dtype=torch.float32, device=raw_masks.device)
            for j in range(raw_masks.shape[0]):
                max_value = torch.max(raw_masks[j,:,:,:])
                if max_value == 0: # contingency in the very rare case where all raw_mask values are 0
                    raw_masks_norm[j,:,:,:] = torch.ones(raw_masks[j,:,:,:].size(), dtype=torch.float32, device=raw_masks.device)
                else:
                    raw_masks_norm[j,:,:,:] = raw_masks[j,:,:,:].clone() / max_value
            init_res[i].pred_masks = raw_masks_norm # replace the original raw_masks
            
            # use detectron2 postprocessing functions to resize outputs and create binary masks
            results = detector_postprocess(init_res[i], height, width)
            # For results format see here:
            # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            masks = results.pred_masks.detach() # ** detaching? # N x H x W
            pred_classes = results.pred_classes.detach() # N
            boxes = results.pred_boxes # N x 4
            boxes_img = boxes.tensor.clone().detach() # save the bboxes on image coordinates
            boxes.scale(scale_x=1/width, scale_y=1/height)
            # need to get the mask features that correspond to the particular image
            feats = mask_features[index:index+num_instances, :, :, :]
            index += num_instances
            predictions.append((feats, masks, boxes.tensor, boxes_img, raw_masks, pred_classes))

        if gt_instances is not None:
            return predictions, mask_rcnn_losses
        else:
            return predictions


    def preprocess_imgs(self, imgs):
        device = imgs.device
        imgs = imgs.cpu().numpy()
        imgs = imgs * 255.0 # assuming the dataloader returns 0...1 images
        imgs = imgs.astype(np.uint8)
        height, width = imgs[0,:,:,:].shape[:2] # assume all imgs have same height, width
        images = []
        for i in range(imgs.shape[0]):
            images.append(self.transform_gen.get_transform(imgs[i,:,:,:]).apply_image(imgs[i,:,:,:]))
        images = [torch.as_tensor(x.astype("float32").transpose(2,0,1)).to(device) for x in images]
        if self.pixel_mean.device != device:
            self.pixel_mean = self.pixel_mean.to(device)
            self.pixel_std = self.pixel_std.to(device)
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        return images, height, width


# Using Spatial Pyramid Network for optical flow estimation
# Code adapted from: https://github.com/sniklaus/pytorch-spynet
class FlowNetwork(nn.Module):
    def __init__(self, model_path):
        super(FlowNetwork, self).__init__()
        # Define the basic CNN module of SPyNet 
        class Basic(nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()
                self.moduleBasic = nn.Sequential(
                    nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            def forward(self, input):
                return self.moduleBasic(input)

        self.moduleBasic = nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
        # Load the pretrained SpyFlow model and map the weights to the network
        model_dict = torch.load(model_path).items()
        self.load_state_dict( {module_name : weights for module_name, weights in model_dict} )
        self.backwarp_tenGrid = {}


    def forward(self, imgs1, imgs2):
        # input img is B x H x W x C in BGR format
        imgs1 = [self.preprocess_imgs(imgs1)]
        imgs2 = [self.preprocess_imgs(imgs2)]
        # generate the resolution pyramid
        for i in range(5):
            if imgs1[0].shape[2] > 32 or imgs1[0].shape[3] > 32:
                imgs1.insert(0, nn.functional.avg_pool2d(input=imgs1[0], kernel_size=2, stride=2, count_include_pad=False))
                imgs2.insert(0, nn.functional.avg_pool2d(input=imgs2[0], kernel_size=2, stride=2, count_include_pad=False))
        flow = imgs1[0].new_zeros([ imgs1[0].shape[0], 2, int(math.floor(imgs1[0].shape[2] / 2.0)), int(math.floor(imgs1[0].shape[3] / 2.0)) ])
        # Pass the images sequentially through the basic modules and combine the flow at each step
        for i in range(len(imgs1)):
            flowUpsampled = nn.functional.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow_warp = self.backwarp(imgs=imgs2[i], flow=flowUpsampled)
            flow = self.moduleBasic[i](torch.cat([ imgs1[i], flow_warp, flowUpsampled ], 1)) + flowUpsampled
        self.backwarp_tenGrid = {}
        return flow


    def preprocess_imgs(self, imgs):
        imgs = np.ascontiguousarray(imgs.transpose(0,3,1,2).astype(np.float32)*(1.0/255.0))
        imgs = torch.FloatTensor(imgs).to(self.moduleBasic[0].moduleBasic[0].weight.device)
        b = (imgs[:, 0:1, :, :] - 0.406) / 0.225
        g = (imgs[:, 1:2, :, :] - 0.456) / 0.224
        r = (imgs[:, 2:3, :, :] - 0.485) / 0.229
        return torch.cat([ r, g, b ], 1)

    
    def backwarp(self, imgs, flow):
        if str(flow.size()) not in self.backwarp_tenGrid:
            tenHorizontal = torch.linspace(-1.0, 1.0, flow.shape[3]).view(1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, flow.shape[2]).view(1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])
            self.backwarp_tenGrid[str(flow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).to(imgs.device)
        flow = torch.cat([ flow[:, 0:1, :, :] / ((imgs.shape[3] - 1.0) / 2.0), flow[:, 1:2, :, :] / ((imgs.shape[2] - 1.0) / 2.0) ], 1)
        return nn.functional.grid_sample(input=imgs, grid=(self.backwarp_tenGrid[str(flow.size())] + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)

