import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from models.feature_extractors.midlevel_vision import InstanceSegmentation
from models.feature_extractors.graph_constructor import GraphConstructorSegmenter
from models.feature_extractors.optical_flow import OpticalFlow
from detectron2.structures import Boxes, BitMasks, Instances
from detectron2.utils.events import EventStorage
from detectron2 import model_zoo
from sklearn.metrics.pairwise import euclidean_distances


class InstanceSegmenter(nn.Module):
    def __init__(self, enable_seg_losses, loss_weights):
        super(InstanceSegmenter, self).__init__()

        model_params = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        model_weights = "../configs/model_final.pth" # our pretrained mask rcnn model weights

        self.inst_segm_model = InstanceSegmentation(model_params, model_weights)
        self.enable_seg_losses = enable_seg_losses # boolean, enables maskrcnn specific losses

        self.flowNet = OpticalFlow()
        self.graph = GraphConstructorSegmenter()
        self.loss_weights = loss_weights


    def forward(self, batch):

        flowBatch, flowBboxes = self.flowNet(batch) # Use flow to generate pseudo-ground-truth for instance segmentation
        flowBatch = flowBatch.detach()
        
        context_length = batch['context_frames'].shape[1]
        sequence_length = batch['target_frames'].shape[1]
        
        imgs = batch['context_frames']
        B, T, C, H, W = imgs.shape # batch, sequence+context length, channels, height, width
        imgs = imgs.permute(0, 1, 3, 4, 2)

        context_binary_masks, context_norm_bboxes, context_feat_mask, context_pred_cls = [],[],[],[]
        context_bboxes_img, context_foreground_masks, context_foreground_binary_masks = [], [], []
        background_masks = torch.zeros((B, 1, H, W), dtype=torch.float32).to(imgs.device)
        rcnn_losses = {}

        for t in range(T):
            imgs0 = imgs[:,t,:,:,:] # get frame t from all sequences in the batch

            if t==T-1 and self.enable_seg_losses: # should only run on the last context frame
                
                gt_instances = self.get_maskrcnn_gt(flowBboxes, im_size=(H,W))
                # Need to run each batch ind separately because it may not contain bboxes
                predictions = []
                for b in range(B):
                    if gt_instances[b] is not None:
                        with EventStorage() as storage: # When in training, the detectron2 model needs to be called under event storage
                            _pred, losses_ = self.inst_segm_model(imgs0[b,:,:,:].unsqueeze(0), gt_instances=[gt_instances[b]])
                            if len(rcnn_losses)==0:
                                rcnn_losses = {**losses_} 
                            else:
                                for key in rcnn_losses:
                                    rcnn_losses[key] += losses_[key] 

                            for loss in rcnn_losses:
                                rcnn_losses[loss] = (rcnn_losses[loss] * self.loss_weights[loss]) / B # normalized by batch
                    else:
                        _pred = self.inst_segm_model(imgs0[b,:,:,:].unsqueeze(0))
                    predictions.append(_pred[0]) # _pred is a list with a single tuple

            else: # if enable_seg_losses is disabled
                with torch.no_grad():
                    predictions = self.inst_segm_model(imgs0)

            _raw_masks_temp, _norm_bboxes_temp, _feat_mask_temp, _pred_classes_temp, _binary_masks_temp = [],[],[],[],[]
            _bboxes_img_temp = []
            for b in range(B):
                feat_masks = predictions[b][0]
                binary_masks = predictions[b][1]
                norm_bboxes = predictions[b][2] # (x1, y1, x2, y2)
                bboxes = predictions[b][3] # (x1, y1, x2, y2), N x 4
                raw_masks = predictions[b][4]
                pred_classes = predictions[b][5]
                _norm_bboxes_temp.append(norm_bboxes.clone())
                _pred_classes_temp.append(pred_classes.clone())
                _binary_masks_temp.append(binary_masks.clone())
                _bboxes_img_temp.append(bboxes)
                if raw_masks.shape[0]==0:
                    raise Exception('No masks found!')

                # Resize raw masks and stack with the mask features
                feat_dim = feat_masks.shape[2] # 14
                raw_masks_small = F.interpolate(raw_masks, size=(feat_dim, feat_dim), mode='nearest')
                _feat_mask_temp.append(torch.cat((raw_masks_small, feat_masks), dim=1)) # M x 257 x 14 x 14 

                # Create a single background mask image for the last context frame
                if t==T-1:
                    img_raw_masks = torch.zeros((raw_masks.shape[0], 1, H, W), dtype=torch.float32).to(imgs.device)
                    for i in range(raw_masks.shape[0]):
                        img_raw_masks[i,0,:,:] = self.paste_mask_in_image(mask=raw_masks[i,:,:,:], box=bboxes[i,:], img_size=(H,W))
                    back_raw_masks, _ = torch.min(1-img_raw_masks.clone(), dim=0, keepdim=True) # equivalent to first doing max and then inverting the probs
                    background_masks[b,:,:,:] = back_raw_masks.squeeze(1)
                    self.check_nan(background_masks, "background_masks")

                    # add img_raw_masks and binary masks as collections of foreground_masks
                    context_foreground_masks.append(img_raw_masks)
                    context_foreground_binary_masks.append(binary_masks.unsqueeze(1).float())
            
            context_norm_bboxes.append(_norm_bboxes_temp)
            context_feat_mask.append(_feat_mask_temp)
            context_binary_masks.append(_binary_masks_temp)
            context_pred_cls.append(_pred_classes_temp)
            context_bboxes_img.append(_bboxes_img_temp)


        # Transpose the lists (from TxB to BxT)        
        context_norm_bboxes = self.transpose_list(context_norm_bboxes, B, context_length)
        context_feat_mask = self.transpose_list(context_feat_mask, B, context_length)
        context_bboxes_img = self.transpose_list(context_bboxes_img, B, context_length)

        # Create pseudo gt for the patch_error_loss and coord_error_loss
        # Instead of using flow to get the target tensor, apply mask-rcnn to the target frames
        # and find tracks of the boxes in the sequence length
        target_feat_masks, target_norm_bboxes, target_bboxes_img, gt_masks = self.get_targets_by_association(batch['target_frames'], context_bboxes_img)


        # Make sure that all target tensors are detached !!
        for b in range(B):
            for t in range(sequence_length):
                target_feat_masks[b][t] = target_feat_masks[b][t].detach()
                self.check_nan(target_feat_masks[b][t], "target_feat_masks")
                target_norm_bboxes[b][t] = target_norm_bboxes[b][t].detach()
                self.check_nan(target_norm_bboxes[b][t], "target_norm_bboxes")
                gt_masks[b][t] = gt_masks[b][t].detach()
                self.check_nan(gt_masks[b][t], "gt_masks")
                target_bboxes_img[b][t] = target_bboxes_img[b][t].detach()
                self.check_nan(target_bboxes_img[b][t], "target_bboxes_img")


        context_norm_bboxes = self.convert_to_shapestack_format(context_norm_bboxes, im_size=(H,W))
        target_norm_bboxes = self.convert_to_shapestack_format(target_norm_bboxes, im_size=(H,W))

        # * context_bboxes_img, target_bboxes_img are non-normalized in the image coordinate
        # * context_norm_bboxes, target_norm_bboxes are the correspoding normalized in the shapestacks format

        data_to_graph = {
            'context_crops': context_feat_mask, # B x context_length x M x 257 x 14 x 14
            'context_norm_bboxes': context_norm_bboxes, # B x context_length x M x 4
        }
        graphs, all_coords = self.graph(data_to_graph, dims=(B,context_length,C,H,W))

        target_data_to_graph = {
            'context_crops': target_feat_masks, # B x sequence_length x M x 257 x 14 x 14
            'context_norm_bboxes': target_norm_bboxes # B x sequence_length x M x 4
        }
        target_graphs, target_all_coords = self.graph(target_data_to_graph, dims=(B,sequence_length,C,H,W))

        graph_info = (graphs, all_coords, target_graphs, target_all_coords)
        visual_info = (context_bboxes_img, target_bboxes_img, context_foreground_masks, context_foreground_binary_masks, flowBboxes)

        return graph_info, background_masks, rcnn_losses, gt_masks, visual_info
        

    def vis_batch(self, batch):
        # assume batch=1, context_length=1
        context_frames = batch['context_frames'][0].permute(0, 2, 3, 1) # 1 x H x W x 3
        target_frames = batch['target_frames'][0].permute(0, 2, 3, 1) # T x H x W x 3
        contx_im_tmp = context_frames[0,:,:,:].cpu().numpy()*255.0
        cv2.imwrite("contx_im.png", contx_im_tmp)
        for i in range(target_frames.shape[0]):
            target_im_tmp = target_frames[i,:,:,:].cpu().numpy()*255.0
            cv2.imwrite("target_im_"+str(i)+".png", target_im_tmp)

    def get_centroids(self, bboxes):
        centroids = torch.zeros((bboxes.shape[0], 2), dtype=torch.float32)
        for i in range(bboxes.shape[0]):
            x1,y1,x2,y2 = bboxes[i,:]
            w,h = x2-x1, y2-y1
            centroids[i,0] = x1 + w/2 # cx
            centroids[i,1] = y1 + h/2 # cy
        return centroids


    def get_targets_by_association(self, target_frames, context_bboxes):
        # Assume context_length=1
        # Associate context_frame predictions to the target frame predictions
        # All target frames outputs should have the same number of instances as the context frame 
        target_frames = target_frames.permute(0, 1, 3, 4, 2)
        target_feat_masks, target_norm_bboxes, target_bboxes_img, gt_masks = [], [], [], []
        for b in range(target_frames.shape[0]):
            cont_boxes = context_bboxes[b][0].detach() # N x 4
            current_boxes = cont_boxes.clone() # tensor that will keep the updated set of boxes each timestep

            imgs0 = target_frames[b,:,:,:,:]
            sequence_length, H, W, C = imgs0.shape
            
            seq_bboxes = torch.zeros((sequence_length, cont_boxes.shape[0],4), dtype=torch.float32, device=imgs0.device)
            seq_norm_bboxes = torch.zeros((sequence_length, cont_boxes.shape[0],4), dtype=torch.float32, device=imgs0.device)
            seq_feat_masks_raw = torch.zeros((sequence_length, cont_boxes.shape[0], 257, 14, 14), dtype=torch.float32, device=imgs0.device)
            seq_binary_masks = torch.zeros((sequence_length, cont_boxes.shape[0], H, W), dtype=torch.float32, device=imgs0.device)

            predictions = self.inst_segm_model(imgs0)
            for t in range(sequence_length):
                bboxes = predictions[t][3].detach() # M x 4
                cents = self.get_centroids(bboxes)

                current_cents = self.get_centroids(current_boxes)

                # M x N of distances between centroids
                dist = euclidean_distances(cents.cpu().numpy(), current_cents.cpu().numpy())
                min_inds = np.argmin(dist, axis=0)            
                current_boxes = bboxes[min_inds, :] # get the corresponding boxes, this should have the same dimension as previous frame boxes

                # select min_inds for other needed targets
                feat_masks = predictions[t][0].detach()
                binary_masks = predictions[t][1].detach()
                norm_bboxes = predictions[t][2].detach() # (x1, y1, x2, y2)
                raw_masks = predictions[t][4].detach()

                # Resize raw masks and stack with the mask features
                feat_dim = feat_masks.shape[2] # 14
                raw_masks_small = F.interpolate(raw_masks, size=(feat_dim, feat_dim), mode='nearest')
                feat_masks_raw = torch.cat((raw_masks_small, feat_masks), dim=1) # M x 257 x 14 x 14 

                seq_bboxes[t, :, :] = current_boxes.clone()
                seq_binary_masks[t, :, :, :] = binary_masks[min_inds, :, :].clone()
                seq_feat_masks_raw[t, :, :, :, :] = feat_masks_raw[min_inds, :, :, :].clone()
                seq_norm_bboxes[t, :, :] = norm_bboxes[min_inds, :].clone()

            target_feat_masks.append(seq_feat_masks_raw)
            target_norm_bboxes.append(seq_norm_bboxes)
            target_bboxes_img.append(seq_bboxes)
            gt_masks.append(seq_binary_masks)
        return target_feat_masks, target_norm_bboxes, target_bboxes_img, gt_masks


    def convert_to_shapestack_format(self, norm_bboxes, im_size):
        # Convert x1,y1,x2,y2 to norm_center_x, norm_center_y, norm_width, norm_height
        H,W = im_size
        out_bboxes = []
        for b in range(len(norm_bboxes)):
            seq_boxes = []
            for t in range(len(norm_bboxes[0])): # assume all batches have same sequence length
                bboxes = norm_bboxes[b][t].clone()
                bboxes[:, 0::2] *= W
                bboxes[:, 1::2] *= H
                
                for i in range(bboxes.shape[0]):
                    if bboxes[i,2] >= W:
                        bboxes[i,2] = W-1
                    if bboxes[i,3] >= H:
                        bboxes[i,3] = H-1
                    if bboxes[i,0] < 0:
                        bboxes[i,0] = 0
                    if bboxes[i,1] < 0:
                        bboxes[i,1] = 0

                box_w = bboxes[:,2]-bboxes[:,0]
                box_h = bboxes[:,3]-bboxes[:,1]
                center_x = bboxes[:,0] + box_w/2
                center_y = bboxes[:,1] + box_h/2
                bboxes[:,0] = center_x / W
                bboxes[:,1] = center_y / H
                bboxes[:,2] = box_w / W
                bboxes[:,3] = box_h / H
                seq_boxes.append(bboxes)
            out_bboxes.append(seq_boxes)
        return out_bboxes


    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print("Found NaN !!!!!", name)


    def paste_mask_in_image(self, mask, box, img_size):
        H,W = img_size
        #img_raw_mask = torch.zeros((H, W), dtype=torch.float32).cuda()
        img_raw_mask = torch.zeros((H, W), dtype=torch.float32).to(mask.device)
        mask_box = box.ceil().detach().cpu().numpy() - 1
        mask_box[mask_box<0] = 0
        mask_w, mask_h = int(mask_box[2]-mask_box[0]), int(mask_box[3]-mask_box[1])
        if mask_w==0 or mask_h==0:
            raise Exception('Degenerate box predicted:', mask_box)
        raw_mask_tmp = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode='nearest')
        img_raw_mask[int(mask_box[1]):int(mask_box[3]), int(mask_box[0]):int(mask_box[2])] = raw_mask_tmp[0,0,:,:]
        return img_raw_mask


    # get gt instances to do a "training" pass through mask rcnn
    # Input: last context frame, gt: created from flowBboxes
    def get_maskrcnn_gt(self, flowBboxes, im_size):
        H,W = im_size
        B = len(flowBboxes)
        gt_instances = []
        for b in range(B):
            im_flow_bboxes = flowBboxes[b]

            if im_flow_bboxes is not None:
                nBoxes = im_flow_bboxes.shape[0]
                masks = torch.zeros((nBoxes,H,W), dtype=torch.float32, device=im_flow_bboxes.device)
                for m in range(nBoxes): # create the binary gt masks
                    box = im_flow_bboxes[m,:]
                    masks[m, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1

                gt_classes = torch.zeros((nBoxes), dtype=torch.int64, device=im_flow_bboxes.device) # always give same class (0) as gt
                gt_masks = BitMasks(masks.bool())
                gt_boxes = Boxes(im_flow_bboxes)
                im_gt_instances = Instances(image_size=(H,W), 
                                gt_boxes=gt_boxes, 
                                gt_classes=gt_classes, 
                                gt_masks=gt_masks)
                gt_instances.append(im_gt_instances)
            else:
                gt_instances.append(None)
        return gt_instances
        

    def write_tensor_image(self, tensor, name, mask_id=0):
        # Converts and writes one of the masks into an image 
        tensor_copy = tensor.clone()
        tensor_copy = tensor_copy.detach().cpu().numpy()
        if tensor_copy.shape[1]>1:
            tensor_img = tensor_copy[mask_id,:,:,:].transpose(1,2,0)*255.0
        else:
            tensor_img = tensor_copy[mask_id,0,:,:]*255.0
        cv2.imwrite(name+str(mask_id)+'.png', tensor_img)


    def transpose_list(self, in_list, B, T):
        all_list = []
        for b in range(B):
            row_list = []
            for t in range(T):
                tensor = in_list[t][b]
                row_list.append(tensor)
            all_list.append(row_list)
        return all_list

