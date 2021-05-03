import torch
import torch.nn as nn
import numpy as np
import os
import glob
import torchvision.transforms as T
from PIL import Image
import PIL
from models.feature_extractors.midlevel_vision import FlowNetwork
from skimage import measure
import cv2
import random

###################
# Script to generate the cubes dataset (pseudo-gt) from shapestack images + flow
# Cubes dataset is used to finetune a mask rcnn instance segmentation model
###################

def vis_flow(flow, img1, img2, name=""):
    import cv2
    import flowiz as fz
    batch_size = img1.shape[0]
    for pair_index in range(batch_size):
        cv2.imwrite(name+str(pair_index)+'_img1_test.png', img1[pair_index, :, :, ::-1])
        cv2.imwrite(name+str(pair_index)+'_img2_test.png', img2[pair_index, :, :, ::-1])
        # save a flow file
        output_single = flow[pair_index,:,:,:]
        out_file = name+str(pair_index)+'_out.flo'
        objOutput = open(out_file, 'wb')
        np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
        np.array([ output_single.shape[2], output_single.shape[1] ], np.int32).tofile(objOutput)
        np.array(output_single.cpu().detach().numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)
        objOutput.close()
        # visualize file
        out_img = fz.convert_from_file(out_file)
        cv2.imwrite(name+str(pair_index)+'_flow_out.png', out_img[:,:,::-1])    


# Create bboxes using flow
def gt_flow_bboxes(flow, img1, img2):
    H,W = flow.shape[2], flow.shape[3]
    x, y = flow[:,0,:,:], flow[:,1,:,:] 
    # Cartesian to polar
    mag = torch.sqrt(x**2 + y**2) # B x H x W
    phi = torch.atan2(y, x)
    # Threshold flow based on the magnitude 
    # keep anything that moves more than a certain percentage of the image size (in pixels)
    mag_thresh = H*0.01
    non_valid_inds = (mag < mag_thresh).nonzero(as_tuple=True) # non valid flow inds
    # visualize remaining flow
    flow[non_valid_inds[0], 0, non_valid_inds[1], non_valid_inds[2]] = 0
    flow[non_valid_inds[0], 1, non_valid_inds[1], non_valid_inds[2]] = 0
    #self.vis_flow(flow=flow, img1=img1, img2=img2, name="after_thresh")
    mag[non_valid_inds[0], non_valid_inds[1], non_valid_inds[2]] = 0

    bboxes, binary_masks = [], []
    for i in range(flow.shape[0]): # batch
        im_mag = mag[i, :, :]
        inds = (im_mag > 0).nonzero() 
        if inds.shape[0] == 0: # all flow values were below the mag_thresh
            bboxes.append(None)
            binary_masks.append(None)
        else:
            # Create a bbox for every connected component
            binary_im = im_mag.clone() # 112 x 112
            binary_im[binary_im>0] = 1
            binary_im = binary_im.cpu().numpy()*255.0
            blobs_labels = measure.label(binary_im, background=0)
            regions = measure.regionprops(blobs_labels)
            boxes_img, masks_img = [], []
            for k in range(len(regions)):
                if regions[k].area > 200:
                    bbox_tmp = regions[k].bbox # given as y1,x1,y2,x2
                    y1,x1,y2,x2 = bbox_tmp
                    bbox_ = torch.tensor([x1,y1,x2,y2], device=flow.device).float()
                    boxes_img.append(bbox_)                    
                    convex_image = regions[k].convex_image # bbox size
                    mask = np.zeros((H,W), dtype=np.uint8)
                    mask[y1:y2, x1:x2] = convex_image
                    mask = torch.tensor(mask, device=flow.device)
                    masks_img.append(mask)
            if len(boxes_img)==0:
                boxes_img = None
                masks_img = None
            else:
                boxes_img = torch.stack(boxes_img)
                masks_img = torch.stack(masks_img)
            bboxes.append(boxes_img)
            binary_masks.append(masks_img)
    return bboxes, binary_masks


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)



H = W = 112
transform_ = [Resize((H, W)), T.ToTensor()]
transform = T.Compose(transform_)

# Model taken from here: https://github.com/sniklaus/pytorch-spynet
flowNet = FlowNetwork(model_path='../configs/network-sintel-final.pytorch')
flowNet.eval()

imgs_folders_path = 'data/shapestacks/frc_35/'
im_dirs = os.listdir(imgs_folders_path)
random.shuffle(im_dirs)

save_dir = 'data/cubes/' 
modes = ["train/", "val/"]

for mode in modes:
    
    if not os.path.exists(save_dir+mode):
        os.makedirs(save_dir+mode)

    dataset = {}
    ind=0

    if mode=="train/":
        dirs = im_dirs[:200]
    else:
        dirs = im_dirs[200:220]

    for i in range(len(dirs)):
        current_dir = imgs_folders_path + dirs[i] + "/"
        img_list = glob.glob(current_dir+"*.jpg")
        img_list.sort()

        imgs = torch.zeros((1, len(img_list), H, W, 3), dtype=torch.float32)
        for k in range(len(img_list)):
            img_path = img_list[k]
            with open(img_path, 'rb') as f:
                with Image.open(f) as image:
                    dst_image = transform(image.convert('RGB'))
                    dst_image = dst_image.permute(1,2,0).unsqueeze(0)
                    imgs[0,k,:,:,:] = dst_image.unsqueeze(0)

        imgs = imgs.cpu().numpy()
        imgs = imgs * 255.0 # assuming the dataloader returns 0...1 images
        imgs = imgs.astype(np.uint8)
        T = imgs.shape[1]
        for t in range(T-1):
            img1 = imgs[:,t,:,:,:]
            img2 = imgs[:,t+1,:,:,:]
            output = flowNet(img1, img2)

            # uncomment following line to visualize the flow
            #vis_flow(output, img1, img2, name="test_"+str(t))

            bboxes, binary_masks = gt_flow_bboxes(output.detach(), img1, img2)
            bboxes = bboxes[0] # N x 4
            binary_masks = binary_masks[0] # N x H x W

            # assume batch is 1
            if bboxes is None:
                continue

            bboxes = bboxes.cpu().numpy()
            binary_masks = binary_masks.cpu().numpy()
            nBoxes = bboxes.shape[0]

            filename = dirs[i]+"_"+str(t)+".png"
            
            filepath = save_dir+mode+filename
            cv2.imwrite(filepath, img1[0, :, :, ::-1])

            dataset[ind] = {'filename':filename,
                            'image_id': ind,
                            'height': H,
                            'width': W,
                            'bboxes': bboxes,
                            'masks': binary_masks}
            ind+=1
            
    np.save(save_dir+mode+"annot.npy", dataset)
