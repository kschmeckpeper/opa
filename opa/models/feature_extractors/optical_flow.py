import torch
import torch.nn as nn
from models.feature_extractors.midlevel_vision import FlowNetwork
import numpy as np
import cv2
from skimage import measure

class OpticalFlow(nn.Module):
    def __init__(self):
        super().__init__()
        # Model taken from here: https://github.com/sniklaus/pytorch-spynet
        self.flowNet = FlowNetwork(model_path='../configs/network-sintel-final.pytorch')

    
    def forward(self, batch):
        # Return T-1 sequences of optical flow between consecutive frames
        # Use the last context_frame + all sequence frames
        context_frame = batch['context_frames'][:,-1,:,:,:].unsqueeze(1)
        imgs = torch.cat( (context_frame, batch['target_frames']), 1 )
        B, T, C, H, W = imgs.shape # batch, context length, channels, height, width
        imgs = imgs.permute(0,1,3,4,2)
        imgs = imgs.cpu().numpy()#.astype(np.uint8)
        imgs = imgs * 255.0 # assuming the dataloader returns 0...1 images
        imgs = imgs.astype(np.uint8)
        flow_batch = torch.zeros((B, T-1, 2, H, W), dtype=torch.float32).to(batch['context_frames']) #.cuda()
        for t in range(T-1):
            img1 = imgs[:,t,:,:,:]
            img2 = imgs[:,t+1,:,:,:]
            output = self.flowNet(img1, img2)
            flow_batch[:,t,:,:,:] = output
            if t==0: # use only flow between context_frame and target_frame
                bboxes = self.gt_flow_bboxes(output.detach(), img1, img2)
        return flow_batch, bboxes # B x T-1 x 2 x H x W,  B x M x 4 (M might differ in each batch_dim)


    # Create bboxes using flow
    def gt_flow_bboxes(self, flow, img1, img2):
        H,W = flow.shape[2], flow.shape[3]
        x, y = flow[:,0,:,:], flow[:,1,:,:] 
        # Cartesian to polar
        mag = torch.sqrt(x**2 + y**2) # B x H x W
        phi = torch.atan2(y, x)
        # Threshold flow based on the magnitude 
        # keep anything that moves more than a certain percentage of the image size (in pixels)
        mag_thresh = H*0.01 #0.015 # 1.12 # 1.68 # 2.24 pixels
        non_valid_inds = (mag < mag_thresh).nonzero(as_tuple=True) # non valid flow inds
        
        # visualize flow
        #flow[non_valid_inds[0], 0, non_valid_inds[1], non_valid_inds[2]] = 0
        #flow[non_valid_inds[0], 1, non_valid_inds[1], non_valid_inds[2]] = 0
        #self.vis_flow(flow=flow, img1=img1, img2=img2, name="after_thresh")
        
        mag[non_valid_inds[0], non_valid_inds[1], non_valid_inds[2]] = 0

        bboxes = []
        for i in range(flow.shape[0]): # batch
            im_mag = mag[i, :, :]
            inds = (im_mag > 0).nonzero() 
            if inds.shape[0] == 0: # all flow values were below the mag_thresh
                bboxes.append(None)
            else:
                # Create a bbox for every connected component
                binary_im = im_mag.clone() # 112 x 112
                binary_im[binary_im>0] = 1
                binary_im = binary_im.cpu().numpy()*255.0
                blobs_labels = measure.label(binary_im, background=0)
                regions = measure.regionprops(blobs_labels)
                boxes_img = []
                for k in range(len(regions)):
                    if regions[k].area > 100:
                        bbox_tmp = regions[k].bbox # given as y1,x1,y2,x2
                        y1,x1,y2,x2 = bbox_tmp
                        bbox_ = torch.tensor([x1,y1,x2,y2], device=flow.device).float()
                        boxes_img.append(bbox_)
                if len(boxes_img)==0:
                    boxes_img = None
                else:
                    boxes_img = torch.stack(boxes_img)
                bboxes.append(boxes_img)
        return bboxes


    def vis_flow(self, flow, img1, img2, name=""):
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