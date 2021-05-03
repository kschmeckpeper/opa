import torch
import torch.nn as nn
from torch_geometric.data import Batch
from models.dynamics_models.encoded_graph_dynamics_model import translate_and_scale_patch
import cv2


class ColorSegmentation(nn.Module):
    def forward(self, instances):
        averages = torch.mean(instances, dim=(2, 3))
        masks = instances > 0.5
        high = torch.argmax(averages, dim=1)
        low = torch.argmin(averages, dim=1)
        combined_masks = []
        for i in range(high.shape[0]):
            combined_masks.append(torch.logical_and(masks[i, high[i], :, :],
                                                    torch.logical_not(masks[i, low[i], :, :])))
        combined_masks = torch.stack(combined_masks)

        combined_masks = combined_masks.to(torch.float)
        combined_masks = combined_masks.view(combined_masks.shape[0], 1, combined_masks.shape[1], combined_masks.shape[2])
        return combined_masks


class GraphConstructor(nn.Module):
    def __init__(self,
                 mask_predictor_feats=16):
        super().__init__()

        self.mask_predictor = nn.Sequential(nn.Conv2d(3, mask_predictor_feats, 3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(mask_predictor_feats, mask_predictor_feats, 3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(mask_predictor_feats, 1, 3, padding=1),
                                            nn.Sigmoid())
        self.mask_predictor_color = ColorSegmentation()
        self.bce_loss = nn.BCELoss()


    def forward(self, data):
        graphs = []
        all_coords_over_time = []
        background_masks_over_time = []
        gt_masks_over_time = []
        bce_error = 0
        # batch_size x sequence (context) length x number of instances x channels x height x width
        B, T, M, C, H, W = data['context_crops'].shape
        for t in range(data['context_crops'].shape[1]):
            instances = []
            batches = []
            edge_indices = []
            all_coords = []
            for batch in range(data['context_crops'].shape[0]):
                prev_instances = {}
                for crop in range(data['context_crops'].shape[2]):
                    instances.append(data['context_crops'][batch][t][crop])
                    curr_index = len(instances) - 1
                    batches.append(batch)
                    x = data['context_norm_bboxes'][batch][t][crop][0] * data['context_frames'].shape[-2]
                    y = data['context_norm_bboxes'][batch][t][crop][1] * data['context_frames'].shape[-1]
                    for i in prev_instances:
                        edge_indices.append([i, curr_index])
                        edge_indices.append([curr_index, i])
                    prev_instances[curr_index] = [x, y]
                    all_coords.append([x,
                                       y,
                                       data['context_norm_bboxes'][batch][t][crop][2],# Not normed
                                       data['context_norm_bboxes'][batch][t][crop][3]])

            edge_indices = torch.tensor(edge_indices, dtype=torch.long, device=data['context_crops'].device)
            edge_indices = edge_indices.t().contiguous()
            instances = torch.stack(instances)
            batches = torch.tensor(batches, dtype=torch.long, device=data['context_crops'].device)

            if instances.shape[1] == 3:
                masks = self.mask_predictor(instances)
                masks_color = self.mask_predictor_color(instances) # gt masks of current frame
                bce_error_frame = self.bce_loss(masks, masks_color)
                bce_error += bce_error_frame

                instances = torch.cat((masks, instances), axis=1)

            graph = Batch(x=instances, batch=batches, edge_index=edge_indices)
            graphs.append(graph)
            all_coords = torch.tensor(all_coords).cuda()
            all_coords_over_time.append(all_coords)

            background_masks = torch.ones((data['context_frames'].shape[0],
                                           1,
                                           data['context_frames'].shape[3],
                                           data['context_frames'].shape[4]),
                                          device=data['context_frames'].device)

            instance_masks_shape = (instances.shape[0], 1, instances.shape[2], instances.shape[3])
            scaled_masks = translate_and_scale_patch(instances[:, 0, :, :].view(instance_masks_shape),
                                                     all_coords,
                                                     instance_masks_shape)
            
            # get the gt masks in the image coordinates
            gt_scaled_masks = translate_and_scale_patch(masks_color, all_coords, masks_color.shape)       
            gt_masks_over_time.append(gt_scaled_masks)
            
            for i in range(instances.shape[0]):
                background_masks[batches[i], :, :, :] -= scaled_masks[i, :, :, :]

            background_masks = nn.functional.relu(background_masks)

            background_masks_over_time.append(background_masks)

        mask_loss = {'bce_mask':bce_error/T}
        
        all_coords_over_time = torch.stack(all_coords_over_time)
        gt_masks_over_time = torch.stack(gt_masks_over_time)

        # Separate the batch and instances dimensions for gt masks
        gt_masks_over_time = gt_masks_over_time.permute(1,0,2,3,4)
        gt_masks_over_time = gt_masks_over_time.view(B, M, T, 1, H, W)
        gt_masks_over_time = gt_masks_over_time.permute(0,2,1,3,4,5) # B, T, M, 1, H, W

        return graphs, all_coords_over_time, background_masks, mask_loss, gt_masks_over_time




class GraphConstructorSegmenter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data, dims):
        graphs = []
        all_coords_over_time = []
        batch_size, context_length, _, height, width = dims
        device = data['context_crops'][0][0].device
        for t in range(context_length):
            instances = []
            batches = []
            edge_indices = []
            all_coords = []
            for batch in range(batch_size):
                prev_instances = {}
                for crop in range(len(data['context_crops'][batch][t])):    
                    instances.append(data['context_crops'][batch][t][crop])
                    curr_index = len(instances) - 1
                    batches.append(batch)
                    x = data['context_norm_bboxes'][batch][t][crop][0] * width 
                    y = data['context_norm_bboxes'][batch][t][crop][1] * height 
                    for i in prev_instances:
                        edge_indices.append([i, curr_index])
                        edge_indices.append([curr_index, i])
                    prev_instances[curr_index] = [x, y]
                    all_coords.append([x,
                                       y,
                                       data['context_norm_bboxes'][batch][t][crop][2],# Not normed
                                       data['context_norm_bboxes'][batch][t][crop][3]])
            
            # when we have only one crop then no edges are added, which crashes later
            if len(edge_indices)==0:
                edge_indices.append([0, 0])

            edge_indices = torch.tensor(edge_indices, dtype=torch.long, device=device)
            edge_indices = edge_indices.t().contiguous()
            instances = torch.stack(instances)
            batches = torch.tensor(batches, dtype=torch.long, device=device)

            graph = Batch(x=instances, batch=batches, edge_index=edge_indices)
            graphs.append(graph)
            all_coords = torch.tensor(all_coords).cuda()
            all_coords_over_time.append(all_coords)
        return graphs, all_coords_over_time



