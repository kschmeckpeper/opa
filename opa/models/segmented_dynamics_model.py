import torch
import torch.nn as nn
import lpips
import itertools

from models.base_model import BaseModel

from models.dynamics_models.encoded_graph_dynamics_model import extract_patch_from_frame

class SegmentedDynamicsModel(BaseModel):
    """ Predicts future states by segmenting the input into a graph
    """
    def __init__(self,
                 segmenter,
                 dynamics_model,
                 generator,
                 accumulate_flows=False,
                 detach_extractor=False,
                 recon_from_first=False,
                 coord_loss_scale=1.0,
                 patch_loss_scale=1.0,
                 mask_patch_loss=False,
                 seg_loss_scale=1.0,
                 pred_attn_loss_scale=1.0,
                 pred_loss_scale=1.0,
                 dataset="robonet",
                 feature_extractor="precropped"):
        super().__init__()

        self._segmenter = segmenter
        self._dynamics_model = dynamics_model
        self._generator = generator

        self._accumulate_flows = accumulate_flows
        self._coord_loss_scale = coord_loss_scale
        self._patch_loss_scale = patch_loss_scale
        self._detach_extractor = detach_extractor
        self._recon_from_first = recon_from_first
        self._mask_patch_loss = mask_patch_loss
        self._seg_loss_scale = seg_loss_scale
        self._pred_attn_loss_scale = pred_attn_loss_scale
        self._pred_loss_scale = pred_loss_scale
        self._lpips = lpips.LPIPS(net='alex')#.cuda()

        self.dataset = dataset
        self.feature_extractor = feature_extractor

        # attn loss
        self.l1_loss = nn.L1Loss(reduction="sum") # manually normalizing based on size of mask
        self.mse_loss = nn.MSELoss()

        self._mask_patch_loss = mask_patch_loss

        self._lpips = lpips.LPIPS(net='alex').cuda()

    def forward(self, batch):
        mse_loss = nn.MSELoss()
        # When segmenter is maskrcnn, if enable_seg_losses=False, then seg_losses is empty

        if self.feature_extractor=="instance_segmenter" or self.feature_extractor=="spxl_segmenter":
            graph_info, background_masks, seg_losses, gt_target_masks, visual_info = self._segmenter(batch)
            (segmented_context_frames, coordinates, segmented_target_frames, target_coordinates) = graph_info
            (context_bboxes_img, target_bboxes_img, context_foreground_masks, context_foreground_binary_masks, flowBboxes) = visual_info

        elif self.feature_extractor=="precropped":
            segmented_context_frames, coordinates, background_masks, seg_losses, _ = self._segmenter(batch)
            #if self._patch_loss_scale != 0 or self._coord_loss_scale != 0:
            target_batch = {'context_frames': batch['target_frames']}
            if 'target_crops' in batch.keys():
                target_batch['context_crops'] = batch['target_crops']
                target_batch['context_norm_bboxes'] = batch['target_norm_bboxes']

            segmented_target_frames, target_coordinates, _, _, gt_target_masks = self._segmenter(target_batch)
        
        else:
            raise Exception('Do not recognize given feature extractor!!')

        if self._detach_extractor:
            segmented_context_frames[-1].x = segmented_context_frames[-1].x.detach()
            coordinates[-1] = coordinates[-1].detach()
            background_masks = background_masks.detach()

            
        
        prev_instances = segmented_context_frames[-1].clone()


        pre_synth_pred_frames = []
        pred_frames = []
        pred_patches = []
        masked_pred_patches = []
        seg_patches = []
        masked_seg_patches = []

        prev_frame = batch['context_frames'][:, -1]
        H,W = prev_frame.shape[2], prev_frame.shape[3]
        prev_output = {'coordinates': coordinates[-1]}
        all_generator_diagnostics = {}
        

        patch_error_loss = torch.tensor(0.0, device=coordinates[-1].device)
        coord_error_loss = torch.tensor(0.0, device=coordinates[-1].device)

        B, sequence_length = batch['actions'].shape[0], batch['actions'].shape[1]
        dynamics_bboxes_img = []

        batch_indices = prev_instances.batch

        for i in range(sequence_length):
            if not self._accumulate_flows:
                instances, _ = self._dynamics_model(prev_instances, batch['actions'][:, i, :])
            else:
                instances, prev_output, dynamics_boxes = self._dynamics_model(prev_instances,
                                                              batch['actions'][:, i, :],
                                                              **prev_output)

            # divide the boxes into their batches
            boxes_tmp = []
            for b in range(B):
                inds = (batch_indices==b).nonzero()
                boxes_tmp.append(dynamics_boxes[inds,:].squeeze(1)) # B x M x 4
            dynamics_bboxes_img.append(boxes_tmp)
            

            if self._patch_loss_scale != 0 or self._coord_loss_scale != 0:
                coord_error_loss = coord_error_loss + self.mse_loss(prev_output['coordinates'].clone(), target_coordinates[i]) / sequence_length
                if not self._mask_patch_loss:
                    patch_error_loss = patch_error_loss + self.mse_loss(prev_output['prev_patch'].clone(), segmented_target_frames[i].x) / sequence_length
                else:
                    pred_patch = prev_output['prev_patch'].clone()
                    seg_patch = segmented_target_frames[i].x
                    pred_mask = pred_patch[:, 0, :, :].view(pred_patch.shape[0],
                                                            1,
                                                            pred_patch.shape[2],
                                                            pred_patch.shape[3])
                    seg_mask = seg_patch[:, 0, :, :].view(pred_mask.shape)
                    masked_pred_patch = pred_mask * pred_patch[:, 1:, :, :]
                    masked_seg_patch = seg_mask * seg_patch[:, 1:, :, :]
                    patch_error_loss = patch_error_loss + self.mse_loss(masked_pred_patch, masked_seg_patch) / sequence_length

            prev_instances = instances


            if not self._recon_from_first:
                pred_frame, generator_diagnostics = self._generator(instances, prev_frame, background_masks)
            else:
                pred_frame, generator_diagnostics = self._generator(instances, batch['context_frames'][:, -1], background_masks)
            pred_frames.append(pred_frame)
            if 'final_synth_pre' in generator_diagnostics['images']:
                pre_synth_pred_frames.append(generator_diagnostics['images']['final_synth_pre'])


            pred_patches.append(torch.sigmoid(prev_output['prev_patch'].clone()[:, 1:, :, :]))
            seg_patches.append(segmented_target_frames[i].x[:, 1:, :, :])


            #if True:#self._patch_loss_scale != 0 or self._coord_loss_scale != 0:
            coord_error_loss = coord_error_loss + mse_loss(prev_output['coordinates'].clone(), target_coordinates[i])
            output_patch = prev_output['prev_patch'].clone()
            
            mask_shape = generator_diagnostics['images']['post_softmax_object_masks'].shape
            output_patch[:, 0, :, :] = extract_patch_from_frame(generator_diagnostics['images']['post_softmax_object_masks'].view(mask_shape[0],
                                                                                                                                  1,
                                                                                                                                  mask_shape[1],
                                                                                                                                  mask_shape[2]),
                                                                prev_output['coordinates'],
                                                                output_patch.shape[-2:]).view(output_patch[:, 0, :, :].shape)
            if not self._mask_patch_loss:
                output_patch[:, 1:, :, :] = torch.sigmoid(output_patch[:, 1:, :, :])
                patch_error_loss = patch_error_loss + mse_loss(output_patch[:, 1:, :, :], segmented_target_frames[i].x[:, 1:, :, :])
            pred_patch = output_patch
            seg_patch = segmented_target_frames[i].x
            pred_mask = pred_patch[:, 0, :, :].view(pred_patch.shape[0],
                                                    1,
                                                    pred_patch.shape[2],
                                                    pred_patch.shape[3])
            seg_mask = seg_patch[:, 0, :, :].view(pred_mask.shape)
            masked_pred_patch = pred_mask * torch.sigmoid(pred_patch[:, 1:, :, :])
            masked_seg_patch = seg_mask * seg_patch[:, 1:, :, :]

            masked_pred_patches.append(masked_pred_patch)
            masked_seg_patches.append(masked_seg_patch)
            if self._mask_patch_loss:
                patch_error_loss = patch_error_loss + mse_loss(masked_pred_patch, masked_seg_patch)


            generator_diagnostics['images'].pop('post_softmax_object_masks')

            prev_frame = pred_frames[-1]

            for data_type in generator_diagnostics:
                if data_type not in all_generator_diagnostics:
                    all_generator_diagnostics[data_type] = {}
                for k in generator_diagnostics[data_type]:
                    if k not in all_generator_diagnostics[data_type]:
                        all_generator_diagnostics[data_type][k] = []
                    all_generator_diagnostics[data_type][k].append(generator_diagnostics[data_type][k].detach())

        pred_frames = torch.stack(pred_frames).permute(1, 0, 2, 3, 4)
        

        # dynamics_bboxes_img is sequence_length x B x M x 4 -- need to transpose first two dimensions
        dynamics_bboxes_img = self._segmenter.transpose_list(dynamics_bboxes_img, B, sequence_length) 

        output = {}

        output['masks'] = {'context_foreground_masks':context_foreground_masks, # B x M x 1 x H x W
                            'context_foreground_binary_masks':context_foreground_binary_masks} # B x M x 1 x H x W
        output['bboxes'] = {'context_bboxes_img':context_bboxes_img, # B x context_length x M x 4
                            'target_bboxes_img':target_bboxes_img, # B x sequence_length x M x 4
                            'dynamics_bboxes_img': dynamics_bboxes_img, # B x sequence_length x M x 4
                            'flow_bboxes': flowBboxes}
        output['images'] = {'pred': pred_frames,
                            'overlay':(pred_frames + batch['target_frames']) / 2.0,
                            'background_masks': background_masks.view(background_masks.shape[0], 1, background_masks.shape[1], background_masks.shape[2], background_masks.shape[3]),
                            'patches_pred': torch.stack(pred_patches).permute(1, 0, 2, 3, 4),
                            'patches_masked_pred': torch.stack(masked_pred_patches).permute(1, 0, 2, 3, 4),
                            'patches_seg': torch.stack(seg_patches).permute(1, 0, 2, 3, 4),
                            'patches_masked_seg': torch.stack(masked_seg_patches).permute(1,  0, 2, 3, 4),
                            }
        for k in all_generator_diagnostics['images'].keys():
            output['images']["generator/" + k] = torch.stack(all_generator_diagnostics['images'][k]).permute(1, 0, 2, 3, 4)

        output['metrics'] = {}
        for k in all_generator_diagnostics['metrics'].keys():
            output['metrics']["generator/"+k] = all_generator_diagnostics['metrics'][k]

        pred_error_loss = self.mse_loss(pred_frames, batch['target_frames'])

        if len(pre_synth_pred_frames) > 0:
            pre_synth_pred_frames = torch.stack(pre_synth_pred_frames).permute(1, 0, 2, 3, 4)
            pred_error_loss += self.mse_loss(pre_synth_pred_frames, batch['target_frames'])
        output['metrics']['lpips'] = torch.mean(self._lpips(pred_frames.reshape(-1,
                                                                                pred_frames.shape[2],
                                                                                pred_frames.shape[3],
                                                                                pred_frames.shape[4]), 
                                                            batch['target_frames'].reshape(-1,
                                                                                           pred_frames.shape[2],
                                                                                           pred_frames.shape[3],
                                                                                           pred_frames.shape[4])))       


        output['metrics']['lpips'] = torch.mean(self._lpips(pred_frames.reshape(-1,
                                                                                pred_frames.shape[2],
                                                                                pred_frames.shape[3],
                                                                                pred_frames.shape[4]), 
                                                            batch['target_frames'].reshape(-1,
                                                                                           pred_frames.shape[2],
                                                                                           pred_frames.shape[3],
                                                                                           pred_frames.shape[4])))
        output['metrics']['coord_error_loss'] = coord_error_loss.detach()
        output['metrics']['patch_error_loss'] = patch_error_loss.detach()
        output['metrics']['pred_error_loss'] = pred_error_loss.detach()
        output['losses'] = {'mse': self._pred_loss_scale * pred_error_loss +
                                   self._coord_loss_scale * coord_error_loss +
                                   self._patch_loss_scale * patch_error_loss}

        if len(gt_target_masks)!=0:
            pred_attn_loss = self.pred_attention_loss(pred_frames, batch['target_frames'], gt_target_masks)
            output['losses']['pred_attn'] = pred_attn_loss * self._pred_attn_loss_scale           
        
        if len(seg_losses)!=0:
            for key in seg_losses:
                seg_losses[key] *= self._seg_loss_scale
            output['losses'].update(seg_losses)
        return output


    def pred_attention_loss(self, pred_frames, target_frames, gt_masks):
        # Penalize predictions specifically on foreground pixels
        B, T, C, H, W = target_frames.shape # batch_size, sequence_length, channels, height , width
        loss = 0
        for b in range(B):
            seq_frame_masks = gt_masks[b]
            for t in range(T):
                gt_frame = target_frames[b,t,:,:,:]
                pred_frame = pred_frames[b,t,:,:,:]
                frame_masks = seq_frame_masks[t].unsqueeze(1)
                masked_gt_frame = gt_frame.unsqueeze(0) * frame_masks # element wise multiplication
                masked_pred_frame = pred_frame.unsqueeze(0) * frame_masks
                N = torch.nonzero(frame_masks, as_tuple=False).shape[0] # number of pixels belonging to foreground
                loss_ = self.l1_loss(masked_pred_frame, masked_gt_frame) / N
                loss += loss_
        return loss / (B*T)
