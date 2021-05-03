import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

#from .pytorch_utils.base_trainer import BaseTrainer
#from .datasets import get_dataset_from_options
#from .models import get_model_from_options
from pytorch_utils.base_trainer import BaseTrainer
from datasets import get_dataset_from_options
from models import get_model_from_options

class Trainer(BaseTrainer):
    """ Implements training for prediction models
    """
    def init_fn(self):
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])
        self.train_ds = get_dataset_from_options(self.options, is_train=True)
        self.test_ds = get_dataset_from_options(self.options, is_train=False)

        self.model = get_model_from_options(self.options)
        self.models_dict = {'model':self.model}
        self.optimizers_dict = {}

        # assign different learning rates to different parts of the overall architecture
        if self.options.feature_extractor=="instance_segmenter":
            # roi_heads corresponds to Fast RCNN predictors with mask head, can be further partitioned into [box_head, box_predictor, mask_head]
            self.optimizers_dict['model'] = \
                    torch.optim.Adam([{'params':self.models_dict['model']._segmenter.inst_segm_model.model.backbone.parameters(), 'lr':self.options.lr_backbone}, # FPN
                                      {'params':self.models_dict['model']._segmenter.inst_segm_model.model.proposal_generator.parameters(), 'lr':self.options.lr_proposals}, # RPN
                                      {'params':self.models_dict['model']._segmenter.inst_segm_model.model.roi_heads.parameters(), 'lr':self.options.lr_rois},
                                      {'params':self.models_dict['model']._segmenter.flowNet.parameters(), 'lr':self.options.lr_flow},
                                      {'params':self.models_dict['model']._dynamics_model.parameters()},
                                      {'params':self.models_dict['model']._generator.parameters()} ],
                                      lr=self.options.lr,
                                      weight_decay=self.options.wd)

        elif self.options.feature_extractor=="spxl_segmenter":
            self.optimizers_dict['model'] = \
                    torch.optim.Adam([{'params':self.models_dict['model']._segmenter.model.parameters(), 'lr':self.options.lr_spxl},
                                      {'params':self.models_dict['model']._dynamics_model.parameters()},
                                      {'params':self.models_dict['model']._generator.parameters()}],
                                      lr=self.options.lr,
                                      weight_decay=self.options.wd)

        else:
            for model in self.models_dict:
                self.optimizers_dict[model] = \
                        torch.optim.Adam([{'params':self.models_dict[model].parameters(),
                                        'initial_lr':self.options.lr}],
                                        lr=self.options.lr,
                                        weight_decay=self.options.wd)


    def train_step(self, input_batch):
        for model in self.models_dict:
            self.models_dict[model].train()

        outputs = self.model(input_batch)

        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].zero_grad()
        for loss in outputs['losses']:
            outputs['losses'][loss].backward(retain_graph=True)
        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].step()

        for scalar_name in ['losses', 'metrics']:
            for k in outputs[scalar_name]:
                outputs[scalar_name][k] = outputs[scalar_name][k].detach()

        return [outputs]

    def train_summaries(self, input_batch, save_images, model_output):
        self._save_summaries(input_batch, model_output, save_images, is_train=True)

    def test(self):
        for model in self.models_dict:
            self.models_dict[model].eval()

        test_data_loader = DataLoader(self.test_ds,
                                      batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        batch = None
        for tstep, batch in enumerate(tqdm(test_data_loader,
                                           desc='Testing',
                                           total=self.options.test_iters)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                out_data = self.model(batch)

            # Exit out of testing so checkpoint can be saved
            if time.time() > self.endtime:
                break

            # Stop testing if test iterations has been exceded
            if tstep > self.options.test_iters:
                break

        self._save_summaries(batch, out_data, save_images=True, is_train=False)


    def _save_summaries(self, batch, output, save_images, is_train=False):
        prefix = 'train/' if is_train else 'test/'

        if save_images:

            # maskrcnn bbox prediction on last context_frame, batch_ind=0
            last_context_frame = batch['context_frames'][0,-1,:,:,:]
            last_context_bboxes = output['bboxes']['context_bboxes_img'][0][-1]
            self.summary_writer.add_image_with_boxes(prefix+"images/maskrcnn_context_bboxes", last_context_frame, last_context_bboxes, self.step_count)
            # pseudo gt on first sequence frame, batch_ind=0
            first_target_frame = batch['target_frames'][0,0,:,:,:]
            first_target_bboxes = output['bboxes']['target_bboxes_img'][0][0]
            self.summary_writer.add_image_with_boxes(prefix+"images/pseudo_gt_target_bboxes", first_target_frame, first_target_bboxes, self.step_count)
            # dynamics bbox prediction on first sequence frame, batch_ind=0
            first_dynamics_bboxes = output['bboxes']['dynamics_bboxes_img'][0][0]
            self.summary_writer.add_image_with_boxes(prefix+"images/dynamics_sequence_bboxes", first_target_frame, first_dynamics_bboxes, self.step_count)
            # maskrcnn foreground masks on last context frame, batch_ind=0, provided as gif
            last_context_masks = output['masks']['context_foreground_masks'][0].unsqueeze(0)
            self.summary_writer.add_video(prefix+"gifs/context_maskrcnn_masks", last_context_masks, self.step_count, fps=1)
            # maskrcnn binary foreground masks on last context frame, batch_ind=0, provided as gif
            last_context_binary_masks = output['masks']['context_foreground_binary_masks'][0].unsqueeze(0)
            self.summary_writer.add_video(prefix+"gifs/context_maskrcnn_binary_masks", last_context_binary_masks, self.step_count, fps=1)
            # bboxes generated from optical flow for the last context_frame, if bbox is None, then only show image, batch_ind=0
            flow_bbox = output['bboxes']['flow_bboxes'][0]
            if flow_bbox is None:
                self.summary_writer.add_image(prefix+"images/flow_bboxes", last_context_frame, self.step_count)
            else:
                self.summary_writer.add_image_with_boxes(prefix+"images/flow_bboxes", last_context_frame, flow_bbox, self.step_count)


            combined_input = torch.cat([batch['context_frames'], batch['target_frames']], axis=1)
            self.summary_writer.add_video(prefix + "gifs/gt_images", combined_input, self.step_count)
            grid_input = make_grid(combined_input.view(-1,
                                                       combined_input.shape[-3],
                                                       combined_input.shape[-2],
                                                       combined_input.shape[-1]),
                                   nrow=combined_input.shape[1],
                                   pad_value=1)
            self.summary_writer.add_image(prefix + "images/gt_images", grid_input, self.step_count)

            for k in output['images']:
                output_3_channel = output['images'][k]

                if output_3_channel.shape[2] > 3:
                    continue

                if output_3_channel.shape[2] == 1:
                    output_3_channel = output_3_channel.repeat(1, 1, 3, 1, 1)
                if output_3_channel.shape[0] == batch['context_frames'].shape[0]:
                    combined_pred = torch.cat([torch.zeros(batch['context_frames'].shape,
                                                           device=self.device,
                                                           dtype=output['images'][k].dtype),
                                               output_3_channel], axis=1)
                else:
                    combined_pred = output_3_channel
                if torch.min(combined_pred) < 0.0 or torch.max(combined_pred) > 1.0:
                    import pdb
                    pdb.set_trace()
                # Save videos for outputs with more than 1 temporal frame
                if output['images'][k].shape[1] > 1:
                    self.summary_writer.add_video(prefix  + "gifs/" +  k, combined_pred, self.step_count)
                grid_pred = make_grid(combined_pred.reshape(-1,
                                                         combined_pred.shape[-3],
                                                         combined_pred.shape[-2],
                                                         combined_pred.shape[-1]),
                                      nrow=combined_pred.shape[1],
                                      pad_value=1)
                self.summary_writer.add_image(prefix + "images/" + k, grid_pred, self.step_count)

        for scalar_type in ['losses', 'metrics']:
            for k in output[scalar_type]:
                self.summary_writer.add_scalar(prefix + k, output[scalar_type][k], self.step_count)

        if is_train:
            self.summary_writer.add_scalar(prefix + "lr", self.get_lr(), self.step_count)
