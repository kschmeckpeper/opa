# Object-centric Video Prediction without Annotation

This is the official code repository for Object-centric Video Prediction without Annotation.


# Installation

We recomend installing the packages in a clean virtual environment.

Install the required packages 
```
pip install -r requirements.txt
pip install -r requirements_2.txt
```
`requirements2.txt` contains the packages that must be installed after torch is set up.

Our code requires the installation of Detectron2 in order to run Mask R-CNN. We provide our own edited repository [here](https://github.com/daniilidis-group/my_detectron2/tree/my_detectron2) that needs to be installed in the same virtual environment as the rest of the requirements. The installation instructions are in the provided repository.

# Usage

We provide a pretrained Mask R-CNN model for instance segmentation in the Shapestacks environment [here](https://drive.google.com/file/d/1hQD-xsofUL6Wz8lUQ9uYCPQ2mxDia6Dr/view?usp=sharing). For optical flow we used the SPyNet from [this](https://github.com/sniklaus/pytorch-spynet) repository. The model can be downloaded from [here](http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch). Both of these models should be included in the `configs` directory.

To train the prediction model given a pretrained segmentation model, run the following command.
```
python3 -u main.py --name EXP_NAME --log_dir PATH_TO_LOG_DIR --model_type accumulated_dynamics --dynamics_model encoded_graph --dataset shapestacks --feature_extractor instance_segmenter --batch_size 1 --test_batch_size 4 --sequence_length 3 --coord_loss_scale 0.001 --patch_loss_scale 1e-06 --detach_extractor --no_graph --recon_from_first --mask_patch_loss --zero_init --seg_loss_scale 1e-08 --pred_attn_loss_scale 0.0001 --lr_backbone 1e-08 --lr_proposals 1e-08 --lr_rois 1e-08
```


