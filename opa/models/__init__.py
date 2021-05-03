""" Prediction models
"""
from .segmented_dynamics_model import SegmentedDynamicsModel

from .feature_extractors import get_feature_extractor_from_options
from .dynamics_models import get_dynamics_model_from_options
from .image_generators import get_image_generator_from_options

def get_model_from_options(options):
    """ Get a model from the options
    """
    if options.model_type == 'accumulated_dynamics':
        return SegmentedDynamicsModel(get_feature_extractor_from_options(options),
                                      get_dynamics_model_from_options(options),
                                      get_image_generator_from_options(options),
                                      detach_extractor=options.detach_extractor,
                                      accumulate_flows=True,
                                      recon_from_first=options.recon_from_first,
                                      coord_loss_scale=options.coord_loss_scale,
                                      patch_loss_scale=options.patch_loss_scale,
                                      mask_patch_loss=options.mask_patch_loss,
                                      seg_loss_scale=options.seg_loss_scale,
                                      pred_attn_loss_scale=options.pred_attn_loss_scale,
                                      pred_loss_scale=options.pred_loss_scale,
                                      dataset=options.dataset,
                                      feature_extractor=options.feature_extractor)

    raise NotImplementedError("Unknown model type requested: {}".format(options.model_type))
