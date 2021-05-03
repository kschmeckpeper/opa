from .graph_constructor import GraphConstructor 
from .instance_segmenter import InstanceSegmenter

def get_feature_extractor_from_options(options):
    """ Gets the feature extractors given the options
    """
    if options.feature_extractor == 'precropped':
        return GraphConstructor()
    if options.feature_extractor == 'instance_segmenter':
        loss_weights = {'loss_rpn_cls':options.loss_rpn_cls,
                        'loss_rpn_loc':options.loss_rpn_loc,
                        'loss_cls':options.loss_cls,
                        'loss_box_reg':options.loss_box_reg,
                        'loss_mask':options.loss_mask}
        return InstanceSegmenter(enable_seg_losses=options.enable_seg_losses, loss_weights=loss_weights)