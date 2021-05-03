from .addition_generator import AdditionGenerator

def get_image_generator_from_options(options):
    """ Gets the image generator from the options
    """
    #if options.dataset == 'robonet':
    if options.feature_extractor=="spxl_segmenter":
        params = {'reduce_features': True, 'small_reduction':True}
    elif options.feature_extractor=="instance_segmenter":
        params = {'reduce_features': True, 'small_reduction':False}
    elif options.feature_extractor=="precropped":
        params = {'reduce_features': False}
    else:
        raise Exception('Do not recognize given feature extractor!!')

    params['final_synth'] = options.final_synth
    return AdditionGenerator(**params)
