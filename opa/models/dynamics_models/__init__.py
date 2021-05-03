
from .encoded_graph_dynamics_model import EncodedGraphDynamicsModel

def get_dynamics_model_from_options(options):
    """ Gets the dynamics model specified by the options
    """
    if options.dynamics_model == 'encoded_graph':
        if options.dataset == 'robonet':
            image_size = [96,128]
            action_size = 4
        elif options.dataset =='shapestacks':
            image_size = [112,112] # [224,224]
            action_size = 2

        if options.feature_extractor=="instance_segmenter":
            params = {
                    'num_input_features': 257,
                    'action_size': action_size, #4,
                    'patch_size': 14,
                    'output_image_size': image_size,
                    'latent_size': 64, # 32
                    'conv_features': 128,
                }
        #elif options.dataset == 'shapestacks':
        elif options.feature_extractor=="precropped":
            params = {
                    'num_input_features': 4,
                    'conv_features': 128,
                    'action_size': action_size, #2,
                    'patch_size': 224,
                    'output_image_size': image_size,
                    'latent_size': 64,
                }
        else:
            raise Exception('Do not recognize given feature extractor!!')

        params['no_graph'] = options.no_graph
        params['zero_init'] = options.zero_init
        return EncodedGraphDynamicsModel(**params)
