#from .pytorch_utils.base_options import BaseOptions
from pytorch_utils.base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    """ Parses command line arguments for training
    This overwrites options from BaseOptions
    """
    def __init__(self): # pylint: disable=super-init-not-called
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=360000000,
                         help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False,
                         action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0, # 4
                         help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                         help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                         help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        in_out = self.parser.add_argument_group('io')
        in_out.add_argument('--log_dir', default='logs', help='Directory to store logs')
        in_out.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        in_out.add_argument('--from_json', default=None,
                            help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=100000,
                           help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=1, help='Batch size')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                                  help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false',
                                  help='Don\'t shuffle testing data')
        train.set_defaults(shuffle_train=True, shuffle_test=True)
        train.add_argument('--summary_steps', type=int, default=10,
                           help='Summary saving frequency')
        train.add_argument('--image_summary_steps', type=int, default=5000,
                           help='Image summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000,
                           help='Chekpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=1000, help='Testing frequency') # 500
        train.add_argument('--dataset', type=str, required=True,
                           choices=['shapestacks'])
        train.add_argument('--sequence_length', type=int, default=1)
        train.add_argument('--context_length', type=int, default=1)


        train.add_argument('--coord_loss_scale', type=float, default=10**-5)
        train.add_argument('--patch_loss_scale', type=float, default=0.0001)
        # GG
        train.add_argument('--seg_loss_scale', type=float, default=1.0) 
        train.add_argument('--pred_attn_loss_scale', type=float, default=1.0)
        train.add_argument('--pred_loss_scale', type=float, default=1.0)
        # Mask rcnn specific loss weights
        train.add_argument('--loss_rpn_cls', type=float, default=0.1)
        train.add_argument('--loss_rpn_loc', type=float, default=0.1)
        train.add_argument('--loss_cls', type=float, default=0.1)
        train.add_argument('--loss_box_reg', type=float, default=0.1)
        train.add_argument('--loss_mask', type=float, default=0.1)

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                           default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                           default=0, help="Weight decay weight")

        self.parser.add_argument('--test_iters', type=int, default=5)

        optimizer_options = self.parser.add_argument_group('Optimizer')
        optimizer_options.add_argument('--lr', type=float, default=1e-3)
        # Learning rates specific to instance_segmenter (Mask Rcnn) components
        optimizer_options.add_argument('--lr_backbone', type=float, default=0)
        optimizer_options.add_argument('--lr_proposals', type=float, default=1e-6)
        optimizer_options.add_argument('--lr_rois', type=float, default=1e-6)
        optimizer_options.add_argument('--lr_flow', type=float, default=0)
        # Learning rate specific to spxl_segmenter
        optimizer_options.add_argument('--lr_spxl', type=float, default=1e-5)


        model_options = self.parser.add_argument_group('Model')

        model_options.add_argument('--model_type', type=str, required=True,
                                   choices=['accumulated_dynamics'])
        model_options.add_argument('--dynamics_model', type=str, required=False,
                                    choices=['encoded_graph'],
                                    default='none')
        model_options.add_argument('--feature_extractor', type=str, required=True,
                                    choices=['dummy_color',
                                             'precropped',
                                             'instance_segmenter'])
        model_options.add_argument('--detach_extractor',
                                   action='store_true',
                                   dest='detach_extractor',
                                   help='Detach gradients from the feature extractor')
        model_options.add_argument('--no_graph',
                                   action='store_true',
                                   dest='no_graph',
                                   help='Replace graph convs with linear layers')

        model_options.add_argument("--recon_from_first",
                                   action="store_true",
                                   dest="recon_from_first",
                                   help="Generate image from the first input image")
        model_options.add_argument("--mask_patch_loss",
                                   action="store_true",
                                   dest="mask_patch_loss",
                                   help="Multiply patches by their masks before calculating the patch loss")

        model_options.add_argument("--zero_init",
                                   action="store_true",
                                   dest="zero_init",
                                   help="Initialize the weights and biases of the last conv layer of the decoder to zero")
        model_options.add_argument("--rand_start",
                                   action="store_true",
                                   dest="rand_start",
                                   help="Train with sequences with random starting times")

        model_options.add_argument("--final_synth",
                                   action="store_true",
                                   dest="final_synth",
                                   help="Add final set of synthetic pixels")

        # GG added option
        model_options.add_argument("--enable_seg_losses",
                                    action="store_true",
                                    dest="enable_seg_losses",
                                    help="Generate pseudo gt for Mask RCNN case and backprop its losses")
