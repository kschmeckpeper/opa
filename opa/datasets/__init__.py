
from datasets.shapestacks_dataset import ShapeStacksDataset

def get_dataset_from_options(options, is_train=True):

    return ShapeStacksDataset(is_train=is_train,
                            sequence_length=options.sequence_length,
                            context_length=options.context_length,
                            rand_start=options.rand_start)
