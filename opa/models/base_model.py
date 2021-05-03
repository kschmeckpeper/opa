import torch.nn as nn

class BaseModel(nn.Module):
    """ The base model that all prediction models should extend.
    """
    def __init__(self):
        super().__init__()

    def forward(self, batch): # pylint: disable=arguments-differ
        """
        Inputs:
            batch - dictionary
        
        Outputs:
            output - dictionary containing the following keys
                images - dictionary containing all images to visualize
                losses - dictionary containing all of the losses used to train the model
                metrics - dictionary containing all metrics to log
        """
        raise NotImplementedError
