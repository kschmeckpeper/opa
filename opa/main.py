
""" Entry point for training
"""

from train_options import TrainOptions
from trainer import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
