import argparse
import json
from collections import namedtuple

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import get_dataset_from_options
from models import get_model_from_options
from pytorch_utils import CheckpointSaver

import lpips
import numpy as np
import cv2
from os.path import join

class Tester():
    def __init__(self, options):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = options

        self.test_ds = get_dataset_from_options(self.options, is_train=False)

        self.model = get_model_from_options(self.options)
        self.models_dict = {'model':self.model}

        self.models_dict = {k:v.to(self.device)
                for k,v in self.models_dict.items()}

        self.optimizers_dict = {}
        # Load the latest checkpoints
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.checkpoint = None
        if self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict,
                    self.optimizers_dict,
                    checkpoint_file=self.options.checkpoint)
        else:
            print("No checkpoint found.  The program  will exit")

        for model in self.models_dict:
            self.models_dict[model].eval()

        self.metrics = {'lpips': lpips.LPIPS(net='alex').cuda(),
                        'mse': torch.nn.MSELoss()
                        }

        self.images_to_save = ['target_frames', 'pred', 'generator/object_pixels']

        self.image_path = "out_images"


    def add_stepwise_metrics(self, batch, out, metrics):
        for metric in self.metrics:
            for i in range(batch['target_frames'].shape[1]):
                metrics['stepwise_{}_{:02d}'.format(metric, i)] = self.metrics[metric](batch['target_frames'][:, i], out['images']['pred'][:, i])

                if metric == "lpips":
                    metrics['stepwise_{}_{:02d}'.format(metric, i)] = torch.mean(metrics['stepwise_{}_{:02d}'.format(metric, i)] )
        return metrics

    def save_images(self, batch, out_data, count):
        for k in self.images_to_save:
            if k in out_data['images']:
                im = out_data['images'][k]
            else:
                im = batch[k]


            for i in range(im.shape[0]):
                for t in range(im.shape[1]):
                    image = im[i, t].permute(1, 2, 0).detach().cpu().numpy()
                    name = "{}_{:04d}_{:02d}.png".format(k, i+count, t)
                    name = name.replace('/', '_')
                    cv2.imwrite(join(self.image_path, name), image * 255)

    def test(self):
        test_data_loader = DataLoader(self.test_ds,
                                      batch_size=1,#self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=False)
        batch = None

        metrics = {}
        count = 0
        for tstep, batch in enumerate(tqdm(test_data_loader,
                                           desc='Testing')):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                out_data = self.model(batch)
                
                out_data['metrics'] = self.add_stepwise_metrics(batch, out_data, out_data['metrics'])


                self.save_images(batch, out_data, count)

                batch_size = batch['context_frames'].shape[0]
                print("batch_size:", batch_size)
                count += batch_size
                for metric in out_data['metrics']:
                    if metric not in metrics:
                        metrics[metric] = [ out_data['metrics'][metric] * batch_size]
                    else:
                        metrics[metric].append( out_data['metrics'][metric] * batch_size)
        
        print("\n\nTested on {} sequences".format(count))
        for metric in metrics:
            metrics[metric] = torch.tensor(metrics[metric])
            print(metric, ": ", torch.mean(metrics[metric]).item(), " +- ", torch.std(metrics[metric]).item() / np.sqrt(count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    with open(args.path, "r") as f:
        json_args = json.load(f)
        json_args['rand_start'] = False
        json_args = namedtuple("json_args", json_args.keys())(**json_args)

    print("json_args", json_args)

    tester = Tester(json_args)

    tester.test()
