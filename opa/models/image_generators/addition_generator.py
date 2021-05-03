import torch
import torch.nn as nn

from models.networks.hourglass import Hourglass

class AdditionGenerator(nn.Module):
    """ Generates a predicted image by taking the
    sum of all the segmentations, weighted by their masks
    """
    def __init__(self,
                 reduce_features=False,
                 small_reduction=False,
                 final_synth=False):
        super().__init__()

        self._reduce_features = reduce_features

        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.conv.weight.data.fill_(0.0)
        self.conv.weight.data[1, 1] = 1
        self.conv.bias.data.fill_(0.0)

        self.hourglass = nn.Sequential(Hourglass(n=2, in_channels=3, out_channels=3, n_features=32),
                                       nn.Sigmoid()
                                       )

        if self._reduce_features:
            self.feature_reducer = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 16, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(16, 3, kernel_size=3, padding=1),
                                         nn.Sigmoid()) 
        else:
            self.feature_reducer = nn.Sigmoid()

        self.final_synth = None

        if final_synth:
            self.final_synth = nn.Sequential(Hourglass(n=2, in_channels=3, out_channels=3, n_features=32),
                                                 nn.Sigmoid()
                                                 )

    def forward(self, graph, prev_image, background_masks):# pylint: disable=arguments-differ
        """
        segments is a list of instances.  Channel 1 is the mask
        and the rest are the image
        """

        all_combined_images = []
        all_combined_masks = []
        curr_batch = -1

        all_images = []
        all_masks = []
        for node in range(len(graph.x)):
            if graph.batch[node] != curr_batch:
                if curr_batch != -1:
                    all_images.append(uncombined_image)
                    all_masks.append(uncombined_masks)
                uncombined_image = [graph.x[node][1:]]
                uncombined_masks = [graph.x[node][0]]
                curr_batch = graph.batch[node]
            else:
                uncombined_image.append(graph.x[node][1:])
                uncombined_masks.append(graph.x[node][0])
        all_images.append(uncombined_image)
        all_masks.append(uncombined_masks)

        all_post_softmax_object_masks = []
        for i in range(len(all_images)):
            stacked_masks = torch.stack(all_masks[i])
            stacked_masks = torch.cat([stacked_masks, torch.ones_like(stacked_masks[0]).view(1, stacked_masks.shape[1], stacked_masks.shape[2])*0.1], dim=0)
            stacked_masks = torch.nn.functional.softmax(stacked_masks, dim=0)
            all_post_softmax_object_masks.append(stacked_masks[:-1])
            all_combined_masks.append(torch.sum(stacked_masks[:-1], dim=0))
            
            reduced_images = self.feature_reducer(torch.stack(all_images[i]))

            combined_image = reduced_images[0] * stacked_masks[0]
            for j in range(1, min(len(all_images[i]), reduced_images.shape[0])):
                combined_image += reduced_images[j] * stacked_masks[j]
            all_combined_images.append(combined_image)

        all_combined_images = torch.stack(all_combined_images)

        all_post_softmax_object_masks = torch.cat(all_post_softmax_object_masks)

        all_combined_masks = torch.stack(all_combined_masks).view(background_masks.shape)

        object_pixels = all_combined_images.clone().detach()

        synthetic_mask = torch.ones_like(background_masks) - background_masks - all_combined_masks.view(background_masks.shape)
        synthetic_mask = torch.clamp(synthetic_mask, 0.0, 1.0)
        original_background_masks = background_masks.clone().detach()
        background_masks = background_masks - all_combined_masks
        background_masks = torch.clamp(background_masks, 0.0, 1.0)


        all_combined_images = prev_image * background_masks + all_combined_images

        object_and_background = all_combined_images.detach()

        masked_background = (prev_image * background_masks).detach()

        synthetic_pixels = self.hourglass(all_combined_images)


        all_combined_images = all_combined_images + synthetic_mask * synthetic_pixels

        masked_synth = (synthetic_mask * synthetic_pixels).detach()



        diagnostics = {'images':{}, 'metrics':{}}
        diagnostics['images'] = {'synthetic_masks': synthetic_mask,
                                 'combined_object_masks': all_combined_masks,
                                 'object_pixels': object_pixels,
                                 'background_masks': background_masks,
                                 'masked_background_pixels': masked_background,
                                 'synthetic_pixels': synthetic_pixels,
                                 'masked_synthetic_pixels': masked_synth,
                                 'object_and_background': object_and_background,
                                 'post_softmax_object_masks': all_post_softmax_object_masks
                                 }
        if self.final_synth is not None:
            diagnostics['images']['final_synth_pre'] = all_combined_images.clone()
            all_combined_images = self.final_synth(all_combined_images.detach())

        return all_combined_images, diagnostics

