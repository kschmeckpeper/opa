import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torch_geometric.nn import GraphConv, InstanceNorm

def extract_patch_from_frame(image, coordinates, output_shape):
    """ This should be the inverse operation to translate_and_scale_patch
    """

    # Translate from coordinates in (x_center, y_center, w, h) to (minx, miny, maxx, maxy)
    xyxy = torch.zeros_like(coordinates)
    xyxy[:, 0] = coordinates[:, 0] - image.shape[-2] * coordinates[:, 2] / 2
    xyxy[:, 1] = coordinates[:, 1] - image.shape[-1] * coordinates[:, 3] / 2
    xyxy[:, 2] = coordinates[:, 0] + image.shape[-2] * coordinates[:, 2] / 2
    xyxy[:, 3] = coordinates[:, 1] + image.shape[-1] * coordinates[:, 3] / 2
    xyxy_with_index = torch.cat((torch.arange(xyxy.shape[0], dtype=xyxy.dtype, device=xyxy.device).view(-1, 1), xyxy), dim=1)

    patches = roi_align(image, xyxy_with_index, output_shape, aligned=True)
    return patches

def translate_and_scale_patch(decoded, coordinates, shape, mask_background=0.0):
    min_x = coordinates[:, 0] - coordinates[:, 2] * shape[-2] / 2
    min_y = coordinates[:, 1] - coordinates[:, 3] * shape[-1] / 2

    decoded_full_size = torch.ones(shape, device=decoded.device) * mask_background
    decoded_full_size[:, :, :decoded.shape[2], :decoded.shape[3]] = decoded

    # Set all parts of the grid to out of bounds
    sampling_grid = torch.ones((shape[0], shape[2], shape[3], 2), device=decoded.device) * -10


    coordinates = coordinates.clone()
    coordinates[:, 2] *= (shape[-2] - 1) / (decoded.shape[-2] - 1)
    coordinates[:, 3] *= (shape[-1] - 1) / (decoded.shape[-1] - 1)
    if coordinates.shape[1] == 2:
        # Transform is only x,y translation
        affine_matrix = torch.zeros((shape[0], 2, 3), device=decoded.device)
        affine_matrix[:, 0, 0] = 1
        affine_matrix[:, 1, 1] = 1
        affine_matrix[:, 0, 2] = -coordinates[:, 0] / (shape[-2]-1) * 2 + 4 / shape[-2]
        affine_matrix[:, 1, 2] = -coordinates[:, 1] / (shape[-1]-1) * 2 + 4 / shape[-1]
    elif coordinates.shape[1] == 4:
        affine_matrix = torch.zeros((shape[0], 2, 3), device=decoded.device)
        affine_matrix[:, 0, 0] = 1/coordinates[:, 2]
        affine_matrix[:, 1, 1] = 1/coordinates[:, 3]

        affine_matrix[:, 0, 2] = -1.0 + 1.0 / coordinates[:, 2]  \
                                 - 2*(min_x/(shape[-2] - 1)) / coordinates[:, 2]
        affine_matrix[:, 1, 2] = -1.0 + 1.0 / coordinates[:, 3] \
                                 - 2*(min_y/(shape[-1] - 1)) / coordinates[:, 3]

    affine_grid = F.affine_grid(affine_matrix, decoded_full_size.shape, align_corners=True)
    decoded_full_size[:, :, :, :] -= mask_background
    decoded_image = F.grid_sample(decoded_full_size, affine_grid, align_corners=True)
    decoded_image[:, :, :, :] += mask_background
    return decoded_image


class EncodedGraphDynamicsModel(nn.Module):
    """ Encodes the input feature space to a single
    latent vector for each instance, then uses
    graph convs to predict future states
    """
    def __init__(self,
                 num_input_features=257,
                 latent_size=64,
                 conv_features=128,
                 action_size=4,
                 transform_size=4,
                 patch_size=14,
                 output_image_size=[96, 128],
                 num_layers=4,
                 no_graph=False,
                 zero_init=False):
        super().__init__()
        self._transform_size = transform_size
        self._output_image_size = output_image_size
        self._no_graph = no_graph

        if patch_size==14:
            # Set apply_logits_to_prev to True if the input patch is in the range [0, 1], and false otherwise
            self._apply_logits_to_prev = False
            self.encoder = nn.Sequential(nn.Conv2d(num_input_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, latent_size, 3, padding=1),
                                         nn.AvgPool2d(kernel_size=2, stride=2))

            self.decoder = nn.Sequential(nn.ConvTranspose2d(latent_size, conv_features, 3, stride=2, output_padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(conv_features, num_input_features, 3, padding=1),
                                         )
        elif patch_size==224:
            self._apply_logits_to_prev = True
            self.encoder = nn.Sequential(nn.Conv2d(num_input_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=3, stride=2),
                                         nn.Conv2d(conv_features, latent_size, 3, padding=1),
                                         nn.AvgPool2d(kernel_size=2, stride=2))

            self.decoder = nn.Sequential(nn.ConvTranspose2d(latent_size, conv_features, 3, stride=2, output_padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(conv_features, conv_features, 3, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=0),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, padding=1, stride=2, output_padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose2d(conv_features, conv_features, 3, stride=2, padding=1, output_padding=1),
                                         nn.ReLU(),
                                         nn.Conv2d(conv_features, num_input_features, 3, padding=1),
                                         )
        
        if zero_init:
            # Set weights and bias of last layer to zero so the model initially uses prev_patch
            self.decoder[-1].weight.data.fill_(0.0)
            self.decoder[-1].bias.data.fill_(0.0)


        if not self._no_graph:
            graph_convs = [GraphConv(latent_size + action_size + self._transform_size, latent_size, aggr="mean")]
            for i in range(num_layers):
                graph_convs.append(GraphConv(latent_size, latent_size, aggr="mean"))
            graph_convs.append(GraphConv(latent_size, latent_size+self._transform_size, aggr="mean"))
        else:
            graph_convs = [nn.Linear(latent_size + self._transform_size + action_size, latent_size)]
            for i in range(num_layers):
                graph_convs.append(nn.Linear(latent_size, latent_size))
            graph_convs.append(nn.Linear(latent_size, latent_size + self._transform_size))

        self.graph_convs = nn.ModuleList(graph_convs)
        
        self.scalar = nn.Parameter(torch.tensor(2.0, requires_grad=True))

    def forward(self, graph, action, encoded=None, coordinates=None, prev_patch=None): # pylint: disable=arguments-differ
        device = graph.x.device
        if encoded is None:
            aligned = graph.x

            if self._apply_logits_to_prev:
                # Moves aligned away from asymptotes to improve numerical stability
                scaled_aligned = aligned * 0.999 + 0.0005

                prev_patch = torch.log(scaled_aligned) - torch.log(1.0 - scaled_aligned)
            else:
                prev_patch = aligned
            encoded = self.encoder(aligned)

            if self._transform_size == 6:
                ones = torch.ones(coordinates.shape[0], 1, device=device)
                zeros = torch.zeros(coordinates.shape[0], 1, device=device)
                coordinates = torch.cat([ones, zeros, coordinates[:, 1].view(-1, 1), zeros, ones, coordinates[:, 0].view(-1, 1)], axis=1)

            encoded = encoded.view(encoded.shape[0], encoded.shape[1])
            encoded = torch.cat([encoded, coordinates.to(encoded.device)], axis=1)
        selected_action = action[graph.batch]

        prev_encoded = encoded.clone()
        encoded = torch.cat([encoded, selected_action], dim=1)

        
        for i, conv in enumerate(self.graph_convs):
            if not self._no_graph and not isinstance(conv, InstanceNorm):
                encoded = conv(encoded, graph.edge_index)
            else:
                encoded = conv(encoded)
            if i < len(self.graph_convs) - 1:
                encoded = F.relu(encoded)
        decoded = self.decoder(encoded[:, :-self._transform_size] .view(encoded.shape[0], -1, 1, 1))
        # Add original
        decoded = decoded + prev_patch
        

        coordinates[:, :2] = coordinates[:, :2] + encoded[:, -self._transform_size:-2]


        target_shape = [graph.x.shape[0],
                        graph.x.shape[1],
                        self._output_image_size[0],
                        self._output_image_size[1]]


        decoded_image = translate_and_scale_patch(decoded, coordinates, target_shape, mask_background=-10**5)

        dynamics_bboxes = self.convert_to_bbox_img(coordinates.clone(), img_size=self._output_image_size)       

        graph.x = decoded_image
        prev_output = {'encoded': encoded, 'prev_patch': decoded, 'coordinates': coordinates}
        return graph, prev_output, dynamics_bboxes


    def paste_mask_in_image(self, mask, box, img_size, mask_background=0.0):
        H,W = img_size
        num_channels = mask.shape[0]
        x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_raw_mask = torch.ones((num_channels, H, W), dtype=torch.float32) * mask_background
        
        mask_w, mask_h = x2-x1, y2-y1

        mask_tmp_resized = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode='nearest')
        img_raw_mask[:, y1:y2, x1:x2] = mask_tmp_resized[0,:,:,:]
        return img_raw_mask


    def convert_to_bbox_img(self, norm_bbox, img_size):
        # Convert center_x, center_y, norm_width, norm_height to x1,y1,x2,y2
        H,W = img_size
        out_bboxes = []
        for i in range(norm_bbox.shape[0]):
            box = norm_bbox[i,:].clone()
            box_w = box[2] * W
            box_h = box[3] * H
            x1 = box[0] - box_w/2
            y1 = box[1] - box_h/2
            x2 = x1 + box_w
            y2 = y1 + box_h
            # check for bounds
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 >= W:
                x2 = W-1
            if y2 >= H:
                y2 = H-1
            # check for degenerate boxes
            if x1 >= x2:
                x1 = x2-1
            if y1 >= y2:
                y1 = y2-1
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0            
            out_bboxes.append(torch.tensor([x1,y1,x2,y2], device=norm_bbox.device))
        out_bboxes = torch.stack(out_bboxes)
        return out_bboxes
