

import copy

import torch.nn.functional as F
from torch import nn
import torch

from torch import Tensor
import lib
import typing as ty

class AEWrapper(nn.Module):
    """
    Autoencoder wrapper class
    """

    def __init__(self, options):
        """

        Args:
            options (dict): Configuration dictionary.
        """
        super(AEWrapper, self).__init__()
        self.options = options
        # self.encoder = ShallowEncoder(options) if options["shallow_architecture"] else Encoder(options)
        # self.encoder = MultiheadAttention(options)
        self.encoder = ResNet(options)
        self.decoder = ShallowDecoder(options) if options["shallow_architecture"] else Decoder(options)
        # self.EmptyModel = EmptyModel()
        
        # Get the last dimension of encoder. This will also be used as the dimension of projection
        input_dim = self.options["dims"][0]
        output_dim = self.options["dims"][1]
        th_dim = self.options["dims"][-1]
        # Two-Layer Projection Network
        # First linear layer, which will be followed with non-linear activation function in the forward()
        self.linear_layer1 = nn.Linear(output_dim, output_dim)
        self.linear_layer2 = nn.Linear(output_dim, th_dim)
        self.linear_layer3 = nn.Linear(th_dim, input_dim)
        self.dropout = nn.Dropout(p=self.options['dropout_rate'])

    def forward(self, x):
        # Forward pass on Encoder
        latent = self.encoder(x, None)
        z = latent
        # Forward pass on Projection
        # # Apply linear layer followed by non-linear activation to decouple final output, z, from representation layer h.
        # # z = F.leaky_relu(self.linear_layer1(latent))

        # z = self.linear_layer1(latent)
        # z = F.leaky_relu(z)
        # # z = self.linear_layer1(z)
        # z = F.leaky_relu(z)
        # z = self.dropout(z)

        # z = self.linear_layer2(z)
        z = F.normalize(z, p=self.options["p_norm"], dim=1) if self.options["normalize"] else z

        # z = self.dropout(z)
        # # Apply final linear layer
        # z = self.linear_layer3(z)
        # # Do L2 normalization
        # z = F.normalize(z, p=self.options["p_norm"], dim=0) if self.options["normalize"] else z
        # z = self.dropout(z)
        # Forward pass on decoder
        # print(latent.shape)
        x_recon = self.decoder(z)
        
        # Return 
        return z, x_recon


class Encoder(nn.Module):
    def __init__(self, options):
        """Encoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(Encoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # Compute the shrunk size of input dimension
        # n_column_subset = int(self.options["dims"][0] / self.options["n_subsets"])
        n_column_subset = self.options["dims"][0]
        # Ratio of overlapping features between subsets
        overlap = self.options["overlap"]
        # Number of overlapping features between subsets
        n_overlap = int(overlap * n_column_subset)
        # Overwrie the input dimension
        self.options["dims"][0] = n_column_subset + n_overlap
        # Forward pass on hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        # Compute the latent i.e. bottleneck in Autoencoder
        self.latent = nn.Linear(self.options["dims"][-2], self.options["dims"][1])

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute the mean i.e. bottleneck in Autoencoder
        latent = self.latent(h)
        return latent


class Decoder(nn.Module):
    def __init__(self, options):
        """Decoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(Decoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)
        # If recontruct_subset is True, output dimension is same as input dimension of Encoder. Otherwise, 
        # output dimension is same as original feature dimension of tabular data
        if self.options["reconstruction"] and self.options["reconstruct_subset"]:
            # Compute the shrunk size of input dimension
            # n_column_subset = int(self.options["dims"][0] / self.options["n_subsets"])
            n_column_subset = self.options["dims"][0] 
            # Overwrie the input dimension
            self.options["dims"][0] = n_column_subset
        # Revert the order of hidden units so that we can build a Decoder, which is the symmetric of Encoder
        # self.options["dims"] = self.options["dims"][::-1]
        # Add hidden layers
        self.hidden_layers = HiddenLayers(self.options)
        # Compute logits and probabilities
        self.logits = nn.Linear(self.options["dims"][-2],1)

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        # Compute logits
        logits = self.logits(h)
        return logits

    
class ShallowEncoder(nn.Module):
    def __init__(self, options):
        """Encoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(ShallowEncoder, self).__init__()
        # Deepcopy options to avoid overwriting the original
        self.options = copy.deepcopy(options)  
        # Compute the shrunk size of input dimension
        # n_column_subset = int(self.options["dims"][0]/self.options["n_subsets"])
        n_column_subset = self.options["dims"][0] 
        # Ratio of overlapping features between subsets
        overlap = self.options["overlap"]
        # Number of overlapping features between subsets
        n_overlap = int(overlap*n_column_subset)
        # Overwrie the input dimension
        self.options["dims"][0] = n_column_subset + n_overlap
        # Forward pass on hidden layers
        # self.hidden_layers = HiddenLayers(self.options)
        # self.hidden_layers = TabularCNN(self.options)
        self.hidden_layers = AttentionRegressionModel(self.options)
        

    def forward(self, h):
        # Forward pass on hidden layers
        h = self.hidden_layers(h)
        return h
    
    
class ShallowDecoder(nn.Module):
    def __init__(self, options):
        """Decoder model

        Args:
            options (dict): Configuration dictionary.
        """
        super(ShallowDecoder, self).__init__()
        # Get configuration that contains architecture and hyper-parameters
        self.options = copy.deepcopy(options)
        # Input dimension of predictor == latent dimension
        input_dim, output_dim = self.options["dims"][0],  1
        # First linear layer with shape (bottleneck dimension, output channel size of last conv layer in CNNEncoder)
        self.first_layer = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        logits = self.first_layer(z)
        return logits
    
    
class HiddenLayers(nn.Module):
    def __init__(self, options):
        """Class to add hidden layers to networks

        Args:
            options (dict): Configuration dictionary.
        """
        super(HiddenLayers, self).__init__()
        self.layers = nn.ModuleList()
        dims = options["dims"]

        for i in range(1, len(dims) - 1):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if options["isBatchNorm"]:
                self.layers.append(nn.BatchNorm1d(dims[i]))

            self.layers.append(nn.LeakyReLU(inplace=False))
            if options["isDropout"]:
                self.layers.append(nn.Dropout(options["dropout_rate"]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TabularCNN(nn.Module):
    def __init__(self, options):
        super(TabularCNN, self).__init__()
        num_features = options['dims'][0]
        self.conv1 = nn.Conv1d(num_features, num_features*2, kernel_size=5, padding=3, stride=3)
        self.conv2 = nn.Conv1d(num_features*2, num_features*3, kernel_size=5, padding=3, stride=3)
        self.conv3 = nn.Conv1d(num_features*3, num_features*4, kernel_size=5, padding=3, stride=3)
        self.fc1 = nn.Linear(num_features*4, num_features*3)
        self.fc2 = nn.Linear(num_features*3, num_features*2)
        self.fc3 = nn.Linear(num_features*2, num_features)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(num_features*2)
        self.bn2 = nn.BatchNorm1d(num_features*3)
        self.bn3 = nn.BatchNorm1d(num_features*4)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = x.unsqueeze(2)  # Add a dummy dimension for 1D convolution
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.squeeze(2)  # Remove the dummy dimension
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.norm3(x)
        x = self.fc3(x)
        # x = self.pool()
        return x.squeeze(1)

class ResNet(nn.Module):
    def __init__(
        self,options) -> None:
        # 8 None None 310 3.1455680991172583 5 relu batchnorm 0.30138168803582194 0.21879360563134626 1
        super().__init__()
        d_feats = 90 #136 #8 for ca 90 year
        d_numerical= d_feats
        categories= None
        d_embedding= None
        d= 310
        d_hidden_factor= 3.1455680991172583
        n_layers= 5
        activation= 'relu'
        normalization= 'batchnorm'
        hidden_dropout= 0.30138168803582194
        residual_dropout= 0.21879360563134626
        d_out= d_feats
        # print('here')

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # print(f'{self.category_embeddings.weight.shape=}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: None) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x