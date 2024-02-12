from torch import nn, Tensor, zeros
from torch.nn.init import normal_, xavier_uniform_, zeros_
import torch


class Discriminator(nn.Module):
    def __init__(self, input_size:int, hidden_size:int,
                 device:torch.device, var=0.02,
                 num_layers=3, normalize=True, 
                 module_type='gru'
                 ) -> None:
        '''
        The discriminator reasons over the presented sequence and returns whether it is legitimate or not.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - hidden_size: dimensionality of the sample returned by the module
            - num_layers: depth of the module
            - seq_len: length of the sequence to embed
            - device: which device should the model run on
            - alpha: parameter for the LeakyReLU function
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        self.module_type = module_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dev = device
        self.normalize = normalize

        if normalize:
            self.norm = nn.BatchNorm1d(input_size, affine=False) # <- ( batch_size, feature_size, seq_len )
        else:
            self.norm = None
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            assert(False)

        # extra linear layer
        self.block = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # initialize weights
        for layer_p in self.module._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    normal_(self.module.__getattr__(p), 0.0, var)
        self.block.apply(init_weights)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        out = x # <- ( batch_size, seq_len, input_size)
        
        if self.normalize:
            out = self.norm(out.permute(0,2,1)).permute(0,2,1) # <- needs ( batch_size, input_size, seq_len )

        if self.module_type == 'lstm':
            out, _ = self.module(out) # shape = ( batch_size, seq_len, hidden_size )
        else:
            out, _ = self.module(out) # shape = ( batch_size, seq_len, hidden_size )

        out = out[:,-1,:] # only consider the last output of each sequence

        out = self.block(out)
        return out


def init_weights(m):
    '''
    Initialized the weights of the nn.Sequential block
    '''
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            zeros_(m.bias)