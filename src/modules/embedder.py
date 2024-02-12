
from torch import nn, Tensor, zeros
from torch.nn.init import normal_
import torch


class Embedder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 device:torch.device, var=0.02, num_layers=3,
                 module_type='gru', normalize=False
                 ) -> None:
        '''
        The Embedder maps the input sequence to a lower dimensionality representation.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - hidden_size: dimensionality of the sample returned by the module
            - num_layers: depth of the module
            - output_size: size of the final output
            - normalize: whether to normalize the samples or not
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        self.module_type = module_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.normalize = normalize
        self.dev=device
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, output_size, num_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, output_size, num_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, output_size, num_layers, batch_first=True)
        else:
            assert(False)

        # Normalization
        if self.normalize:
            self.norm = nn.InstanceNorm1d(output_size, affine=True)
        else:
            self.norm = None

        # Activation
        self.activation = nn.Sigmoid()

        # initialize weights
        for layer_p in self.module._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    normal_(self.module.__getattr__(p), 0.0, var)


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        batch_size = x.size()[0]
        h0 = zeros(self.num_layers, batch_size, self.output_size, device=self.dev) # initial state

        if self.module_type == 'lstm':
            c0 = zeros(self.num_layers, batch_size, self.output_size, device=self.dev)
            out, _ = self.module(x, (c0, h0)) # shape = ( batch_size, seq_len, hidden_size )
        else:
            out, _ = self.module(x, h0) # shape = ( batch_size, seq_len, hidden_size )

        if self.normalize:
            # required shape (batch_size, output_size, seq_len )
            out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        out = self.activation(out)
        return out
