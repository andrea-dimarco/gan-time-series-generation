from torch import nn, Tensor, zeros
from torch.nn.init import normal_
import torch

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 device:torch.device, use_activation:bool=True, num_layers=3,
                 module_type='gru', normalize=False) -> None:
        '''
        The Generator takes a sequence of iid samples and generates a sequence in the latent space.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - hidden_size: dimensionality of the sample returned by the module
            - num_layers: depth of the module
            - output_size: size of the final output
            - seq_len: length of the sequence to embed
            - normalize: whether to normalize the samples or not
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        self.module_type = module_type
        self.num_layers = num_layers
        self.num_final_layers = int(num_layers/3+1)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.normalize = normalize
        self.dev = device
        self.use_activation = use_activation
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Wrong module_type argument.")

        self.fc = nn.Linear(hidden_size, output_size)

        self.activation = nn.Sigmoid()

        # Normalization
        if self.normalize:
            self.norm = nn.InstanceNorm1d(output_size, affine=True)
        else:
            self.norm = None


    def forward(self, x: Tensor) -> Tensor:
        '''
        Forward pass
        '''
        if self.module_type == 'lstm':
            out, _ = self.module(x) # shape = ( batch_size, seq_len, output_size )
        else:
            out, _ = self.module(x) # shape = ( batch_size, seq_len, output_size )

        if self.normalize:
            # required shape (batch_size, output_size, seq_len )
            out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        out = self.fc(out)

        if self.use_activation:
            out = self.activation(out)

        return out



def init_weights(m):
    '''
    Initialized the weights of the Linear layer.
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.zeros_(m.bias)