from torch import nn, Tensor, zeros
from torch.nn.init import normal_
import torch

class Supervisor(nn.Module):
    def __init__(self, input_size:int, hidden_size:int,
                 device:torch.device, normalize:bool=False,
                 num_layers:int=3, module_type:str='gru'
                ) -> None:
        '''
        The Supervisors takes a sequence in the latent space and returns a new sequence in the latent space.
         This agent aims to close the DIFFERENCES between the latent space mapped by te EMBEDDER 
         and the latent space spanned by the GENERATOR, they must be the same latent space.
        The supervisor will need to behave as the identity function when presented sequences mapped from the
         real data by the EMBEDDER and only modify synthetic sequences made by the GENERATOR.
        The loss from the supervisor will lead the MBEDDER and the GENERATOR to span the same space.
        Args:
            - module_type: what module to use between RNN, GRU and LSTM
            - input_size: dimensionality of one sample
            - num_layers: depth of the module
            - seq_len: length of the sequence to embed
            - device: which device should the model run on
        '''
        assert(module_type in ['rnn', 'gru', 'lstm'])

        super().__init__()
        self.module_type = module_type
        self.num_layers = num_layers
        self.num_final_layers = int(num_layers/3+1)
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.normalize = normalize
        self.dev = device
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Wrong module_type argument.")

        self.fc = nn.Linear(hidden_size, input_size)

        # Normalization
        if self.normalize:
            self.norm = nn.InstanceNorm1d(input_size, affine=True)
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

        return out



def init_weights(m):
    '''
    Initialized the weights of the Linear layer.
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.zeros_(m.bias)