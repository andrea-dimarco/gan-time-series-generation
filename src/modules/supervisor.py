from torch import nn, Tensor, zeros
from torch.nn.init import normal_


class Supervisor(nn.Module):
    def __init__(self, input_size, var=0.02, num_layers=3, module_type='gru') -> None:
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
        self.output_size = input_size
        
        # input.shape = ( batch_size, seq_len, feature_size )
        if self.module_type == 'rnn':
            self.module = nn.RNN(input_size, input_size, num_layers, batch_first=True)
        elif self.module_type == 'gru':
            self.module = nn.GRU(input_size, input_size, num_layers, batch_first=True)
        elif self.module_type == 'lstm':
            self.module = nn.LSTM(input_size, input_size, num_layers, batch_first=True)
        else:
            assert(False)

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
        h0 = zeros(self.num_layers, batch_size, self.output_size) # initial state

        if self.module_type == 'lstm':
            c0 = zeros(self.num_layers, batch_size, self.output_size)
            out, _ = self.module(x, (c0, h0)) # shape = ( batch_size, seq_len, output_size )
        else:
            out, _ = self.module(x, h0) # shape = ( batch_size, seq_len, output_size )

        out = self.activation(out)
        return out