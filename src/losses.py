'''
This script contains the functions of the different losses.
'''

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from typing import Tuple
import torch



def discrimination_loss(Y_real: Tensor, Y_fake: Tensor,
                        Y_fake_e: Tensor, gamma: float=0.1
) -> torch.Tensor:
    '''
    This function computes the loss for the DISCRIMINATOR module.

    Arguments:
        - Y_real: Discriminator's results on the sequences of the EMBEDDINGS from the REAL data (H)
        - Y_fake: Discriminator's results on the sequences resulted from the SUPERVISOR with the GENERATED sequence (H_hat)
        - Y_fake_e: Discriminator's results on the GENERATED sequences (E_hat)
        - gamma: weight for the loss on Y_fake_e data TODO: find a good value for it

    Returns:
        - D_loss: float with the overall loss of the Discriminator module
    '''
    loss_f = CrossEntropyLoss()

    # Adversarial truth
    valid = torch.zeros_like(Y_real) + torch.tensor([1,0])
    fake  = torch.zeros_like(Y_real) + torch.tensor([0,1])

    loss_real   = loss_f(valid, Y_real)
    loss_fake   = loss_f(fake,  Y_fake)
    loss_fake_e = loss_f(fake,  Y_fake_e)
    D_loss = loss_real + loss_fake + gamma*loss_fake_e 

    return D_loss


def generation_loss(Y_fake: Tensor, Y_fake_e: Tensor,
                    X: Tensor, H: Tensor,
                    H_hat_supervise: Tensor, X_hat: Tensor,
                    gamma: float=0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    This function computes the loss for the DISCRIMINATOR module.

    Arguments:
        - Y_fake: Discriminator's results on the sequences resulted from the SUPERVISOR with the GENERATED sequence (H_hat)
        - Y_fake_e: Discriminator's results on the GENERATED sequences (E_hat)
        - X: The real data
        - H: The sequence EMBEDDED from the real data (X)
        - H_hat_supervise: The sequence rturned by the SUPERVISOR on the legitimate EMBEDDING (H) of the real data (X)
        - X_hat: the sequence obtained by asking the RECOVERY module to reconstruct a feature space sequence from a SUPERVISED (H_hat), GENERATED (E_hat) sequence from noise (Z)
        - gamma: weight for the loss on Y_fake_e data TODO: find a good value for it

    Returns:
        - G_loss: float with the overall loss of the GENERATOR module
        - S_loss: the supervised loss
    '''
    adversarial_loss = CrossEntropyLoss() 
    reconstruction_loss = MSELoss()

    # Adversarial truth
    valid = torch.zeros_like(Y_fake) + torch.tensor([1,0])

    # Adversarial
    GA_loss = adversarial_loss(valid, Y_fake)
    GA_loss_e = adversarial_loss(valid, Y_fake_e)

    # Supervised loss
    S_loss = reconstruction_loss(H[:,1:,:], H_hat_supervise[:,:-1,:])

    # Deviation loss
    G_loss_V1 = torch.mean(
        torch.abs(
            torch.sqrt(torch.var(X_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(X, dim=0) + 1e-6)))
    G_loss_V2 = torch.mean(
        torch.abs((torch.mean(X_hat, dim=0)) - (torch.mean(X, dim=0))))
    G_loss_V = G_loss_V1 + G_loss_V2

    # Put it back together
    G_loss = GA_loss + gamma*GA_loss_e + 100*torch.sqrt(S_loss) + 100*G_loss_V 

    return G_loss, S_loss


def reconstruction_loss(X: Tensor, X_tilde: Tensor,
                        S_loss: torch.float32, gamma=0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    This function computes the loss for the DISCRIMINATOR module.
    
    Arguments:
        - X: The original data
        - X_tilde: the data reconstructed by the RECOVERY module from the EMBEDDING (H) of the original data (X)
        - GS_loss: the supervised loss returned as the second result of generator_loss(...)

    Returns:
        - E_loss: float with the overall loss of the Discriminator module
        - R_loss: float with the reconstruction loss for the Recovery module
    '''
    loss_f = MSELoss()

    R_loss = loss_f(X, X_tilde)
    L = 10*torch.sqrt(R_loss)
    E_loss = L + gamma*S_loss

    return E_loss, R_loss

