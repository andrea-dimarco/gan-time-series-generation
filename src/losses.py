import torch


# Modules
from modules.generator import Generator
from modules.embedder import Embedder
from modules.discriminator import Discriminator
from modules.recovery import Recovery
from modules.supervisor import Supervisor




def D_loss(X: torch.Tensor, Z: torch.Tensor,
           Emb:Embedder, Gen:Generator,
           Sup:Supervisor, Dis:Discriminator, discrimination_loss=torch.nn.BCELoss(),
           w1:float=0.40, w2:float=0.40, w3:float=0.20
) -> torch.Tensor:
    '''
    This function computes the loss for the DISCRIMINATOR module.

    Arguments:
        - `X`: batch of real sequences
        - `Z`: batch of random noise sequences

    Returns:
        - `D_loss`: tensor with one element containing the Discriminator module's loss
    '''
    # Compute model outputs
        # 1. Embedder
    H = Emb(X)
        # 2. Generator
    E_hat = Gen(Z) 
        # 3. Supervisor
    H_hat = Sup(E_hat)
        # 4. Discriminator
    Y_fake   = Dis(H_hat)
    Y_real   = Dis(H)
    Y_fake_e = Dis(E_hat)


    # Adversarial truths
    valid = torch.ones_like(Y_real)
    fake  = torch.zeros_like(Y_fake)


    # Loss Components
    loss_real   = discrimination_loss(Y_real,   valid)
    loss_fake   = discrimination_loss(Y_fake,   fake)
    loss_fake_e = discrimination_loss(Y_fake_e, fake)

    return w1*loss_real + w2*loss_fake + w3*loss_fake_e



def G_loss(X: torch.Tensor, Z: torch.Tensor,
           Gen:Generator, Sup:Supervisor,
           Rec:Recovery, Dis:Discriminator, 
           discrimination_loss=torch.nn.BCELoss(), reconstruction_loss=torch.nn.MSELoss(),
           w1:float=0.10, w2:float=0.35, w3:float=0.10, w4:float=0.45
) -> torch.Tensor:
    '''
    This function computes the loss for the GENERATOR module.

    Arguments:
        - `X`: batch of real sequences
        - `Z`: batch of random noise sequences

    Returns:
        - `G_loss`: tensor with one element containing the Generator module's loss
    '''
    # Compute model outputs
        # 1. Generator
    E_hat = Gen(Z) 
        # 2. Supervisor
    H_hat = Sup(E_hat)
        # 3. Recovery
    X_hat = Rec(E_hat)
        # 4. Discriminator
    Y_fake   = Dis(H_hat)
    Y_fake_e = Dis(E_hat)


    # Loss components
        # 1. Adversarial truth
    valid = torch.ones_like(Y_fake)
        # 2. Adversarial loss
    GA_loss   = discrimination_loss(Y_fake,   valid)
    GA_loss_e = discrimination_loss(Y_fake_e, valid)
        # 3. Supervised loss
    S_loss    = reconstruction_loss(H_hat[:,1:,:], E_hat[:,:-1,:])
        # 4. Deviation loss
    G_loss_mu = torch.mean(
        torch.abs((torch.mean(X_hat, dim=0)) - (torch.mean(X, dim=0))))
    G_loss_std = torch.mean(
        torch.abs(
            torch.sqrt(torch.var(X_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(X, dim=0) + 1e-6)))
    G_loss_V  = G_loss_mu + G_loss_std

    return w1*GA_loss + w2*GA_loss_e + w3*S_loss*0.0 + w4*G_loss_V 



def S_loss(X: torch.Tensor, Z: torch.Tensor,
           Emb:Embedder, Gen:Generator,
           Sup:Supervisor, reconstruction_loss=torch.nn.MSELoss(),
           w1:float=0.4, w2:float=0.6, scaling_factor=1000
) -> torch.Tensor:
    '''
    This function computes the loss for the SUPERVISOR module.

    Arguments:
        - `X`: batch of real sequences
        - `Z`: batch of random noise sequences

    Returns:
        - `S_loss`: tensor with one element containing the Supervisor module's loss
    '''
    # Compute model outputs
        # 1. Embedder
    H = Emb(X)
        # 2. Generator
    E_hat = Gen(Z) 
        # 3. Supervisor
    H_hat = Sup(E_hat)
    H_hat_supervise = Sup(H)

    # Loss components
        # 1. Reconstruction Loss
    Rec_loss = reconstruction_loss(H, H_hat_supervise)
    #Rec_loss = reconstruction_loss(H, H_hat_supervise)
        # 2. Deviation Loss
    Dev_loss_mu = torch.mean(
        torch.abs(
            torch.sqrt(torch.var(H_hat, dim=0) + 1e-6) - torch.sqrt(torch.var(H_hat_supervise, dim=0) + 1e-6)))
    Dev_loss_std = torch.mean(
        torch.abs((torch.mean(H_hat, dim=0)) - (torch.mean(H_hat_supervise, dim=0))))
    Dev_loss = Dev_loss_mu + Dev_loss_std

    # Supervised loss
    return (w1*Rec_loss + w2*Dev_loss)*scaling_factor



def E_loss(X: torch.Tensor,
           Emb:Embedder, Sup:Supervisor,
           Rec:Recovery,
           reconstruction_loss=torch.nn.MSELoss(),
           w1: float=0.5, w2:float=0.5
) -> torch.Tensor:
    '''
    This function computes the loss for the EMBEDDER module.

    Arguments:
        - `X`: batch of real sequences
        - `Z`: batch of random noise sequences

    Returns:
        - `E_loss`: tensor with one element containing the Embedder module's loss
    '''
    # Compute model outputs
        # 1. Embedder
    H = Emb(X)
        # 2. Supervisor
    H_hat_supervise = Sup(H)
        # 3. Recovery
    X_tilde = Rec(H)

    # Loss Components
    R_loss = reconstruction_loss(X, X_tilde)
    S_loss = reconstruction_loss(H, H_hat_supervise)

    return w1*R_loss + w2*S_loss



def R_loss(X: torch.Tensor, 
           Emb:Embedder, Rec:Recovery,
           reconstruction_loss=torch.nn.MSELoss(),
           scaling_factor=10
) -> torch.Tensor:
    '''
    This function computes the loss for the RECOVERY module.

    Arguments:
        - `X`: batch of real sequencess

    Returns:
        - `R_loss`: tensor with one element containing the Recovery module's loss
    '''
    # Compute model outputs
        # 1. Embedder
    H = Emb(X)
        # 2. Recovery
    X_tilde = Rec(H)

    return reconstruction_loss(X, X_tilde)*scaling_factor