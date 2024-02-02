class LambdaLR:
    def __init__(self, n_epochs: int, decay_start_epoch: int) -> None:
        '''
        Linearly decay the leraning rate to 0, starting from `decay_start_epoch`
        to the final epoch.

        Arguments:
            - n_epochs: total number of epochs
            - decay_start_epoch: epoch in which the learning rate starts to decay
        '''
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        '''
        One step of lr decay:
        - if `epoch < self.decay_start_epoch` it doesn't change the learning rate.
        - Otherwise, it linearly decay the lr to reach zero

        Arguments:
            - epoch: current epoch
        
        Returns:
            - Learning rate multiplicative factor
        '''
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )