import torch


class PolynomialLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimiser: torch.optim.Optimizer, max_epoch: int, power: float = 1.0,
                 min_lr: float = 0):
        """Implementation of a polynomial LR decay scheduler.

        Step is taken per-epoch.

        Args:
            optimiser: The optimiser to update the learning rate for.
            max_epoch: The maximum epoch number (used to determine how much to step by).
            power: Power for polynomial decay.
            min_lr: LR to finish at.
        """
        self.t_max = max_epoch
        self.power = power
        self.min_lr = min_lr

        super().__init__(optimiser)

    def get_lr(self):
        # t starts at 0.
        t = self._step_count - 1
        assert t >= 0

        factor = (1 - t / self.t_max) ** self.power
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]
