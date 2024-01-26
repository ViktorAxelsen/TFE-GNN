from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iter, after_scheduler):
        self.warmup_iter = warmup_iter
        self.after_scheduler = after_scheduler
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_iter:
            self.after_scheduler.step(epoch - self.warmup_iter)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_iter:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr * (float(self.last_epoch) / self.warmup_iter) for base_lr in self.base_lrs]
