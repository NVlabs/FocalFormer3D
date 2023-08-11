import enum
import torch
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class Fading(Hook):
    def __init__(self, fade_epoch = 100000):
        self.fade_epoch = fade_epoch

    def before_train_epoch(self, runner):
        if runner.epoch >= self.fade_epoch:
            for i, transform in enumerate(runner.data_loader.dataset.dataset.pipeline.transforms):
                if type(transform).__name__ == 'ObjectSample':
                    runner.data_loader.dataset.dataset.pipeline.transforms.pop(i)
                    break