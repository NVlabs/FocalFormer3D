import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from time import perf_counter
from collections import namedtuple
from contextlib import nullcontext

global_timer = {}

class unit_timer:
    def __init__(self, skip=10):
        self.total_time = 0.
        self.count = 1e-10
        self.skip_count = skip
    
    def update_once(self, time):
        if self.skip_count <= 0:
            self.total_time += time
            self.count += 1
        else:
            self.skip_count -= 1

    @property
    def avg_time(self):
        if self.skip_count <= 0:
            return self.total_time / self.count
        else:
            return np.nan

class torch_timer(object):
    prefix = []

    def __init__(self, name='', record=False, sync=False):
        self.name = name
        self.record = record
        self.sync = sync
        if self.record:
            global_timer[name] = global_timer.get(name, unit_timer())

    @classmethod
    def update_prefix(cls, name):
        cls.prefix.append(name)
    
    @classmethod
    def get_prefix(cls):
        return '.'.join(cls.prefix)
    
    @classmethod
    def remove_prefix(cls):
        cls.prefix.pop(-1)

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start_time = perf_counter()
        self.update_prefix(self.name)

    def __exit__(self, type=None, value=None, traceback=None):
        if self.sync:
            torch.cuda.synchronize()
        once_time = perf_counter() - self.start_time
        time_str = f'--- {self.get_prefix()}: {once_time:.4f}s'

        if self.record:
            global_timer[self.name].update_once(once_time)
            time_str += f' ({global_timer[self.name].avg_time:.4f})'

        print(time_str)
        self.remove_prefix()

def T(name='', record=False, enable=False, sync=False):
    if enable:
        return torch_timer(name=name, record=record, sync=sync)
    else:
        return nullcontext()
