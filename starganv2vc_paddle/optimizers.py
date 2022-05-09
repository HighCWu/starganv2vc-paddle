#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import paddle
from paddle import nn
from paddle.optimizer import Optimizer
from functools import reduce
from paddle.optimizer import AdamW

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        
    def get_lr(self):
        return max([self.optimizers[key].get_lr() 
                    for key in self.keys])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def set_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].set_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def clear_grad(self, key=None):
        if key is not None:
            self.optimizers[key].clear_grad()
        else:
            _ = [self.optimizers[key].clear_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step(*args) for key in self.keys]

def define_scheduler(params):
    print(params)
    # scheduler = paddle.optim.lr_scheduler.OneCycleLR(
    #     max_lr=params.get('max_lr', 2e-4),
    #     epochs=params.get('epochs', 200),
    #     steps_per_epoch=params.get('steps_per_epoch', 1000),
    #     pct_start=params.get('pct_start', 0.0),
    #     div_factor=1,
    #     final_div_factor=1)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=params.get('max_lr', 2e-4),
        T_max=10)

    return scheduler

def build_optimizer(parameters_dict, scheduler_params_dict):
    schedulers = dict([(key, define_scheduler(params)) \
                       for key, params in scheduler_params_dict.items()])

    optim = dict([(key, AdamW(parameters=parameters_dict[key], learning_rate=sch, weight_decay=1e-4, beta1=0.1, beta2=0.99, epsilon=1e-9))
                   for key, sch in schedulers.items()])


    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim