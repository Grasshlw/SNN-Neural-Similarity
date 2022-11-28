import os
import json
from tqdm import tqdm
import torch
import numpy as np
from collections import OrderedDict

from spikingjelly.activation_based import functional

from model.Net import *


__all__ = ["Activation", "SNNActivation"]


class Activation:
    def __init__(self, model, stimulus_path, device="cuda:0"):
        self.model = model.to(device)
        self.set_stimulus(stimulus_path)
        self.device = device
    
        self.features = None
        self.batch_size = 1

    def set_stimulus(self, stimulus_path):
        self.stimulus = torch.load(stimulus_path)

    def hook_fn(self, module, inputs, outputs):
        pass

    def layer_activation(self, layer_name):
        pass


class SNNActivation(Activation):
    def __init__(self, model, stimulus_path, device="cuda:0", _mean=False):
        super().__init__(model, stimulus_path, device)
        self._mean = _mean
    
    def hook_fn(self, module, inputs, outputs):
        self.features.append(outputs.data.cpu())

    def layer_activation(self, layer_name):
        self.model.eval()
        
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            inputs = torch.zeros((1, 3, 224, 224)).to(self.device)
            inputs = preprocess_input(inputs)
            self.features = []
            self.model(inputs)
            if len(self.features) == 1:
                layer_dim = list(self.features[0].size())
            else:
                layer_dim = [len(self.features)] + list(self.features[0].size())
            hook.remove()
        layer_dim.pop(1)

        stimulus_dataset = torch.utils.data.TensorDataset(self.stimulus)
        n_stimulus = len(stimulus_dataset)
        stimulus_dataloader = torch.utils.data.DataLoader(stimulus_dataset, batch_size=self.batch_size)

        if self._mean:
            activation = torch.empty([n_stimulus] + layer_dim[1:], dtype=torch.float)
        else:
            activation = torch.empty([n_stimulus] + layer_dim, dtype=torch.float)
        
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                inputs = preprocess_input(inputs)

                self.features = []
                self.model(inputs)
                if len(self.features) == 1:
                    features = self.features[0]
                else:
                    features = torch.empty([layer_dim[0], bs] + layer_dim[1:], dtype=torch.float)
                    for i in range(layer_dim[0]):
                        features[i] = self.features[i]
                if self._mean:
                    activation[n: n + bs] = features.mean(dim=0)
                else:
                    activation[n: n + bs] = features.transpose(0, 1)
                functional.reset_net(self.model)
                n += bs
            hook.remove()

        return activation
