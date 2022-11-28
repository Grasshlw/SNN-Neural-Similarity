import os
import torch
import numpy as np

from activation import SNNActivation


class VisualModel:
    def __init__(self, model, model_name, layers_name, stimulus_path, _time_step=0, _is_time_step=False, _n_dims=2, _normalize=True, device="cpu"):
        self.model_name = model_name
        self.layers_name = layers_name

        self._time_step = _time_step
        self._is_time_step = _is_time_step
        self._n_dims = _n_dims
        self._normalize = _normalize

        self.activation = SNNActivation(model, stimulus_path, device)
    
    def _z_score(self, x):
        mean_ = np.mean(x)
        std_ = np.std(x)
        x = (x - mean_) / (std_ + 1e-10)
        
        return x

    def _process_model_data(self, x):
        x = x.numpy()
        if self._time_step > 0:
            if not self._is_time_step:
                x = np.mean(x, axis=1)
                x = x.reshape((x.shape[0], -1))
            else:
                x = x.reshape((x.shape[0], x.shape[1], -1))
                x = x.transpose(0, 2, 1)
                if self._n_dims == 2:
                    x = x.reshape((x.shape[0], -1))
        else:
            x = x.reshape((x.shape[0], -1))
        if self._normalize:
            x = self._z_score(x)
        return x

    def __len__(self):
        return len(self.layers_name)
    
    def __getitem__(self, key):
        model_data = self.activation.layer_activation(self.layers_name[key])
        model_data = self._process_model_data(model_data)
        return model_data

