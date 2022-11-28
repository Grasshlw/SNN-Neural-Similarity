import os
import torch
import numpy as np
import time
from tqdm import tqdm


class Benchmark:
    def __init__(self, neural_dataset, metric, save_path=None, suffix="", trial=1, _sample=1, _noise=0.0, _random=False):
        self.neural_dataset = neural_dataset
        self.metric = metric
        self.save_path = save_path
        self.suffix = suffix
        self.trial = trial

        self._sample = _sample
        self._noise = _noise
        self._random = _random
    
    def _print_score(self, brain_area, model_name, layer_index, score, _time, num_layers, max_len_brain_areas):
        _length = 1
        while num_layers // 10 > 0:
            _length += 1
            num_layers //= 10
        _length += len(model_name) + 1
        model_layer = f"{model_name}_{layer_index}"
        print(f"%-{max_len_brain_areas}s %-{_length}s: %.6f  time: %.4fs" % (brain_area, model_layer, score, _time))

    def _print_all_scores(self, brain_areas, model_name, num_layers, scores):
        _length = 1
        while num_layers // 10 > 0:
            _length += 1
            num_layers //= 10
        _length += len(model_name) + 1
        
        print("All scores:")

        print(f"%{_length}s" % (' '), end='')
        for brain_area in brain_areas:
            print(f" %-10s" % (brain_area), end='')
        print()

        for i in range(scores.shape[0]):
            print(f"%-{_length}s" % (f"{model_name}_{i + 1}"), end='')
            for j in range(scores.shape[1]):
                print(f" %.8f" % (scores[i, j]), end='')
            print()

    def _save_scores(self, scores, model_name):  
        os.makedirs(self.save_path, exist_ok=True)
        np.save(os.path.join(self.save_path, f"{model_name}{self.suffix}.npy"), scores)

    def __call__(self, visual_model):
        model_name = visual_model.model_name
        num_layers = len(visual_model)
        num_areas = len(self.neural_dataset)
        brain_areas = self.neural_dataset.keys()

        max_len_brain_areas = 0
        for brain_area  in brain_areas:
            if len(brain_area) > max_len_brain_areas:
                max_len_brain_areas = len(brain_area)

        scores = np.zeros((self.trial, num_layers, num_areas))
        for layer_index in range(num_layers):
            print(f"Loading activation of model layer {layer_index + 1}")
            model_data = visual_model[layer_index]
            print("Start evaluating neural similarity...")
            for area_index, brain_area in enumerate(brain_areas):
                start_time = time.time()
                
                neural_data = self.neural_dataset[area_index]
                for i in range(self.trial):
                    if self._sample > 1:
                        _index = np.random.choice(model_data[1], size=model_data[1] // self._sample, replace=False)
                        model_data_ = model_data[:, _index]
                    elif self._noise > 0:
                        noise = np.random.normal(loc=0.0, scale=self._noise, size=model_data.shape)
                        model_data_ = model_data + noise
                    elif self._random:
                        model_data_ = np.random.normal(loc=0.0, scale=1.0, size=model_data.shape)
                    else:
                        model_data_ = model_data
                    
                    scores[i, layer_index, area_index] = self.metric.score(model_data_, neural_data)
                self._print_score(brain_area, model_name, layer_index + 1, np.mean(scores[:, layer_index, area_index]), time.time() - start_time, 
                                  num_layers, max_len_brain_areas)
            print()
        
        self._print_all_scores(brain_areas, model_name, num_layers, np.mean(scores, axis=0))
        
        if self.trial == 1:
            scores = np.squeeze(scores, axis=0)
        if self.save_path is not None:
            self._save_scores(scores, model_name)


    