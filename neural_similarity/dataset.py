import os
import numpy as np


class NeuralDataset:
    def __init__(self, dataset_name, brain_areas, data_path, **kwargs):
        self.dataset_name = dataset_name
        if isinstance(brain_areas, int):
            self.brain_areas = [brain_areas]
        else:
            self.brain_areas = brain_areas
        self.data_path = data_path
        
        self.neural_dataset = {}
        for i in range(len(self.brain_areas)):
            self.neural_dataset[self.brain_areas[i]] = eval(f"self.{dataset_name}")(self.brain_areas[i], **kwargs)
    
    def allen_natural_scenes(self, brain_area, time_step=8, exclude=True, threshold=0.8, _is_time_step=False):
        neural_data = np.load(os.path.join(self.data_path, self.dataset_name, f"{brain_area}_{time_step}.npy"))

        if exclude:
            shr = np.load(os.path.join(self.data_path, self.dataset_name, f"shr_{brain_area}.npy"))
            neural_data = neural_data[:, :, shr >= threshold]

        neural_data = neural_data / 50
        if not _is_time_step:
            neural_data = np.sum(neural_data, axis=1)
            neural_data /= 25

        return neural_data

    def macaque_face(self, brain_area, exclude=True, threshold=0.1, time_step=None, _is_time_step=False):
        neural_data = np.load(os.path.join(self.data_path, self.dataset_name, f"{brain_area}.npy"))

        if exclude:
            noise_ceiling = np.load(os.path.join(self.data_path, self.dataset_name, "noise_ceiling.npy"))
            neural_data = neural_data[:, noise_ceiling >= threshold]
        
        return neural_data

    def macaque_synthetic(self, brain_area, time_step=None, exclude=None, threshold=None, _is_time_step=False):
        return np.load(os.path.join(self.data_path, self.dataset_name, f"{brain_area}.npy"))

    def __len__(self):
        return len(self.brain_areas)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.neural_dataset[self.brain_areas[key]]
        elif isinstance(key, str):
            return self.neural_dataset[key]
        else:
            raise KeyError(f"'{key}' doesn't exist")
    
    def keys(self):
        return self.brain_areas

    def size(self, key):
        if isinstance(key, int):
            print(f"Brain Area: {self.brain_areas[key]}")
            print(f"Number of Stimulus: {self.neural_dataset[self.brain_areas[key]].shape[0]}  Number of Neuron: {self.neural_dataset[self.brain_areas[key]].shape[1]}")
        elif isinstance(key, str):
            print(f"Brain Area: {key}")
            print(f"Number of Stimulus: {self.neural_dataset[key].shape[0]}  Number of Neuron: {self.neural_dataset[key].shape[1]}")
        else:
            raise KeyError(f"'{key}' doesn't exist")