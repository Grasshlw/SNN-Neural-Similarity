import os
import json
import torch

from dataset import NeuralDataset
from visualmodel import VisualModel
from metric import CCAMetric, RSAMetric, CKAMetric, RegMetric
from benchmark import Benchmark

from model.Net import *


def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allen_natural_scenes", type=str, choices=["allen_natural_scenes", "macaque_synthetic", "macaque_face"], help="dataset name")
    parser.add_argument("--dataset-path", default="neural_data/", type=str, help="dataset path")
    parser.add_argument("--stimulus-path", default="stimulus/allen_natural_scenes_224.pt", type=str, help="stimulus path")
    
    parser.add_argument("--metric", default="RSA", type=str, choices=["CCA", "RSA", "Regression", "CKA"], help="neural similarity metric")
    parser.add_argument("--reduction", default="TSVD", type=str, choices=["TSVD", "PCA"], help="method of dimension reduction")
    parser.add_argument("--dim", default=40, type=int, help="dimension after dimension reduction")
    parser.add_argument("--neural-reduction", action="store_true", help="whether to reduce the dimension of neural data")
    parser.add_argument("--k", default=-1, type=int, help="k-fold cross-validation ('-1' indicates leave-one-out cross-validation)")
    parser.add_argument("--kernel", default="linear", type=str, choices=["linear", "rbf"], help="kernel function of CKA")
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    
    parser.add_argument("--model-name", default="sew_resnet18", type=str, help="name of model")
    parser.add_argument("--model-checkpoint", default="../model_checkpoint/sew_resnet18.pth", type=str, help="model checkpoint path")
    parser.add_argument("--device", default="cuda:0", type=str, help="torch device")
    parser.add_argument('--T', default=4, type=int, help="total time-steps")

    parser.add_argument("--output-path", default="results/", help="path to save outputs")

    args = parser.parse_args()
    return args


def dataset_areas(dataset):
    brain_areas_dict = {'allen_natural_scenes': ['visp', 'visl', 'visrl', 'visal', 'vispm', 'visam'], 
                        'macaque_synthetic': ['V4', 'IT'], 
                        'macaque_face': ['AM']}
    return brain_areas_dict[dataset]


def metric_preset(args):
    if args.metric == "CCA":
        return CCAMetric(reduction=args.reduction, dims=args.dim, neural_reduction=args.neural_reduction, seed=args.seed)
    elif args.metric == "RSA":
        return RSAMetric()
    elif args.metric == "Regression":
        return RegMetric(reduction=args.reduction, dims=args.dim, splits=args.k, seed=args.seed)
    elif args.metric == "CKA":
        return CKAMetric(kernel=args.kernel)


def save_path_preset(args):
    save_path = os.path.join(args.output_path, args.metric.lower(), args.dataset)
    
    suffix_dict = {'CCA': f"_{args.reduction}_{args.dim}",
                   'RSA': "",
                   'Regression': f"_{args.reduction}_{args.dim}",
                   'CKA': f"_{args.kernel}"}
    suffix = suffix_dict[args.metric]
    if args.metric == 'Regression' and args.k != -1:
        suffix += f"_{args.k}"
    
    return save_path, suffix


def main(args):
    print(args)
    print("Loading neural data")
    neural_dataset = NeuralDataset(args.dataset, dataset_areas(args.dataset),
                                   data_path=args.dataset_path)
    print("Loading model")
    model = eval(args.model_name)()
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    model = preprocess_model(model)
    device = torch.device(args.device)
    visual_model = VisualModel(model, args.model_name, get_layers(), args.stimulus_path, _time_step=args.T, device=device)

    metric = metric_preset(args)
    save_path, suffix = save_path_preset(args)
    benchmark = Benchmark(neural_dataset, metric, save_path=save_path, suffix=suffix)
    benchmark(visual_model)


if __name__=="__main__":
    args = get_args()
    main(args)
