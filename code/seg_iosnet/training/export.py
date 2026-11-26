import argparse
import torch
from pl_model import SegDGCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Export a Tooth Seg Model")
    parser.add_argument('-i', '--input-model', type=str, required=True, help='Input model')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Output model path')
    parser.add_argument('--num-points', type=int, default=10000, help='Point Number [default: 10000]')
    parser.add_argument('-b', '--batch-size', type=int, default=3, help='Batch Size of ONNX [default: 3]')
    parser.add_argument('--num-features', type=int, default=15, help='Number of features per point')
    parser.add_argument('--num-categories', type=int, default=2, help='Number of categories')
    parser.add_argument('--dynamic', action='store_true', help='Make batch dim dynamic')

    args = parser.parse_args()
    return args

args = parse_args()


M = SegDGCNN.load_from_checkpoint(args.input_model)

dynamic_axes = {
    'points' : {0 : 'batch_size'},
    'cat' : {0 : 'batch_size'},
    'seg_probs' : {0 : 'batch_size'},
}

input_sample = [torch.randn((args.batch_size, args.num_features, args.num_points)),
                torch.rand((args.batch_size, args.num_categories))] # Fix BS=3
M.to_onnx(args.output_path, input_sample, export_params=True,
          input_names=['points', 'cat'],
          output_names=['seg_probs'],
          dynamic_axes=dynamic_axes if args.dynamic else {})
