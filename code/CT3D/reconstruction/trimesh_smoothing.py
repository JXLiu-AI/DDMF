import trimesh
import argparse
import numpy as np
parser = argparse.ArgumentParser(description = 'Trimesh smoothing')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')
parser.add_argument('-m', '--m', type=str, default = '', help='')
# iteration for laplacian smoothing
# iter = 6
args = parser.parse_args()
# import pdb;pdb.set_trace()
temp = args.m
strlist = temp.split(',')

a = np.array([[float(strlist[0]),0,0,0],
              [0,float(strlist[1]),0,0],
              [0,0,float(strlist[2]),0],
              [0,0,0,1]])


mesh = trimesh.load(args.input_dir)
mesh = mesh.copy().apply_transform(a)
# mesh2 = trimesh.smoothing.filter_laplacian(mesh,iterations=iter)
mesh.export(args.output_dir)
