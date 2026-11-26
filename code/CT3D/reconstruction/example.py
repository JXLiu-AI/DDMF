import numpy as np
import os
from tqdm import tqdm
import argparse
from PIL import Image
import mcubes


images = []


parser = argparse.ArgumentParser(description = 'CT reconstruction')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')

args = parser.parse_args()


for dirName, subdirList, fileList in os.walk(args.input_dir):
    fileList = sorted(fileList)

    image = np.array(Image.open(os.path.join(dirName, fileList[0])))
    X = image.shape[0]
    Y = image.shape[1]
    images = np.zeros((len(fileList),X,Y),dtype=np.int32)
    for i,filename in enumerate(fileList):
        image = np.array(Image.open(os.path.join(dirName, filename)))
        # image[image<10] = 0
        # image[image>10] = 255
        images[i]=image

# The 255 is the isosurface, but I believe there is a bug.
vertices, triangles = mcubes.marching_cubes(images, 255)
mcubes.export_mesh(vertices, triangles, "sphere.dae", "MySphere")
# images.tofile(os.path.join(args.output_dir,'output.raw'))

# pcd = o3d.geometry.PointCloud(xyz)
# pcd.points = o3d.utility.Vector3dVector(xyz)
# o3d.visualization.draw_geometries([])
