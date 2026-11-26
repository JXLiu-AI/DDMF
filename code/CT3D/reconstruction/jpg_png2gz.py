import SimpleITK as sitk
from PIL import Image
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description = 'CT reconstruction')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

for dirName, subdirList, fileList in os.walk(args.input_dir):
    fileList = sorted(fileList)
    image = np.array(Image.open(os.path.join(dirName, fileList[0])))
    X = image.shape[0]
    Y = image.shape[1]
    images = np.zeros((len(fileList),X,Y),dtype=np.int32)

    for i,filename in enumerate(fileList):
        image = np.array(Image.open(os.path.join(dirName, filename)))
        images[i]=image

out = sitk.GetImageFromArray(images)
sitk.WriteImage(out,os.path.join(args.output_dir,'mask.nii.gz'))