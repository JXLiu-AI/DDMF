from PIL import Image
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description = 'CT reconstruction')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for PNG files')
parser.add_argument('-out', '--output_dir', default = '', help='Output directory for RAW file')

args = parser.parse_args()

if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

for dirName, subdirList, fileList in os.walk(args.input_dir):
    fileList = sorted(fileList)
    image = Image.open(os.path.join(dirName, fileList[0]))
    
    for i,filename in enumerate(fileList):
        image.save(os.path.join(args.output_dir,filename[:-3]+'png'))