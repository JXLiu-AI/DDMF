import os, sys
import pydicom
import pydicom.uid
import PIL
import imageio
import argparse
import cv2
import numpy as np
parser = argparse.ArgumentParser(description = 'dcm2jpg')

parser.add_argument('-in', '--input_dir', default = '', help='Input directory for dcm files')
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

outpath = args.output_path
os.makedirs(outpath, exist_ok=True)

old_dir = args.input_dir              
iterms = os.listdir(old_dir)
for iterm in iterms:
    if '.dcm' in iterm:
        newer_file1 = os.path.join(old_dir, iterm)
        basename = os.path.basename(newer_file1)
        basename_name, basename_extend = os.path.splitext(basename)
        out_path = os.path.join(outpath, basename_name + ".jpg")
        try:
            ds = pydicom.read_file(newer_file1)
            img = ds.pixel_array  
        except:
            ds = pydicom.read_file(newer_file1, force=True)
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            img = ds.pixel_array  
        #import pdb;pdb.set_trace()

        imageio.imwrite(out_path, img)
                    # except ValueError as e:
                    #     ds = pydicom.read_file(newer_file1)
                    #     #ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                    #     img = ds.pixel_array 
                    #     scipy.misc.imsave(out_path, img)
                    # except:
                    #     continue
