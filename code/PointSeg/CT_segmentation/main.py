import os
import sys
import logging as log

# from lib.trainer import Trainer
from lib.segmentation import *
from lib.options import parse_options
from seg import seg
# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":
    args, args_str = parse_options()
    log.info(f'Parameters: \n{args_str}')
    segmentation = seg(args)
    #segmentation.export_deleted_points()
    segmentation.export_pcd(args)

