""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/batch_load_scannet_data.py

Batch mode in loading Scannet scenes with vertices and ground truth labels for semantic and instance segmentations

Usage example: python ./batch_load_scannet_data.py
"""

import os
import sys
import datetime
import numpy as np
from data.urbanbis.load_urbanbis_data import export
import pdb

SCANNET_DIR = 'scans'
SCAN_NAMES = sorted([line.rstrip() for line in open('/workspace/UrbanQA/data/urbanbis/meta_data/city_area.txt')])
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = '/workspace/UrbanQA/data/urbanbis/urbanbis_data/'

def batch_export():
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
    main_path = "/workspace/Datasets/UrbanBIS/UrbanBIS/CityQA"

    for scan_name in SCAN_NAMES:
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
        city_name = scan_name.split("_")[0]
        area_number = "_".join(scan_name.split("_")[1:])
              
        export(main_path, city_name, area_number, output_filename_prefix)
             
        print('-'*20+'done')

if __name__=='__main__':    
    batch_export()
