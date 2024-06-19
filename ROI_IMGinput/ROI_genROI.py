## Import Libraries #
import sys
import pickle
import os
import shutil
import argparse
import time
from tqdm import tqdm

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms 
from torchvision.transforms import ToTensor
import torchvision.models as models 
import torch.nn.functional as F
from torchvision.utils import make_grid

###########################################
sys.path.append('/home/wtepsan/TOOLS/')  

## Import feeder #      
import feeder.segment_rgbbody_ntu_UPDATE as segment_rgbbody_ntu

## Import Utils #          
import Utils.ROI_dataset as ROI_dataset 
import Utils.Loss as Loss
import Utils.ROI_tools as ROI_tools
import Utils.RGBModels as RGBModels

openposekeys = {
    "nose":0,
    "neck":1,
    "Rshoulder":2,
    "Rellbow":3,
    "Rwrist":4,
    "Lshoulder":5,
    "Lelbow":6,
    "Lwrist":7,
    "Rhip":8,
    "Rknee":9,
    "Rankle":10,
    "Lhip":11,
    "Lknee":12,
    "Lankle":13,
    "Reye":14,
    "Leye":15,
    "Rear":16,
    "Lear":17,
}

if __name__ == '__main__':
        # print("--- CODE TESTING ---") #
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin')
    parser.add_argument('--end')

    arg = parser.parse_args()
    begin = int(arg.begin)
    end = int(arg.end)

    openpose_path = "/project/lt200048-video/Dataset/NTU-RGBD/Openpose/openpose_ntu60_unzip/"

    PATH_FRAMEIMAGES = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"
    PATH_OPTICALFLOWIMAGE = "/project/lt200048-video/NTU_Optical/"

    PATH_SAVE = "/home/wtepsan/ROI_IMGinput-Experiments/_TESTS_/"

    ########################################
    EXPERIMENT_NAME = "ALL_IMAGES_MIXX"
    ########################################

    PATH_SAVE_EPERIMENT_NAME = PATH_SAVE+EXPERIMENT_NAME
    if not os.path.exists(PATH_SAVE_EPERIMENT_NAME):
        os.makedirs(PATH_SAVE_EPERIMENT_NAME) 

    datanames = os.listdir(PATH_FRAMEIMAGES) 
    datanames.sort()
    datanames = datanames[begin:end]

    print(f"Gen ROI begin from {begin} to {end}")
    print(f"Number of Gen ROI images is {len(datanames)}") 
    

    # joints_chosen =["nose", "Rwrist", "Lwrist", "Rhip", "Lhip","Rankle", "Lankle"]
    joints_chosen = ["nose", "Rwrist", "Lwrist", "Rankle", "Lankle"]
    joints = [openposekeys[key] for key in joints_chosen if key in openposekeys]
    # joints=[0, 4, 7, 10, 13]

    filenames = [
        "S007C001P015R001A020.json", "S011C003P002R002A057.json", "S017C001P016R002A013.json",
        "S007C001P015R001A021.json", "S011C003P002R002A058.json", "S017C001P016R002A014.json",
        "S007C001P015R001A022.json", "S011C003P002R002A059.json", "S017C001P016R002A015.json",
        "S007C001P015R001A023.json", "S011C003P002R002A060.json", "S017C001P016R002A016.json",
        "S007C001P015R001A024.json", "S011C003P007R001A001.json", "S017C001P016R002A017.json",
        "S007C001P015R001A025.json", "S011C003P007R001A002.json", "S017C001P016R002A018.json",
        "S007C001P015R001A026.json", "S011C003P007R001A003.json", "S017C001P016R002A019.json",
        "S007C001P015R001A027.json", "S011C003P007R001A004.json", "S017C001P016R002A020.json",
        "S007C001P015R001A028.json", "S011C003P007R001A005.json", "S017C001P016R002A021.json",
        "S007C001P015R001A029.json", "S011C003P007R001A006.json", "S017C001P016R002A022.json",
        "S007C001P015R001A030.json", "S011C003P007R001A007.json", "S017C001P020R001A027.json"
    ]

    # Remove ".json" extension and make them a list of strings
    datanames = [filename[:-5] for filename in filenames]

    for dataname in tqdm(datanames):

        # img0 =  segment_rgbbody_ntu.constructfullbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=280)
        # img1 = segment_rgbbody_ntu.construct2partbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=140)
        # img2 =  segment_rgbbody_ntu.construct3partbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=96)
        # img3 = segment_rgbbody_ntu.construct_st_roi_3Joints(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=48, h=48)
        # img4 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[4, 7, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)
        img5 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[0, 4, 7, 10, 13], random_interval=False, choosing_frame_method = 'mix', w=48, h=48)
        # img6 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[4, 7, 8, 11, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)
        # img7 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[0, 4, 7, 8, 11, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)

        try:
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_fullbody.jpg"
            # img0.save(saveroipath)  
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_2partbody.jpg"
            # img1.save(saveroipath)  
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_3partbody.jpg"
            # img2.save(saveroipath)  
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_3joints.jpg"
            # img3.save(saveroipath)
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_4joints.jpg"
            # img4.save(saveroipath)   
            saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_5joints.jpg"
            img5.save(saveroipath)    
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_6joints.jpg"
            # img6.save(saveroipath)
            # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_7joints.jpg"
            # img7.save(saveroipath)
        except:
            print(f"Something Wrong with {dataname}")

    # for dataname in tqdm(datanames):
    #     frame_file = PATH_FRAMEIMAGES + dataname + "/1.jpg"
    #     frame = Image.open(frame_file)
    #     filename = dataname
    #     setup_id = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    #     camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    #     subject_id = int( filename[filename.find('P') + 1:filename.find('P') + 4])
    #     duplicate_id = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    #     action_id = int(filename[filename.find('A') + 1:filename.find('A') + 4])

    #     skeleton_file_name = segment_rgbbody_ntu.filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)

    #     x = 960
    #     y = 540
    #     # img = segment_rgbbody_ntu.cropFramefromXY(frame, x, y, w=100, h=100) ## -> 200 x 200 
    #     openpose_file_, frame_file_ = segment_rgbbody_ntu.openposeFile(frame_file, 1, skeleton_file_name, openpose_path) 

    #     # img1 = segment_rgbbody_ntu.construct3partbodyfromvideo(openpose_file_, frame_file, action_id, w=100, h=100)
    #     # img2 = segment_rgbbody_ntu.construct3partbodyfromvideo(openpose_file_, frame_file, action_id, w=100, h=100)
    #     # img1 =  segment_rgbbody_ntu.constructfullbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, w=48, h=48*3)
    #     # img2 =  segment_rgbbody_ntu.construct3partbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, w=48, h=48)

    #     # img1 = segment_rgbbody_ntu.crop2bodypart(openpose_file_, frame_file, action_id, w=90, h=140)
    #     img1 = segment_rgbbody_ntu.construct2partbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=140)
    #     try:
    #         saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_2PARTS5FRAMES.jpg"
    #         img1.save(saveroipath)  
    #     except:
    #         print(f"Something Wrong with {dataname}")
        
    ### FOR ROI N JOINTS ###
    # for dataname in tqdm(datanames):

    #     frame_file = PATH_FRAMEIMAGES + dataname + "/1.jpg"
    #     frame = Image.open(frame_file)
    #     filename = dataname
    #     setup_id = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    #     camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    #     subject_id = int( filename[filename.find('P') + 1:filename.find('P') + 4])
    #     duplicate_id = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    #     action_id = int(filename[filename.find('A') + 1:filename.find('A') + 4])

    #     skeleton_file_name = segment_rgbbody_ntu.filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)

    #     x = 960
    #     y = 540
    #     # img = segment_rgbbody_ntu.cropFramefromXY(frame, x, y, w=100, h=100) ## -> 200 x 200 
    #     openpose_file_, frame_file_ = segment_rgbbody_ntu.openposeFile(frame_file, 1, skeleton_file_name, openpose_path) 

    #     # img1 = segment_rgbbody_ntu.construct3partbodyfromvideo(openpose_file_, frame_file, action_id, w=100, h=100)
    #     # img2 = segment_rgbbody_ntu.construct3partbodyfromvideo(openpose_file_, frame_file, action_id, w=100, h=100)
    #     img1 =  segment_rgbbody_ntu.constructfullbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, w=90, h=280)
    #     img2 =  segment_rgbbody_ntu.construct3partbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, w=90, h=90)
 
    #     try:
    #         saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_FullBody.jpg"
    #         img1.save(saveroipath)
    #         saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_3PartBody.jpg"
    #         img2.save(saveroipath)
    #     except:
    #         print(f"Something Wrong with {dataname}")

        #########
        # choosing_frame_method = 'fix'
        # img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=joints, random_interval=True, choosing_frame_method=choosing_frame_method, w=48, h=48)
        # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_" + choosing_frame_method + ".jpg"
        # img.save(saveroipath)

        # choosing_frame_method = "randombegin_randomlength"
        # img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=joints, random_interval=True, choosing_frame_method=choosing_frame_method, w=48, h=48)
        # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_" + choosing_frame_method + ".jpg"
        # img.save(saveroipath)

        # choosing_frame_method = "random_from_each_interval"
        # img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=joints, random_interval=True, choosing_frame_method=choosing_frame_method, w=48, h=48)
        # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_" + choosing_frame_method + ".jpg"
        # img.save(saveroipath)

        # choosing_frame_method = "random"
        # img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=joints, random_interval=True, choosing_frame_method=choosing_frame_method, w=48, h=48)
        # saveroipath = PATH_SAVE_EPERIMENT_NAME + "/" + dataname + "_" + choosing_frame_method + ".jpg"
        # img.save(saveroipath)
