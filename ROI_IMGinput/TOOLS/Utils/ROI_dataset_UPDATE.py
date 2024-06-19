## Define Dataset Class ##

# Importing Libraries and Module
import os
import json
import argparse
import random
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import Utils.ROI_tools as ROI_tools
# import ROI_tools

import sys        
sys.path.append('/home/wtepsan/TOOLS/')  

## Import feeder #      
import feeder.segment_rgbbody_ntu_UPDATE as segment_rgbbody_ntu

## Import Utils #          
import Utils.ROI_dataset_UPDATE as ROI_dataset 
import Utils.Loss as Loss
import Utils.ROI_tools as ROI_tools
import Utils.RGBModels as RGBModels


# sys.path.append('/home/wtepsan/ROI_Attentions/Attentions/')       
# import framesXopticalflow


############################################################################################################
### Set Global Variables ### 
############################################################################################################

YOLO_ROI = "/project/lt200048-video/DatasetGen_NTU60/ROI_from_YOLOskelton2_1person_clean2/"
YOLO_SKE = "/project/lt200048-video/DatasetGen_NTU60/YOLOskeletons2_clean/"
RGB_PATH = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"

# XSUB TRAIN 40320: TEST 16560:: 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
XSUB_TRAIN = ['P001', 'P002', 'P004', 'P005', 'P008', 'P009', 'P013', 'P014', 'P015',  'P016', 'P017', 'P018', 'P019', 'P025', 'P027', 'P028', 'P031', 'P034', 'P035', 'P038'] 

# XVIEW TRAIN 37920: TEST 18960:: 2, 3
XVIEW_TRAIN = ['C002', 'C003']

############################################################################################################
### SPLIT TRAIN/TEST ### 
############################################################################################################

def datanames_train_test(benchmark='xsub', datapath = RGB_PATH):
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []  
    alldatanameslist = os.listdir(datapath)
    datanameslist = []
    for dataname in alldatanameslist:
        datanamefolder = datapath+"/"+dataname
        fileindatanamefolderlist = os.listdir(datanamefolder)
        if len(fileindatanamefolderlist) > 16: 
            datanameslist.append(dataname)

    print(f"ROI Image path:{datapath} \nNumber of data set {len(datanameslist)}")

    if benchmark == 'xsub':
        for dataname in datanameslist:
            if dataname[dataname.find('P'): dataname.find('P')+4] in XSUB_TRAIN:
                Xtrain.append(dataname)
                ytrain.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
            else:
                Xtest.append(dataname)
                ytest.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
    elif benchmark == 'xview':
         for dataname in datanameslist:
            if dataname[dataname.find('C'): dataname.find('C')+4] in XVIEW_TRAIN:
                Xtrain.append(dataname)
                ytrain.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
            else:
                Xtest.append(dataname)
                ytest.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
    else: 
        print("No such benchmark")
    # List of datanames with thier labels 
    return Xtrain, ytrain, Xtest, ytest


def datanames_train_test_from_roifolder(benchmark='xsub', datapath = RGB_PATH):
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []  
    alldatanameslist = os.listdir(datapath)
    datanameslist = []
    for dataname in alldatanameslist:
        # datanamefolder = datapath+"/"+dataname
        #fileindatanamefolderlist = os.listdir(datanamefolder)
        # if len(fileindatanamefolderlist) > 16: 
        datanameslist.append(dataname.split(".")[0])

    print(f"ROI Image path:{datapath} \nNumber of data set {len(datanameslist)}")

    if benchmark == 'xsub':
        for dataname in datanameslist:
            if dataname[dataname.find('P'): dataname.find('P')+4] in XSUB_TRAIN:
                Xtrain.append(dataname)
                ytrain.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
            else:
                Xtest.append(dataname)
                ytest.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
    elif benchmark == 'xview':
         for dataname in datanameslist:
            if dataname[dataname.find('C'): dataname.find('C')+4] in XVIEW_TRAIN:
                Xtrain.append(dataname)
                ytrain.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
            else:
                Xtest.append(dataname)
                ytest.append(int(dataname[dataname.find('A')+1: dataname.find('A')+4])-1)
    else: 
        print("No such benchmark")
    # List of datanames with thier labels 
    return Xtrain, ytrain, Xtest, ytest
############################################################################################################
### Data Class ### 
############################################################################################################

class ROI_dataset_on_generated_roi(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None): 
    # INPUT: Image Path, Label, transform
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):
        dataname =  self.datanames[idx]

        label = torch.tensor(self.labels[idx])
        
        datanameimgpath = os.path.join(YOLO_ROI, dataname+".jpg")
        img = cv2.imread(datanameimgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # CHECK #
        # savepathcheck = "/home/wtepsan/NTU_RGB_train/Utils/_testcode_output/"+dataname
        # img.save(savepathcheck+"on_generated_roi.jpg")

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname


class ROI_dataset_on_the_fly(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None): 
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        
        # CHECK #
        # savepathcheck = "/home/wtepsan/NTU_RGB_train/Utils/_testcode_output/"+dataname
        # img.save(savepathcheck+"_on_the_fly_.jpg")

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname


class ROI_dataset_on_the_fly_openpose(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        img = segment_rgbbody_ntu.construct_st_roi(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip)
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname


class ROI_dataset_on_the_fly_openpose_update_fix_frames(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        img = segment_rgbbody_ntu.construct_st_roi_for_optical_flow_experiment(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip)
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname


class ROI_dataset_on_the_fly_openpose_update_fix_frames_opticalflow(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        img = framesXopticalflow.ROIframesXopticalflows(dataname, opticalflownumberlist=[3,6,9,12,15])
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname
    

class ROI_dataset_on_gen_roi_openpose_update_fix_frames(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        ROIPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_no/"
        imgpath = ROIPATH+ str(dataname) + ".jpg"
        img = Image.open(imgpath)
        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        # img = framesXopticalflow.ROIframesXopticalflows(dataname, opticalflownumberlist=[3,6,9,12,15])
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname

class ROI_dataset_on_gen_roi_openpose_update_fix_frames_opticalflow(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        ROIPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_opticalflow/"
        imgpath = ROIPATH+ str(dataname) + ".jpg"
        img = Image.open(imgpath)
        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        # img = framesXopticalflow.ROIframesXopticalflows(dataname, opticalflownumberlist=[3,6,9,12,15])
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname

class ROI_dataset_on_gen_roi_openpose_update_fix_frames_double(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        ROIPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_opticalflow/"
        imgpath = ROIPATH + str(dataname) + ".jpg"
        img1  = Image.open(imgpath)

        ROIXOPTICALFLOWPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_no/"
        imgxopticalflowpath = ROIXOPTICALFLOWPATH+ str(dataname) + ".jpg"
        img2 = Image.open(imgxopticalflowpath)

        
        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        # img = framesXopticalflow.ROIframesXopticalflows(dataname, opticalflownumberlist=[3,6,9,12,15])
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroi1path = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+"_1.jpg"
            saveroi2path = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+"_2.jpg"
            img1.save(saveroi1path)
            img2.save(saveroi2path)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label, dataname


class ROI_dataset_on_gen_roi_openpose_update_fix_frames_roistack(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])
        
        ROIPATH =  "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_no/"
        imgpath = ROIPATH + str(dataname) + ".jpg"
        img1  = Image.open(imgpath)

        ROIXOPTICALFLOWPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_opticalflow/"
        imgxopticalflowpath = ROIXOPTICALFLOWPATH+ str(dataname) + ".jpg"
        img2 = Image.open(imgxopticalflowpath)

        img = Image.new('RGB', (480,960))
        img.paste(img1, (0,0))
        img.paste(img2, (0,480))

        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath) 

        if self.transform is not None:
            img = self.transform(img) 
        
        return img, label, dataname

#### FOR JSSES ####
class ROI_dataset_on_the_fly_openpose_nJoint(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False, temporal_rgb_frames=5, choosing_frame_method='fix', joints=[0, 4, 7, 10, 13], w=48, h=48):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
        self.temporal_rgb_frames = temporal_rgb_frames
        self.choosing_frame_method = choosing_frame_method
        self.joints= joints
        self.w = w
        self.h = h
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])

        # img =  ROI_tools.construct_st_roi(dataname, YOLO_SKE, RGB_PATH, 5)
        # img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip)
        ####### segment_rgbbody_ntu.construct_st_roi((filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):
        # img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip)
        img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints= self.joints, w=self.w, h=self.h)
        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            saveroipath = "/home/wtepsan/NTU_MMNet_NewBaseLine/_results/roi/"+dataname+".jpg"
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname
    

#### FOR JSSES ####
class ROI_dataset_on_the_fly_openpose_genIMG(Dataset):
    def __init__(self, datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False, temporal_rgb_frames=5, choosing_frame_method='random', imginput='fullbody', joints=[0, 4, 7, 10, 13], w=48, h=48):  
        ####### , : True/False // random_interval: True/False // mode=test/train/eval
        self.datanames = datanames
        self.labels = datanamelabels
        self.transform = transform
        self.saveroi = saveroi
        self.evaluation = evaluation 
        self.random_interval = random_interval
        self.random_flip=random_flip
        self.temporal_rgb_frames = temporal_rgb_frames
        self.choosing_frame_method = choosing_frame_method
        self.imginput = imginput
        self.joints= joints
        self.w = w
        self.h = h
    
    def __len__(self):
        return len(self.datanames)
    
    def __getitem__(self, idx):

        dataname =  self.datanames[idx]
        label = torch.tensor(self.labels[idx])

        if self.imginput == 'fullbody':
            img =  segment_rgbbody_ntu.constructfullbodyfromvideo(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, w=96, h=288) # w=90, h=280)                                 # (dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints= self.joints, w=self.w, h=self.h)
        elif self.imginput == 'body2parts':
            img =  segment_rgbbody_ntu.construct2partbodyfromvideo(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, w=96, h=144)
        elif self.imginput == 'body3parts':
            img =  segment_rgbbody_ntu.construct3partbodyfromvideo(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, w=96, h=96)
        elif self.imginput == 'roi3':
            joints = [0, 4, 7]
            img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints=joints, w=self.w, h=self.h)
        elif self.imginput == 'roi4':
            joints = [4, 7, 10, 13]
            img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints=joints, w=self.w, h=self.h)
        elif self.imginput == 'roi5':
            joints = [0, 4, 7, 10, 13]
            img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints=joints, w=self.w, h=self.h)
        elif self.imginput == 'roi6':
            joints= [4, 7, 8, 11, 10, 13]
            img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints=joints, w=self.w, h=self.h)
        elif self.imginput == 'roi7':
            joints= [0, 4, 7, 8, 11, 10, 13]
            img = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, evaluation=self.evaluation, random_interval=self.random_interval, random_flip=self.random_flip, temporal_rgb_frames = self.temporal_rgb_frames, choosing_frame_method = self.choosing_frame_method, joints=joints, w=self.w, h=self.h)
        
        # img0 =  segment_rgbbody_ntu.constructfullbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=280)
        # img1 = segment_rgbbody_ntu.construct2partbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=140)
        # img2 =  segment_rgbbody_ntu.construct3partbodyfromvideo(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=90, h=96)
        # img3 = segment_rgbbody_ntu.construct_st_roi_3Joints(dataname, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=48, h=48)
        # img4 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[4, 7, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)
        # img5 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[0, 4, 7, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)
        # img6 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[4, 7, 8, 11, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)
        # img7 = segment_rgbbody_ntu.construct_st_roi_nJoints(dataname, joints=[0, 4, 7, 8, 11, 10, 13], random_interval=False, choosing_frame_method = 'random', w=48, h=48)


        ## CHECK ##
        if self.saveroi:
            print("SAVE ROI SAVE ROI", dataname)
            SAVEPATH = "/home/wtepsan/ROI_IMGinput-Experiments/_TESTS_/check-before-train-img_fttt/"+self.imginput+"/"
            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH) 
            saveroipath = SAVEPATH+dataname +".jpg"
            print(f"Image save at{saveroipath}")
            img.save(saveroipath)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, dataname


if __name__ == '__main__':
    #     # print("--- CODE TESTING ---") #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--begin')
    # parser.add_argument('--end')

    # arg = parser.parse_args()
    # begin = int(arg.begin)
    # end = int(arg.end)

    PATH_FRAMEIMAGES = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"
    datanames = os.listdir(PATH_FRAMEIMAGES) 
    datanames.sort()
    # datanames = datanames[begin:end]
    # print(f"Gen ROI begin from {begin} to {end}")
    # print(f"Number of Gen ROI images is {len(datanames)}")
    
    # for dataname in tqdm(datanames):
    #     img = segment_rgbbody_ntu.construct_st_roi_for_optical_flow_experiment(dataname)
    #     try:
    #         saveroipath = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_no/"+dataname+".jpg"
    #         img.save(saveroipath)
    #     except:
    #         print(f"Something Wrong with {dataname}")
    begin = 20
    end = 40
    datanames = datanames[begin:end]

    for dataname in datanames:
        ROIPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_no/"
        imgpath = ROIPATH + str(dataname) + ".jpg"
        img1  = Image.open(imgpath)

        ROIXOPTICALFLOWPATH = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_opticalflow/"
        imgxopticalflowpath = ROIXOPTICALFLOWPATH+ str(dataname) + ".jpg"
        img2 = Image.open(imgxopticalflowpath)

        img = Image.new('RGB', (480,960))
        img.paste(img1, (0,0))
        img.paste(img2, (0,480))

        saveroipath = "/home/wtepsan/ROI_Attentions/Utils/_testcode_output/"+dataname+".jpg"
        img.save(saveroipath)
