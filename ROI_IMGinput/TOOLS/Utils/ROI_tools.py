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

########################################
### YOLO KEYPOINTS ###
YOLO_Keypoints = {
    0: "nose", #head 
    1: "neck",
    2: "l_shoulder", 
    3: "l_elbow",
    4: "l_hand", #
    5: "r_shoulder",
    6: "r_elbow",
    7: "r_hand", #
    8: "l_hip",
    9: "l_knee",
    10: "l_foot", #
    11: "r_hip",
    12: "r_knee",
    13: "r_foot", #
    14: "l_eye",
    15: "r_eye", 
    16: "l_ear", 
}
########################################
### Setting Data Paths ### 

YOLO_ROI = "/project/lt200048-video/DatasetGen_NTU60/ROI_from_YOLOskeletons2_1person_clean/"
YOLO_SKE = "/project/lt200048-video/DatasetGen_NTU60/YOLOskeletons2_clean/"
RGB_PATH = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"

########################################
### READ JSON FILES ### 
def construct_yolo_json_dataname_frame_path(dataname, framenumber):
    if framenumber < 10:
        jsonframefile = dataname + "_00" + str(framenumber)+".json"
    elif framenumber < 100:
        jsonframefile = dataname + "_0" + str(framenumber)+".json"
    else:
        jsonframefile = dataname + "_" + str(framenumber)+".json"

    jsondatanameframepath = os.path.join(dataname, jsonframefile)

    return jsondatanameframepath

### S001C001P001R001A056_000.json -> skeletons size (numberofframe, 17, 3)
def read_skeletons_keypoints_from_dataname_frame(jsonfolder, dataname, framenumber):
    jsonframefilepath = construct_yolo_json_dataname_frame_path(dataname, framenumber)
    jsonframefilepath = os.path.join(jsonfolder, jsonframefilepath) # Actual json path
    
    if jsonframefilepath:
        with open(jsonframefilepath, 'r') as f:
            jsondict = json.load(f) # skeleton 
    else:
        jsondict = dict()
        jsondict["skeletons"] = []
    
    skeletons = jsondict["skeletons"]
    
    return skeletons #GET LIST OF SKELETON KEYPO

########################################
### Random List of Frame Numbers ### 

def random_frame_chosen(len_frames, temporal_rgb_frames, evaluation = False, random_interval = True, random_flip = False, flip = False):

    sequence_length = temporal_rgb_frames+1
    sample_interval = len_frames // sequence_length
    
    if sample_interval == 0:
        f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
        frame_range = f(temporal_rgb_frames, len_frames)
        
    else:
        if not evaluation:
            # Randomly choose sample interval and start frame
            # print("Randomly choose sample interval and start frame")
            start_i=0
            if random_interval: 
                sample_interval = np.random.randint(1, len_frames // sequence_length + 1)
                start_i = np.random.randint(0, len_frames - sample_interval * sequence_length + 1)
            #if random_roi_move:

            if random_flip:
                flip = np.random.random() < 0.5

            # aline selection to the two sides
            frame_range = range(start_i, len_frames, sample_interval)
    
        else:
            # Start at first frame and sample uniformly over sequence
            # print("Start at first frame and sample uniformly over sequence")
            start_i = 0
            flip = False
            frame_range = range(start_i, len_frames, sample_interval)

        n_frame_range = len(frame_range)
        begin = 0
        
        if n_frame_range > 5:
            begin = np.random.randint(0, n_frame_range-5)

        frame_range = frame_range[begin:begin+5]
        frame_range = [*frame_range]

        return frame_range


########################################
### GEN ROI ### 

def crop_body(dataname, yoloskepath=YOLO_SKE, rgbpath=RGB_PATH, framenumber=0):  

    actionid = int(dataname[dataname.find("A")+1:dataname.find("A")+4])

    frameimagename = str(framenumber)+".jpg"
    frameimagenamepath = rgbpath+"/"+dataname+"/"+frameimagename
    frame = Image.open(frameimagenamepath)

    skeletons = read_skeletons_keypoints_from_dataname_frame(yoloskepath, dataname, framenumber)

    skeletons = np.array(skeletons)

    if skeletons.ndim == 4:
        skeletons = skeletons[0]
    elif skeletons.ndim <3:
        skeletons = np.zeros((1,17,3))

    frameconcat=Image.new('RGB' , (96,480) , (0,0,0))
    
    if actionid < 50:
        if len(skeletons) > 0:
            nose_xy = skeletons[0, 0, 0:2]
            l_hand_xy = skeletons[0, 9, 0:2]
            r_hand_xy = skeletons[0, 10, 0:2]
            l_foot_xy = skeletons[0, 15, 0:2]
            r_foot_xy = skeletons[0, 16, 0:2]

            head = frame.crop((nose_xy[0]-48, nose_xy[1] - 48, nose_xy[0] + 48, nose_xy[1] + 48)) #2*3+1
            L_hand = frame.crop((l_hand_xy[0]-48, l_hand_xy[1] - 48, l_hand_xy[0] + 48, l_hand_xy[1] + 48))
            R_hand = frame.crop((r_hand_xy[0]-48, r_hand_xy[1] - 48, r_hand_xy[0] + 48, r_hand_xy[1] + 48))
            L_leg = frame.crop((l_foot_xy[0]-48, l_foot_xy[1] - 48, l_foot_xy[0] + 48, l_foot_xy[1] + 48))
            R_leg = frame.crop((r_foot_xy[0]-48, r_foot_xy[1] - 48, r_foot_xy[0] + 48, r_foot_xy[1] + 48))
            frameconcat.paste(head, (0,0))
            frameconcat.paste(L_hand, (0,96))
            frameconcat.paste(R_hand, (0,192))
            frameconcat.paste(L_leg, (0,288))
            frameconcat.paste(R_leg, (0,384))
    else:
        n_skeleton = len(skeletons) 
        if n_skeleton > 2:
            n_range = 2 
        else:
            n_range = n_skeleton

        for i in range(n_range):
            nose_xy = skeletons[i, 0, 0:2]
            l_hand_xy = skeletons[i, 9, 0:2]
            r_hand_xy = skeletons[i, 10, 0:2]
            l_foot_xy = skeletons[i, 15, 0:2]
            r_foot_xy = skeletons[i, 16, 0:2]
            

            head = frame.crop((nose_xy[0]-24, nose_xy[1] - 48, nose_xy[0] + 24, nose_xy[1] + 48)) #2*3+1
            L_hand = frame.crop((l_hand_xy[0]-24, l_hand_xy[1] - 48, l_hand_xy[0] + 24, l_hand_xy[1] + 48))
            R_hand = frame.crop((r_hand_xy[0]-24, r_hand_xy[1] - 48, r_hand_xy[0] + 24, r_hand_xy[1] + 48))
            L_leg = frame.crop((l_foot_xy[0]-24, l_foot_xy[1] - 48, l_foot_xy[0] + 24, l_foot_xy[1] + 48))
            R_leg = frame.crop((r_foot_xy[0]-24, r_foot_xy[1] - 48, r_foot_xy[0] + 24, r_foot_xy[1] + 48))

            frameconcat.paste(head, (i*48,0))
            frameconcat.paste(L_hand, (i*48,96))
            frameconcat.paste(R_hand, (i*48,192))
            frameconcat.paste(L_leg, (i*48,288))
            frameconcat.paste(R_leg, (i*48,384))
 
    return frameconcat

def construct_st_roi(dataname, yoloskepath = YOLO_SKE, rgbpath = RGB_PATH, temporalrgbframes=5):

    jsonfileslist = os.listdir(yoloskepath+"/"+dataname)
    numberofframe = len(jsonfileslist) 

    framerange = random_frame_chosen(numberofframe, temporalrgbframes, evaluation = False, random_interval = True, random_flip = False, flip = False)
    framerange.sort()
 
    roirgb = Image.new('RGB', (96*temporalrgbframes, 480), (0,0,0))
    for i in range(len(framerange)): 
        cropbodyrgb =crop_body(dataname, yoloskepath, rgbpath, framerange[i])
        roirgb.paste(cropbodyrgb, (i*96+1,0))

    return roirgb
    

if __name__ == '__main__':
    json_folder = YOLO_SKE
    image_folder = RGB_PATH
    data_names = os.listdir(json_folder)

    #data_names = data_names[0:100]
    
    temporal_rgb_frames = 5
    for i in range(100):
        len_frames = np.random.randint(20) + 50
        print(random_frame_chosen(len_frames, temporal_rgb_frames))
    # for data_name in tqdm(data_names):
    #     data_name_json = json_folder+"/"+ data_name 


        #roi_rgb = construct_st_roi(data_name, YOLO_SKE, RGB_PATH, 5)
        #roi_rgb.save("/project/lt200048-video/DatasetGen_NTU60/ROI_from_YOLOskelton2_1person_clean2/"+data_name+".jpg")

