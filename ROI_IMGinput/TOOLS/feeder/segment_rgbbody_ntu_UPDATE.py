#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:37 2019

@author: bruce
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import time
import random

try:
    from feeder import openpose_tools
except:
    import openpose_tools

frame_path = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"
openpose_path = "/project/lt200048-video/Dataset/NTU-RGBD/Openpose/openpose_ntu60_unzip/"
save_path = "/home/wtepsan/ROI_Attention/TEST/"

frame_path_120 = '/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/'
openpose_path_120 = '/project/lt200048-video/Dataset/NTU-RGBD/Openpose/openpose_ntu60_unzip/'
save_path_120 = '/project/lt200048-video/wtepsan/NTU_RGB_MMNet_NewBaseLine/data/ntu/roi/'  

PATH_FRAMEIMAGES = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"
PATH_OPENPOSE = "/project/lt200048-video/Dataset/NTU-RGBD/Openpose/openpose_ntu60_unzip/"
PATH_OPTICALFLOWIMAGE = "/project/lt200048-video/NTU_Optical/"

SAVEJSONFOLDER = "/project/lt200210-action/DatasetGen/LocalMaxMinFrames/"

debug = False

def filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id):
    skeleton_file_name = ''
    if setup_id/10 >= 1:
        skeleton_file_name = skeleton_file_name +'S0' + str(setup_id)
    else:
        skeleton_file_name = skeleton_file_name + 'S00' +  str(setup_id)

    if camera_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'C0' +  str(camera_id)
    else:
        skeleton_file_name = skeleton_file_name + 'C00' +  str(camera_id)

    if subject_id/100 >= 1:
        skeleton_file_name = skeleton_file_name + 'P' +  str(subject_id)
    elif subject_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'P0' +  str(subject_id)
    else:
        skeleton_file_name = skeleton_file_name + 'P00' +  str(subject_id)

    if duplicate_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'R0' +  str(duplicate_id)
    else:
        skeleton_file_name = skeleton_file_name + 'R00' +  str(duplicate_id)

    if action_id/100 >= 1:
        skeleton_file_name = skeleton_file_name + 'A' +  str(action_id)
    elif action_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'A0' +  str(action_id)
    else:
        skeleton_file_name = skeleton_file_name + 'A00' +  str(action_id)

    return skeleton_file_name

def openposeFile(frame_file, frame, skeleton_file_name, openpose_path):
    frame_file_ = frame_file + '/' + str(frame) + '.jpg'
    frame_ = ''
    if frame/100 >= 1:
        frame_ = str(frame)
    elif frame/10 >= 1:
        frame_ = '0' + str(frame)
    else:
        frame_ ='00' + str(frame)
    openpose_file_ = openpose_path + skeleton_file_name + '/' + skeleton_file_name + '_rgb_000000000'+ frame_ + '_keypoints.json'

    return openpose_file_, frame_file_

####################################
##### CHOOSING SKELETON METHOD ##### 
def openposecleanFile(openposefile):
    print("OH SOMETHING WRONG WITH SKELETON")
    openposefilename = openposefile.split("/")[-1]
    dataname = openposefilename[0:20]
    framenumber = int(openposefilename[25:37])
    #  S001C001P001R001A001_000.json
    if framenumber < 10:
        openposeclean_file = dataname+"/"+dataname+"_00"+str(framenumber)+".json"
    elif framenumber < 100:
        openposeclean_file = dataname+"/"+dataname+"_0"+str(framenumber)+".json"
    else:
        openposeclean_file = dataname+"/"+dataname+"_"+str(framenumber)+".json"

    return os.path.join(openpose_path_clean_1action, openposeclean_file)

def choose_skeleton_openpose(skeleton):
    numberofjoint = 18
    numberofpeople = len(skeleton['people'])
    peopleindex = 0


    if numberofpeople > 1:
        skeleton_array = np.zeros((numberofpeople, numberofjoint, 3))
        for people_index in range(numberofpeople):
            skeleton_pose = skeleton['people'][people_index]['pose_keypoints_2d']
            for i in range(numberofjoint):
                skeleton_array[people_index, i, :] = skeleton_pose[3*i: 3*i+3]
    
    print(skeleton_array.shape, skeleton_array)

    return peopleindex

####################################
####################################
def cropBodyfromPILimg(openpose_file, frame_pil_img, action_id, flip): 
    frame = frame_pil_img
    frame_width, frame_height = frame.size

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f)
 

    # calculate which people?
    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            # return ''
            print(f"{openpose_file} len(skeleton['people']) < 1")
            return Image.new('RGB', (96,480), (0,0,0))
        
        elif len(skeleton['people']) > 1:
            skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

            people_index = 0
            head_x = skeleton['people'][people_index]['pose_keypoints_2d'][0]
            head_y = skeleton['people'][people_index]['pose_keypoints_2d'][1]
            L_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][12]
            L_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][13]
            R_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][21]
            R_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][22]
            L_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][30]
            L_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][31]
            R_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][39]
            R_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][40]
            
        else: # len(skeleton['people']) == 1:
            people_index = 0
            head_x = skeleton['people'][people_index]['pose_keypoints_2d'][0]
            head_y = skeleton['people'][people_index]['pose_keypoints_2d'][1]
            L_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][12]
            L_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][13]
            R_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][21]
            R_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][22]
            L_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][30]
            L_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][31]
            R_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][39]
            R_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-48, head_y - 48, head_x + 48, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-48, L_hand_y - 48, L_hand_x + 48, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-48, R_hand_y - 48, R_hand_x + 48, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-48, L_leg_y - 48, L_leg_x + 48, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-48, R_leg_y - 48, R_leg_x + 48, R_leg_y + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))
        #print('frame_concat   if')
        return frame_concat

    elif len(skeleton['people']) > 1:
        # filter the non-subjects
        # print('frame_file: ', frame_file)
        # print('number of prople:', len(skeleton['people']))

        # cropping the area
        head_x = skeleton['people'][0]['pose_keypoints_2d'][0]
        head_y = skeleton['people'][0]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][0]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][0]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][0]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][0]['pose_keypoints_2d'][22]
        L_leg_x = skeleton['people'][0]['pose_keypoints_2d'][30]
        L_leg_y = skeleton['people'][0]['pose_keypoints_2d'][31]
        R_leg_x = skeleton['people'][0]['pose_keypoints_2d'][39]
        R_leg_y = skeleton['people'][0]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-24, head_y - 48, head_x + 24, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-24, L_hand_y - 48, L_hand_x + 24, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-24, R_hand_y - 48, R_hand_x + 24, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-24, L_leg_y - 48, L_leg_x + 24, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-24, R_leg_y - 48, R_leg_x + 24, R_leg_y + 48))

        head_x_1 = skeleton['people'][1]['pose_keypoints_2d'][0]
        head_y_1 = skeleton['people'][1]['pose_keypoints_2d'][1]
        L_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][12]
        L_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][13]
        R_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][21]
        R_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][22]
        L_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][30]
        L_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][31]
        R_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][39]
        R_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][40]

        head_1 = frame.crop((head_x_1-24, head_y_1 - 48, head_x_1 + 24, head_y_1 + 48)) #2*3+1
        L_hand_1 = frame.crop((L_hand_x_1-24, L_hand_y_1 - 48, L_hand_x_1 + 24, L_hand_y_1 + 48))
        R_hand_1 = frame.crop((R_hand_x_1-24, R_hand_y_1 - 48, R_hand_x_1 + 24, R_hand_y_1 + 48))
        L_leg_1 = frame.crop((L_leg_x_1-24, L_leg_y_1 - 48, L_leg_x_1 + 24, L_leg_y_1 + 48))
        R_leg_1 = frame.crop((R_leg_x_1-24, R_leg_y_1 - 48, R_leg_x_1 + 24, R_leg_y_1 + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(R_hand, (0,96))
            frame_concat.paste(R_hand_1, (48,96))
            frame_concat.paste(L_hand, (0,192))
            frame_concat.paste(L_hand_1, (48,192))
            frame_concat.paste(R_leg, (0,288))
            frame_concat.paste(R_leg_1, (48,288))
            frame_concat.paste(L_leg, (0,384))
            frame_concat.paste(L_leg_1, (48,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(L_hand_1, (48,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(R_hand_1, (48,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(L_leg_1, (48,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat.paste(R_leg_1, (48,384))
        #print('frame_concat   elif')
        return frame_concat
    else:
        #print(len(skeleton['people']))
        #return ''
        print(f"{openpose_file} something wrong")
        return Image.new('RGB', (96,480), (0,0,0))


def cropBody(openpose_file, frame_file, action_id, flip):
    #upper=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #lower=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #whole=Image.new( 'RGB' , (224,448) , (0,0,0) )
    
    frame = Image.open(frame_file)
    frame_width, frame_height = frame.size

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f)

    # print("CHEKCING choose_skeleton_openpose:")

    # choose_skeleton_openpose(skeleton)

    # calculate which people?
    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        
        elif len(skeleton['people']) > 1:
            ###OLD Choosing index of skeleton by Mask Depth###
            # #print('frame_file: ', frame_file)
            # #print('number of prople:', len(skeleton['people']))
            # frame_file_split = frame_file.split('/')
            # frame_num = int(frame_file_split[7].split('.')[0])
            # if frame_num/1000 >= 1:
            #     depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-0000'+str(frame_num)+'.png'
            # elif frame_num/100 >= 1:
            #     depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-00000'+str(frame_num)+'.png'
            # elif frame_num/10 >= 1:
            #     depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-000000'+str(frame_num)+'.png'
            # else:
            #     depth_frame_file = depth_path + frame_file_split[6] +'/MDepth-0000000'+str(frame_num)+'.png'
            # #print(depth_frame_file)

            # depth_frame = Image.open(depth_frame_file)
            # depth_frame = depth_frame.resize((1338,1080))
            # depth_frame_arr = np.fromiter(iter(depth_frame.getdata()), np.uint16)
            # depth_frame_arr.resize(1080, 1338)

            # people_dist_min = 4500
            # joint = 1
            # for p in range(len(skeleton['people'])):
            #     #for i in []: # openpose joint 2, 5, 8, 11
            #     x = int(skeleton['people'][p]['pose_keypoints_2d'][(joint+1)*3-3])
            #     y = int(skeleton['people'][p]['pose_keypoints_2d'][(joint+1)*3-2])
            #     k = 0
            #     if x >= 1338:
            #         x = 1336
            #         #print('frame_file: x ', frame_file)
            #     if y >= 1080:
            #         y = 1078
            #         #print('frame_file: y ', frame_file)
            #     #print('(x, y): ', x, y)
            #     people_dist = 0
            #     for i in [-3, -2,-1, 0, 1, 2, 3]:
            #         for j in [-3, -2,-1, 0, 1, 2, 3]:
            #             if depth_frame_arr[y+i][x+j-291] > 0:
            #                 people_dist = people_dist + depth_frame_arr[y+i][x+j-291]
            #                 k = k + 1
            #     #print(p, people_dist)
            #     if people_dist > 0:
            #         people_dist = people_dist/k
            #         #print('people_dist, k: ',people_dist, k)
            #         if people_dist <= people_dist_min:
            #             people_dist_min = people_dist
            #             people_index = p

            ### Replace by our method ###
            # people_index = choose_skeleton_openpose(skeleton)

            skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

            people_index = 0
            head_x = skeleton['people'][people_index]['pose_keypoints_2d'][0]
            head_y = skeleton['people'][people_index]['pose_keypoints_2d'][1]
            L_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][12]
            L_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][13]
            R_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][21]
            R_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][22]
            L_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][30]
            L_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][31]
            R_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][39]
            R_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][40]
            
        else: # len(skeleton['people']) == 1:
            people_index = 0
            head_x = skeleton['people'][people_index]['pose_keypoints_2d'][0]
            head_y = skeleton['people'][people_index]['pose_keypoints_2d'][1]
            L_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][12]
            L_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][13]
            R_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][21]
            R_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][22]
            L_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][30]
            L_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][31]
            R_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][39]
            R_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-48, head_y - 48, head_x + 48, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-48, L_hand_y - 48, L_hand_x + 48, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-48, R_hand_y - 48, R_hand_x + 48, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-48, L_leg_y - 48, L_leg_x + 48, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-48, R_leg_y - 48, R_leg_x + 48, R_leg_y + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))
        #print('frame_concat   if')
        return frame_concat

    elif len(skeleton['people']) > 1:
        # filter the non-subjects
        # print('frame_file: ', frame_file)
        # print('number of prople:', len(skeleton['people']))

        # cropping the area
        head_x = skeleton['people'][0]['pose_keypoints_2d'][0]
        head_y = skeleton['people'][0]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][0]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][0]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][0]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][0]['pose_keypoints_2d'][22]
        L_leg_x = skeleton['people'][0]['pose_keypoints_2d'][30]
        L_leg_y = skeleton['people'][0]['pose_keypoints_2d'][31]
        R_leg_x = skeleton['people'][0]['pose_keypoints_2d'][39]
        R_leg_y = skeleton['people'][0]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-24, head_y - 48, head_x + 24, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-24, L_hand_y - 48, L_hand_x + 24, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-24, R_hand_y - 48, R_hand_x + 24, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-24, L_leg_y - 48, L_leg_x + 24, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-24, R_leg_y - 48, R_leg_x + 24, R_leg_y + 48))

        head_x_1 = skeleton['people'][1]['pose_keypoints_2d'][0]
        head_y_1 = skeleton['people'][1]['pose_keypoints_2d'][1]
        L_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][12]
        L_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][13]
        R_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][21]
        R_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][22]
        L_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][30]
        L_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][31]
        R_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][39]
        R_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][40]

        head_1 = frame.crop((head_x_1-24, head_y_1 - 48, head_x_1 + 24, head_y_1 + 48)) #2*3+1
        L_hand_1 = frame.crop((L_hand_x_1-24, L_hand_y_1 - 48, L_hand_x_1 + 24, L_hand_y_1 + 48))
        R_hand_1 = frame.crop((R_hand_x_1-24, R_hand_y_1 - 48, R_hand_x_1 + 24, R_hand_y_1 + 48))
        L_leg_1 = frame.crop((L_leg_x_1-24, L_leg_y_1 - 48, L_leg_x_1 + 24, L_leg_y_1 + 48))
        R_leg_1 = frame.crop((R_leg_x_1-24, R_leg_y_1 - 48, R_leg_x_1 + 24, R_leg_y_1 + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(R_hand, (0,96))
            frame_concat.paste(R_hand_1, (48,96))
            frame_concat.paste(L_hand, (0,192))
            frame_concat.paste(L_hand_1, (48,192))
            frame_concat.paste(R_leg, (0,288))
            frame_concat.paste(R_leg_1, (48,288))
            frame_concat.paste(L_leg, (0,384))
            frame_concat.paste(L_leg_1, (48,384))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(L_hand_1, (48,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(R_hand_1, (48,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(L_leg_1, (48,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat.paste(R_leg_1, (48,384))
        #print('frame_concat   elif')
        return frame_concat
    else:
        #print(len(skeleton['people']))
        return ''

# to set the continue point S006C002P019R002A039

done = False
'''
for setup_id in range(1,21):     # 1:20 Diferernt height and distance
    if setup_id < sss:
        continue
    for camera_id in range(1,4):     # 1:3 camera views
        if setup_id < sss + 1 and camera_id < ccc:
            continue
        for subject_id in range(1,41):   # 1:40 distinct subjects aged between 10 to 35
            if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp:
                continue
            for duplicate_id in range(1,3):  # 1:2 Performance action twice, one to left camera, one to right camera
                if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr:
                    continue
                for action_id in range(1,61):    # 1:60 Action class [11,12,30,31,53]
                    if setup_id < sss + 1 and camera_id < ccc + 1 and subject_id < ppp + 1 and duplicate_id < rrr +1 and action_id < aaa:
                        continue
'''

def construct_st_roi(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):
    sequence_length = temporal_rgb_frames + 1

    setup_id = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(filename[filename.find('A') + 1:filename.find('A') + 4])

    #if action_id > 60:
    #    return ''

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name
    #print(frame_file)
        
    fivefs_concat=Image.new('RGB', (96*temporal_rgb_frames,480) , (0,0,0) )
    if os.path.isdir(frame_file):# and action_id == 50:

        # load the frames' file name from folder
        frames = os.listdir(frame_file)
    
        start_i = 0
        # checked all len(frames) are  > 6
        sample_interval = len(frames) // sequence_length
        flip = False
        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            if not evaluation:
                # Randomly choose sample interval and start frame
                start_i=0
                if random_interval:
                    #print('random_interval:::::::::::::',random_interval)
                    sample_interval = np.random.randint(1, len(frames) // sequence_length + 1)
                    start_i = np.random.randint(0, len(frames) - sample_interval * sequence_length + 1)
                #if random_roi_move:

                if random_flip:
                    flip = np.random.random() < 0.5

                # aline selection to the two sides
                frame_range = range(start_i, len(frames), sample_interval)

                #print(flip)
                #print(start_i, sample_interval)
            else:
                # Start at first frame and sample uniformly over sequence
                start_i = 0
                flip = False
                frame_range = range(start_i, len(frames), sample_interval)

        # print("CHECK filename and len and frame_range", filename, len(frames), frame_range)

        i=0
        for frame in frame_range:

            if frame != 0 and frame != (sequence_length*sample_interval):

                #print(frame)
                if not debug:
                    #openpose_file_, frame_file_ = openposeFile(frame_file, frame, skeleton_file_name, openpose_path)
                    frame_croped = ''
                    frame_ = frame
                    # find the closest non'' frame
                    while frame_croped == '':
                        if action_id < 61:
                            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path)
                        else:
                            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path_120)
                        # both openpose and RGB frame should exist
                        if os.path.isfile(openpose_file_) and os.path.isfile(frame_file_):
                            frame_croped = cropBody(openpose_file_, frame_file_, action_id, flip)
                            #print('file consistent: ',openpose_file_)
                        else:
                            #print('file_unconsistent: ',openpose_file_, os.path.isfile(openpose_file_))
                            string = str(frame_file_ + " {}\n" + openpose_file_ + ' {}\n').format(os.path.isfile(frame_file_), os.path.isfile(openpose_file_))
                            with open('file_unconsistent_crop.txt', 'a') as fd:
                                fd.write(f'\n{string}')

                        frame_ = frame_ + 1
                        if frame_ > len(frames):
                            frame_croped = Image.new( 'RGB' , (96,480) , (0,0,0) )
                            break

                    fivefs_concat.paste(frame_croped, (i*96,0))
                    i+=1
            '''
            plt.imshow(fivefs_concat)
            plt.suptitle('Corpped Body Parts')
            plt.show()
            '''
        # for generating st-roi
        '''
        if action_id < 61:
            frames_save = save_path + skeleton_file_name +'.png'
        else:
            frames_save = save_path_120 + skeleton_file_name +'.png'
        fivefs_concat.save(frames_save,"PNG")
        time.sleep(0.01)
        #'''
    return fivefs_concat


def construct_st_roi_for_optical_flow_experiment(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames=5):
    
    sequence_length = temporal_rgb_frames + 1
    flip = False

    setup_id = int(
        filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(
        filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(
        filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(
        filename[filename.find('A') + 1:filename.find('A') + 4])

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name

    fivefs_concat=Image.new('RGB', (96*temporal_rgb_frames,480) , (0,0,0))

    if os.path.isdir(frame_file):

        dataname = filename.split(".")[0]
        opticalflowimagepath = PATH_OPTICALFLOWIMAGE + dataname + "/"
        opticalflowimagelist = os.listdir(opticalflowimagepath)

        frames = os.listdir(frame_file)
        nframeimage = len(frames)
        skipnumber = int(nframeimage/20)

        opticalflownumberlist = [3,6,9,12,15]
        if "0.jpg" in opticalflowimagelist:
            frameimagenlist = [int((opticalflownumber)*(skipnumber)) for opticalflownumber in opticalflownumberlist]
        else:
            frameimagenlist = [int((opticalflownumber-1)*(skipnumber)) for opticalflownumber in opticalflownumberlist]

        for i in range(5):
            frame = frameimagenlist[i]
            frame_croped = ''
            frame_ = frame

            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path) 

            try: 
                frame_croped = cropBody(openpose_file_, frame_file_, action_id, flip)
                if frame_croped == '':
                    print(f"{filename} return empty string")
                    frame_croped = Image.new( 'RGB' , (96,480) , (0,0,0) )
            except: 
                frame_croped = Image.new( 'RGB' , (96,480) , (0,0,0) )

            fivefs_concat.paste(frame_croped, (i*96,0))


    return fivefs_concat


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

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

def framechosen(n_frame, n_chosen, choosing_frame_method='random'):
    # method randombegin_randomlength, random_from_each_interval, random, fix
    intervallength = n_frame//n_chosen 
    n_list = list(range(n_frame))

    if choosing_frame_method == 'randombegin_randomlength':
        begin = random.randint(0, intervallength)
        length = random.randint(0, intervallength)
        chosen_frames = [begin + i*length for i in range(n_chosen)]
    elif choosing_frame_method == 'random_from_each_interval':
        chosen_frames = []
        for i in range(n_chosen):
            length = random.randint(0, intervallength)
            chosen_frames.append(length+i*intervallength)
    elif choosing_frame_method == 'random':
        chosen_frames = random.sample(n_list, n_chosen)  # Randomly choose n_chosen numbers from the list
        chosen_frames.sort()
    else: # CHOOSE AT THE CENTER AT EACH INTERVAL
        chosen_frames = [intervallength//2 + i*intervallength for i in range(n_chosen)]

    return chosen_frames


def cropFramefromJoint(skeleton, frame, people_index, joint, w=48, h=48):
    x_index = 3*joint
    y_index = 3*joint+1
    x = skeleton['people'][people_index]['pose_keypoints_2d'][x_index]
    y = skeleton['people'][people_index]['pose_keypoints_2d'][y_index]

    framecrop = frame.crop((x-w, y - h, x + w, y + h)) 

    return framecrop

def cropBodyUpdate(openpose_file, frame_file, action_id, flip, joints=[0,4,7,10,13], w=48, h=48): 
    
    frame = Image.open(frame_file)
    frame_width, frame_height = frame.size

    n_joints = len(joints)
    frame_concat=Image.new('RGB', (2*w, 2*n_joints*h) , (0,0,0))

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f) 

    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        
        elif len(skeleton['people']) > 1: 

            skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

        people_index = 0
        for i in range(n_joints):
            frame_crop = cropFramefromJoint(skeleton, frame, people_index, joints[i], w, h) 
            frame_concat.paste(frame_crop, (0, 2*h*i))
            
        if flip: 
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT) 

        return frame_concat

    elif len(skeleton['people']) > 1:

        people_index = 0
        n_joints = len(joints)
        frame_concat=Image.new('RGB', (2*w, 2*n_joints*h) , (0,0,0))

        for i in range(n_joints): 
            frame_crop = cropFramefromJoint(skeleton, frame, 0, joints[i], int(w/2), h) 
            frame_concat.paste(frame_crop, (0, 2*h*i))

            frame_crop = cropFramefromJoint(skeleton, frame, 1, joints[i], int(w/2), h) 
            frame_concat.paste(frame_crop, (w, 2*h*i))
            
        if flip: 
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT) 

        return frame_concat 
    
    else:
        #print(len(skeleton['people']))
        return ''

def construct_st_roi_nJoints(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', joints=[0,4,7,10,13], w=48, h=48):
    
    n_joints = len(joints) 
    sequence_length = temporal_rgb_frames + 1

    flip = False

    setup_id = int(
        filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(
        filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(
        filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(
        filename[filename.find('A') + 1:filename.find('A') + 4])

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name

    fivefs_concat=Image.new('RGB', (2*w*temporal_rgb_frames, 2*h*n_joints) , (0,0,0))

    if os.path.isdir(frame_file):
        frames = os.listdir(frame_file)  
        sample_interval = len(frames) // sequence_length

        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            if choosing_frame_method == 'min':
                        # def framechosen_fft(dataname, n_frame, n_chosen, choosing_frame_method='max'): 
                frame_range = framechosen_fft(skeleton_file_name, len(frames), temporal_rgb_frames, choosing_frame_method='min')
            elif choosing_frame_method == 'max':
                frame_range = framechosen_fft(skeleton_file_name, len(frames), temporal_rgb_frames, choosing_frame_method='max')
            elif choosing_frame_method == 'mix': 
                frame_range = framechosen_fft(skeleton_file_name, len(frames), temporal_rgb_frames, choosing_frame_method='mix')
            elif choosing_frame_method == 'btwmax':
                frame_range = framechosen_fft(skeleton_file_name, len(frames), temporal_rgb_frames, choosing_frame_method='btwmax')
            elif choosing_frame_method == 'btwmin': 
                frame_range = framechosen_fft(skeleton_file_name, len(frames), temporal_rgb_frames, choosing_frame_method='btwmin')
            else:
                print("ELSEELSEELSEELSE")
                if not evaluation:
                    frame_range = framechosen(len(frames), temporal_rgb_frames, choosing_frame_method=choosing_frame_method)
                else:
                    start_i = sample_interval//2
                    flip = False
                    frame_range = range(start_i, len(frames), sample_interval)
                
        i=0
        for frame in frame_range:
            # frame = frameimagenlist[i]
            frame_croped = ''
            frame_ = frame

            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path) 

            try: 
                frame_croped = cropBodyUpdate(openpose_file_, frame_file_, action_id, flip, joints=joints, w=w, h=48)
                if frame_croped == '':
                    print(f"{filename} return empty string")
                    frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h) , (0,0,0) )
            except: 
                frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h), (0,0,0) )

            fivefs_concat.paste(frame_croped, (i*2*w,0))
            i+=1

    return fivefs_concat


##############################################################################################################
##############################################################################################################
##############################################################################################################

def cropFramefromXY(frame, x, y, w=48, h=48): 
    framecrop = frame.crop((x-w, y - h, x + w, y + h)) 
    return framecrop

def findcenterofskeleton(skeleton_XY_positions):
    n_positions = int(len(skeleton_XY_positions)/3)
    positions_x = [skeleton_XY_positions[3*i] for i in range(n_positions)]
    positions_y = [skeleton_XY_positions[3*i+1] for i in range(n_positions)] 
    left = min([value for value in positions_x if value!=0]) 
    right = max([value for value in positions_x if value!=0])
    top = min([value for value in positions_y if value!=0]) 
    bottom = max([value for value in positions_y if value!=0])

    ## CALCULATE CENTER ##
    center_x_skeleton = (left+right)/2
    center_y_skeleton = (top+bottom)/2

    return center_x_skeleton, center_y_skeleton

openposekeys = {
    "nose":0,
    "neck":1,
    "Reye":14,
    "Leye":15,
    "Rear":16,
    "Lear":17,
    "Rshoulder":2,
    "Rellbow":3,
    "Rwrist":4,
    "Lshoulder":5,
    "Lelbow":6,
    "Lwrist":7,

    "Rhip":8,
    "Lhip":11,

    "Rknee":9,
    "Rankle":10,
    "Lknee":12,
    "Lankle":13,

}


def findcenterof2bodypart(skeleton_XY_positions):

    top_joints_name = ["nose", "neck", "Reye", "Leye", "Rear", "Lear", "Rshoulder", "Rellbow", "Rwrist", "Lshoulder", "Lelbow", "Lwrist", "Rhip", "Lhip"]
    top_joints_index = [openposekeys[name] for name in top_joints_name]

    lower_joints_index = ["Rhip", "Lhip", "Rknee", "Rankle", "Lknee", "Lankle"]
    lower_joints_index = [openposekeys[name] for name in lower_joints_index]

    try:
        positions_x = [skeleton_XY_positions[3*i] for i in top_joints_index]
        positions_x = [num for num in positions_x if num > 0]

        positions_y = [skeleton_XY_positions[3*i+1] for i in top_joints_index] 
        positions_y = [num for num in positions_y if num > 0]

        center_x_top = sum(positions_x) / len(positions_x)
        center_y_top = sum(positions_y) / len(positions_y)
    except:
        center_x_top = 0
        center_y_top = 0


    try:
        positions_x = [skeleton_XY_positions[3*i] for i in lower_joints_index]
        positions_x = [num for num in positions_x if num > 0]

        positions_y = [skeleton_XY_positions[3*i+1] for i in lower_joints_index] 
        positions_y = [num for num in positions_y if num > 0]

        center_x_lower = sum(positions_x) / len(positions_x)
        center_y_lower = sum(positions_y) / len(positions_y)
    except:
        center_x_lower = 0
        center_y_lower = 0

    return center_x_top, center_y_top, center_x_lower, center_y_lower


def findcenterof3bodypart(skeleton_XY_positions):

    top_joints_name = ["nose", "neck", "Reye", "Leye", "Rear", "Lear", "Rshoulder", "Lshoulder"]
    top_joints_index = [openposekeys[name] for name in top_joints_name]

    middle_joints_name = ["Rshoulder", "Rellbow", "Rwrist", "Lshoulder", "Lelbow", "Lwrist", "Lhip", "Rhip"]
    middle_joints_index = [openposekeys[name] for name in middle_joints_name]

    lower_joints_index = ["Rknee", "Rankle", "Lknee", "Lankle"]
    lower_joints_index = [openposekeys[name] for name in lower_joints_index]

    try:
        positions_x = [skeleton_XY_positions[3*i] for i in top_joints_index]
        positions_x = [num for num in positions_x if num > 0]

        positions_y = [skeleton_XY_positions[3*i+1] for i in top_joints_index] 
        positions_y = [num for num in positions_y if num > 0]

        center_x_top = sum(positions_x) / len(positions_x)
        center_y_top = sum(positions_y) / len(positions_y)
    except:
        center_x_top = 0
        center_y_top = 0

    try:
        positions_x = [skeleton_XY_positions[3*i] for i in middle_joints_index]
        positions_x = [num for num in positions_x if num > 0]

        positions_y = [skeleton_XY_positions[3*i+1] for i in middle_joints_index] 
        positions_y = [num for num in positions_y if num > 0]

        center_x_middle = sum(positions_x) / len(positions_x)
        center_y_middle = sum(positions_y) / len(positions_y)
    except:
        center_x_middle = 0
        center_y_middle = 0


    try:
        positions_x = [skeleton_XY_positions[3*i] for i in lower_joints_index]
        positions_x = [num for num in positions_x if num > 0]

        positions_y = [skeleton_XY_positions[3*i+1] for i in lower_joints_index] 
        positions_y = [num for num in positions_y if num > 0]

        center_x_lower = sum(positions_x) / len(positions_x)
        center_y_lower = sum(positions_y) / len(positions_y)
    except:
        center_x_lower = 0
        center_y_lower = 0

    return center_x_top, center_y_top, center_x_middle, center_y_middle, center_x_lower, center_y_lower


def cropFullBody(openpose_file, frame_file, action_id, flip=False, w=48, h=48): 

    frame = Image.open(frame_file)
    framecrop =  Image.new('RGB', (2*w, (2*h)) , (0,0,0))

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f)

    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        
        elif len(skeleton['people']) > 1: 

            skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

        skeleton_XY_positions = skeleton['people'][0]['pose_keypoints_2d']

        center_x_skeleton, center_y_skeleton = findcenterofskeleton(skeleton_XY_positions)
        framecrop = cropFramefromXY(frame, center_x_skeleton, center_y_skeleton, w=w, h=h)
        
        if flip:
            framecrop=framecrop.transpose(Image.FLIP_LEFT_RIGHT)

        return framecrop
        
    elif len(skeleton['people']) > 1:
        skeleton_XY_positions = skeleton['people'][0]['pose_keypoints_2d']
        center_x_skeleton, center_y_skeleton = findcenterofskeleton(skeleton_XY_positions)
        framecrop_person = cropFramefromXY(frame, center_x_skeleton, center_y_skeleton, w=int(w/2), h=h)
        framecrop.paste(framecrop_person, (0, 0))

        skeleton_XY_positions = skeleton['people'][1]['pose_keypoints_2d']
        center_x_skeleton, center_y_skeleton = findcenterofskeleton(skeleton_XY_positions)
        framecrop_person = cropFramefromXY(frame, center_x_skeleton, center_y_skeleton, w=int(w/2), h=h)
        framecrop.paste(framecrop_person, (w, 0))

        return framecrop 
    
    else:
        return ''

def crop2bodypart(openpose_file, frame_file, action_id, flip=False, w=48, h=48): 

    frame = Image.open(frame_file) 
    frame_concat =  Image.new('RGB', (2*w, (2*h*2)) , (0,0,0))

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f) 

    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        
        elif len(skeleton['people']) > 1: 

            skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

        skeleton_XY_positions = skeleton['people'][0]['pose_keypoints_2d']
        center_x_top, center_y_top, center_x_lower, center_y_lower = findcenterof2bodypart(skeleton_XY_positions)
        framecrop_part = cropFramefromXY(frame, center_x_top, center_y_top, w=w, h=h)
        frame_concat.paste(framecrop_part, (0, 0)) 
        framecrop_part = cropFramefromXY(frame, center_x_lower, center_y_lower, w=w, h=h)
        frame_concat.paste(framecrop_part, (0, int(2*h)))
        
        if flip:
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)

        return frame_concat
        
    elif len(skeleton['people']) > 1:

        skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

        skeleton_XY_positions = skeleton['people'][0]['pose_keypoints_2d']
        center_x_top, center_y_top, center_x_lower, center_y_lower = findcenterof2bodypart(skeleton_XY_positions)
        framecrop_part = cropFramefromXY(frame, center_x_top, center_y_top, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (0, 0)) 
        framecrop_part = cropFramefromXY(frame, center_x_lower, center_y_lower, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (0, int(2*h)))

        skeleton_XY_positions = skeleton['people'][1]['pose_keypoints_2d']
        center_x_top, center_y_top, center_x_lower, center_y_lower = findcenterof2bodypart(skeleton_XY_positions)
        framecrop_part = cropFramefromXY(frame, center_x_top, center_y_top, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (w, 0))
        framecrop_part = cropFramefromXY(frame, center_x_lower, center_y_lower, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (w, int(2*h)))

        return frame_concat
    
    else:
        return ''

def crop3bodypart(openpose_file, frame_file, action_id, flip=False, w=48, h=48): 

    frame = Image.open(frame_file) 
    frame_concat =  Image.new('RGB', (2*w, (2*h*3)) , (0,0,0))

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f) 

    if len(skeleton['people']) == 1 or action_id < 50: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        
        elif len(skeleton['people']) > 1: 

            skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

        skeleton_XY_positions = skeleton['people'][0]['pose_keypoints_2d']
        center_x_top, center_y_top, center_x_middle, center_y_middle, center_x_lower, center_y_lower = findcenterof3bodypart(skeleton_XY_positions)
 
        framecrop_part = cropFramefromXY(frame, center_x_top, center_y_top, w=w, h=h)
        frame_concat.paste(framecrop_part, (0, 0))
        framecrop_part = cropFramefromXY(frame, center_x_middle, center_y_middle, w=w, h=h)
        frame_concat.paste(framecrop_part, (0, int(2*h)))
        framecrop_part = cropFramefromXY(frame, center_x_lower, center_y_lower, w=w, h=h)
        frame_concat.paste(framecrop_part, (0, int(4*h)))
        
        if flip:
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)

        return frame_concat
        
    elif len(skeleton['people']) > 1:

        skeleton = openpose_tools.clean_openpose_skeleton(openpose_file, action_id)

        skeleton_XY_positions = skeleton['people'][0]['pose_keypoints_2d']
        center_x_top, center_y_top, center_x_middle, center_y_middle, center_x_lower, center_y_lower = findcenterof3bodypart(skeleton_XY_positions)
        framecrop_part = cropFramefromXY(frame, center_x_top, center_y_top, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (0, 0))
        framecrop_part = cropFramefromXY(frame, center_x_middle, center_y_middle, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (0, int(2*h)))
        framecrop_part = cropFramefromXY(frame, center_x_lower, center_y_lower, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (0, int(4*h)))


        skeleton_XY_positions = skeleton['people'][1]['pose_keypoints_2d']
        center_x_top, center_y_top, center_x_middle, center_y_middle, center_x_lower, center_y_lower = findcenterof3bodypart(skeleton_XY_positions)
        framecrop_part = cropFramefromXY(frame, center_x_top, center_y_top, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (w, 0))
        framecrop_part = cropFramefromXY(frame, center_x_middle, center_y_middle, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (w, int(2*h)))
        framecrop_part = cropFramefromXY(frame, center_x_lower, center_y_lower, w=int(w/2), h=h)
        frame_concat.paste(framecrop_part, (w, int(4*h)))

        return frame_concat
    
    else:
        return ''

def constructfullbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5,  choosing_frame_method = 'random', w=48, h=48):
            #(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'fix', joints=[0,4,7,10,13], w=48, h=48):
    sequence_length = temporal_rgb_frames + 1

    flip = False

    setup_id = int(
        filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(
        filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(
        filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(
        filename[filename.find('A') + 1:filename.find('A') + 4])

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name

    fivefs_concat=Image.new('RGB', (2*w*temporal_rgb_frames, 2*h) , (0,0,0))

    if os.path.isdir(frame_file):
        frames = os.listdir(frame_file)  
        sample_interval = len(frames) // sequence_length

        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            # frame_range =  random.sample(frames_list, 5)
            # frame_range.sort()

            if not evaluation:
                frame_range = framechosen(len(frames), temporal_rgb_frames, choosing_frame_method=choosing_frame_method)
            else:
                start_i = sample_interval//2
                flip = False
                frame_range = range(start_i, len(frames), sample_interval)

        i=0
        for frame in frame_range:
            # frame = frameimagenlist[i]
            frame_croped = ''
            frame_ = frame

            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path) 

            try: 
                frame_croped = cropFullBody(openpose_file_, frame_file_, action_id, flip=False, w=w, h=h)
                                #cropFullBody(openpose_file, frame_file, action_id, flip=False, w=48, h=48)
                if frame_croped == '':
                    print(f"{filename} return empty string")
                    frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h) , (0,0,0) )
            except: 
                frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h), (0,0,0) )

            fivefs_concat.paste(frame_croped, (i*2*w,0))
            i+=1

    return fivefs_concat

def construct2partbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=48, h=48):
    # (filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5,  choosing_frame_method = 'random', w=48, h=48):
    sequence_length = temporal_rgb_frames + 1

    flip = False
    setup_id = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(filename[filename.find('A') + 1:filename.find('A') + 4])

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name

    fivefs_concat=Image.new('RGB', (2*w*temporal_rgb_frames, 2*2*h) , (0,0,0))

    if os.path.isdir(frame_file):
        frames = os.listdir(frame_file)  
        sample_interval = len(frames) // sequence_length

        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            # frame_range =  random.sample(frames_list, 5)
            # frame_range.sort()

            if not evaluation:
                frame_range = framechosen(len(frames), temporal_rgb_frames, choosing_frame_method=choosing_frame_method)
            else:
                start_i = sample_interval//2
                flip = False
                frame_range = range(start_i, len(frames), sample_interval)

        i=0
        for frame in frame_range:
            # frame = frameimagenlist[i]
            frame_croped = ''
            frame_ = frame

            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path) 

            try:
                                # crop3bodypart(openpose_file, frame_file, action_id, flip=False, w=48, h=48)
                frame_croped = crop2bodypart(openpose_file_, frame_file_, action_id, flip=False, w=w, h=h)
                if frame_croped == '':
                    print(f"{filename} return empty string")
                    frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h) , (0,0,0) )
            except: 
                frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h), (0,0,0) )

            fivefs_concat.paste(frame_croped, (i*2*w,0))
            i+=1

    return fivefs_concat

def construct3partbodyfromvideo(filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5, choosing_frame_method = 'random', w=48, h=48):
    # (filename, evaluation=False, random_interval=False, random_roi_move=False, random_flip=False, temporal_rgb_frames = 5,  choosing_frame_method = 'random', w=48, h=48):
    sequence_length = temporal_rgb_frames + 1

    flip = False
    setup_id = int(filename[filename.find('S') + 1:filename.find('S') + 4])
    camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])
    subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
    duplicate_id = int(filename[filename.find('R') + 1:filename.find('R') + 4])
    action_id = int(filename[filename.find('A') + 1:filename.find('A') + 4])

    skeleton_file_name = filename_construct(setup_id, camera_id, subject_id, duplicate_id, action_id)
    
    if action_id < 61:
        frame_file = frame_path + skeleton_file_name
    else:
        frame_file = frame_path_120 + skeleton_file_name

    fivefs_concat=Image.new('RGB', (2*w*temporal_rgb_frames, 3*2*h) , (0,0,0))

    if os.path.isdir(frame_file):
        frames = os.listdir(frame_file)  
        sample_interval = len(frames) // sequence_length

        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            # frame_range =  random.sample(frames_list, 5)
            # frame_range.sort()

            if not evaluation:
                frame_range = framechosen(len(frames), temporal_rgb_frames, choosing_frame_method=choosing_frame_method)
            else:
                start_i = sample_interval//2
                flip = False
                frame_range = range(start_i, len(frames), sample_interval)

        i=0
        for frame in frame_range:
            # frame = frameimagenlist[i]
            frame_croped = ''
            frame_ = frame

            openpose_file_, frame_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path) 

            try:
                                # crop3bodypart(openpose_file, frame_file, action_id, flip=False, w=48, h=48)
                frame_croped = crop3bodypart(openpose_file_, frame_file_, action_id, flip=False, w=w, h=h)
                if frame_croped == '':
                    print(f"{filename} return empty string")
                    frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h) , (0,0,0) )
            except: 
                frame_croped = Image.new( 'RGB' , (w,temporal_rgb_frames*h), (0,0,0) )

            fivefs_concat.paste(frame_croped, (i*2*w,0))
            i+=1

    return fivefs_concat


if __name__ == '__main__':
    print("--MAIN--")