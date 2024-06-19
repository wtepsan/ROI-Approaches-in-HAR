#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:37 2019

@author: bruce
"""
import os
import json
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from PIL import Image
import numpy as np
import time
from tqdm import tqdm
import argparse


openpose_path = '/project/lt200048-video/Dataset/NTU-RGBD/Openpose/openpose_ntu60_unzip/'
openpose_cleaning_path = "/home/wtepsan/NTU_MMNet_NewBaseLine/_testcode_/cleanopenpose2/"

def skeleton_chosen_info(skeleton):
    """
    Input: skeleton
    Output: Information about skeleton  
        skeleton_frame_distance2center: list of distance from center of each skeleton 
            [distance of skeleton 1, distance of skeleton 2, ...]
        skeleton_missing_joints: missing joints of each skeleton
            [number of joints that are not detected from skeleton 1, number of joints that are not detected from skeleton 2, ...]
        boundary: 
            [[left, right, top, bottom] of skeleton 1, [left, right, top, bottom] of skeleton 2, ...]
    """

    number_of_people = len(skeleton['people'])

    center_x = 960
    center_y = 540

    skeleton_frame_distance2center = []
    skeleton_missing_joints =[]
    confidence_value_averages =[]
    center_x_list = []
    boundary = []


    # print(number_of_people)

    for people_index in range(number_of_people):
        skeleton_XY_positions = skeleton['people'][people_index]['pose_keypoints_2d']

        n_positions = int(len(skeleton_XY_positions)/3)

        positions_x = [skeleton_XY_positions[3*i] for i in range(n_positions)]
        positions_y = [skeleton_XY_positions[3*i+1] for i in range(n_positions)] 
        confidence_value = [skeleton_XY_positions[3*i+2] for i in range(n_positions)] 

        ## GET BOUNDARY OF SKELETONS
        left = min([value for value in positions_x if value!=0]) 
        right = max([value for value in positions_x if value!=0])
        top = min([value for value in positions_y if value!=0]) 
        bottom = max([value for value in positions_y if value!=0])

        ## CALCULATE CENTER ##
        center_x_skeleton = (left+right)/2
        center_y_skeleton = (top+bottom)/2
        
        center_x_list.append(center_x_skeleton)

        distance = (center_x_skeleton-center_x)**2+(center_y_skeleton-center_y)**2
        skeleton_frame_distance2center.append(distance)
        
        number_zeros = positions_x.count(0) + positions_y.count(0)
        skeleton_missing_joints.append(number_zeros)

        ## BOUNDARY 
        boundary.append([left, right, top, bottom])

        ## CONFIDENCE VALUE
        confidence_value = [value for value in confidence_value if value != 0] 
        if len(confidence_value) == 0:
            confidence_value_average = 0
        else:
             confidence_value_average = sum(confidence_value) / len(confidence_value)
        confidence_value_averages.append(confidence_value_average)

        ## Generate Dict for Output
        skeleton_chosen_info_dict = dict()
        skeleton_chosen_info_dict['skeleton_frame_distance2center'] = skeleton_frame_distance2center
        skeleton_chosen_info_dict['skeleton_missing_joints'] = skeleton_missing_joints
        skeleton_chosen_info_dict['confidence_value_averages'] = confidence_value_averages
        skeleton_chosen_info_dict['center_x_list'] = center_x_list
        skeleton_chosen_info_dict['boundary'] = boundary

    return skeleton_chosen_info_dict

####################################################################################
####################################################################################
####################################################################################
####################################################################################

### START TEST CODE ###
def clean_openpose(split_number):
    openpose_folder_list = os.listdir(openpose_path) 

    openpose_folder_list.sort()

    start = int(6000*split_number)
    end = int(6000*(split_number+1))

    openpose_list_chosen = openpose_folder_list[start: end] 

    print(f"Now cleaning openpose from {start} to {end}")
    
    for data_name in tqdm(openpose_list_chosen):

        action_id = int(data_name[data_name.find('A') + 1:data_name.find('A') + 4])

        folder = os.path.join(openpose_path, data_name)
        openpose_jsons_files_list = os.listdir(folder) 
        openpose_jsons_files_list.sort() 

        for openpose_json_file in openpose_jsons_files_list:

            openpose_json_file_path = os.path.join(folder, openpose_json_file)

            # LOAD JSON
            if openpose_json_file_path:
                with open(openpose_json_file_path, 'r') as f:
                    skeleton = json.load(f) # skeleton 

            number_of_skeleton_found = len(skeleton['people'])

            if number_of_skeleton_found>0:
                skeleton_chosen_info_dict = skeleton_chosen_info(skeleton) 

                skeleton_frame_distance2center = skeleton_chosen_info_dict['skeleton_frame_distance2center'] 
                skeleton_missing_joints = skeleton_chosen_info_dict['skeleton_missing_joints']
                confidence_value_averages = skeleton_chosen_info_dict['confidence_value_averages'] 
                center_x_list = skeleton_chosen_info_dict['center_x_list']  
                boundary = skeleton_chosen_info_dict['boundary'] 
                
                confidence_value_averages_array = np.array(confidence_value_averages)
                people_index_1st = np.argmax(confidence_value_averages_array) 
                confidence_value_averages_array[people_index_1st] = -1
                people_index_2nd = np.argmax(confidence_value_averages_array)

                # CLEAN JSON
                skeleton_clean = dict()

                skeleton_clean['people'] = [] 
    
                # Choose 2 Index if action>50
                if action_id < 50:
                    skeleton_clean['people'].append(skeleton['people'][people_index_1st]) 

                else:  
                    if center_x_list[people_index_1st] <= center_x_list[people_index_2nd]:
                        skeleton_clean['people'].append(skeleton['people'][people_index_1st])
                        skeleton_clean['people'].append(skeleton['people'][people_index_2nd])
                    else:
                        skeleton_clean['people'].append(skeleton['people'][people_index_2nd])
                        skeleton_clean['people'].append(skeleton['people'][people_index_1st])
                    

                # SAVE JSON
                skeleton_data_folder_save_path = os.path.join(openpose_cleaning_path, data_name)

                isExist = os.path.exists(skeleton_data_folder_save_path)
                if not isExist:
                    os.makedirs(skeleton_data_folder_save_path)

                skeleton_json_file_path =  os.path.join(skeleton_data_folder_save_path, openpose_json_file)
                with open(skeleton_json_file_path, "w") as outfile:
                    json.dump(skeleton_clean, outfile)

            else:
                print(f"NO SKELETON FOUND IN {data_name, openpose_json_file}")

def clean_openpose_skeleton(openpose_json_file_path, action_id):

    # LOAD JSON
    if openpose_json_file_path:
        with open(openpose_json_file_path, 'r') as f:
            skeleton = json.load(f) # skeleton 

    number_of_skeleton_found = len(skeleton['people'])

    skeleton_clean = dict()
    # skeleton_clean['version'] = 1.2 
    # skeleton_clean['data_name'] =  data_name
    # skeleton_clean['openpose_json_file'] =  openpose_json_file 
    skeleton_clean['people'] = [] 

    if number_of_skeleton_found == 0:
        print(f"NO SKELETON FOUND IN {openpose_json_file_path}")
    elif number_of_skeleton_found == 1:
        people_index = 0
        skeleton_clean['people'].append(skeleton['people'][people_index])
    else:
        skeleton_chosen_info_dict = skeleton_chosen_info(skeleton) 
        skeleton_frame_distance2center = skeleton_chosen_info_dict['skeleton_frame_distance2center'] 
        skeleton_missing_joints = skeleton_chosen_info_dict['skeleton_missing_joints']
        confidence_value_averages = skeleton_chosen_info_dict['confidence_value_averages'] 
        center_x_list = skeleton_chosen_info_dict['center_x_list']  
        boundary = skeleton_chosen_info_dict['boundary'] 
        
        confidence_value_averages_array = np.array(confidence_value_averages)
        people_index_1st = np.argmax(confidence_value_averages_array) 
        confidence_value_averages_array[people_index_1st] = -1
        people_index_2nd = np.argmax(confidence_value_averages_array)

        # CLEAN JSON 
        if action_id < 50:
            if abs(960 - center_x_list[people_index_1st]) <  abs(960 - center_x_list[people_index_2nd]):
                skeleton_clean['people'].append(skeleton['people'][people_index_1st])
            else:
                skeleton_clean['people'].append(skeleton['people'][people_index_2nd])

        else: 
            if center_x_list[people_index_1st] < center_x_list[people_index_2nd]:
                skeleton_clean['people'].append(skeleton['people'][people_index_1st])
                skeleton_clean['people'].append(skeleton['people'][people_index_2nd])
            else:
                skeleton_clean['people'].append(skeleton['people'][people_index_2nd])
                skeleton_clean['people'].append(skeleton['people'][people_index_1st]) 
    
    return skeleton_clean
        
def clean_openpose_version2(openpose_list): 
    
    print(f"Total generating is {len(openpose_list)}")

    for data_name in tqdm(openpose_list):
        print(f"Generating data name: {data_name}")

        folder = os.path.join(openpose_path, data_name)
        openpose_jsons_files_list = os.listdir(folder) 
        openpose_jsons_files_list.sort() 

        for openpose_json_file in openpose_jsons_files_list:
            action_id = int(data_name[data_name.find('A') + 1:data_name.find('A') + 4])
            openpose_json_file_path = os.path.join(folder, openpose_json_file)

            skeleton_clean = clean_openpose_skeleton(openpose_json_file_path, action_id)

            # SAVE SKELETONS
            skeleton_data_folder_save_path = os.path.join(openpose_cleaning_path, data_name)

            isExist = os.path.exists(skeleton_data_folder_save_path)
            if not isExist:
                os.makedirs(skeleton_data_folder_save_path)

            skeleton_json_file_path =  os.path.join(skeleton_data_folder_save_path, openpose_json_file)
            with open(skeleton_json_file_path, "w") as outfile:
                json.dump(skeleton_clean, outfile)


if __name__ == '__main__': 
    # parser = argparse.ArgumentParser() 
    # parser.add_argument('-b', '--begin', default='0', type=int)
    # parser.add_argument('-e', '--end', default='56880', type=int)
    # args = parser.parse_args() 

    # openpose_folder_list = os.listdir(openpose_path)
    # openpose_folder_list.sort() 

    # openpose_list = openpose_folder_list[args.begin: args.end]
    openpose_list = ["S013C003P028R001A035", "S013C002P015R002A035", "S013C002P015R002A036"]
    print("List of names to gen:")
    for name in openpose_list: 
        print(name)

    clean_openpose_version2(openpose_list) 

