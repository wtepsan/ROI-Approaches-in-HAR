#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:32:11 2020

@author: bruce
"""

import argparse
import pickle
import os
import sys
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--protocols', default='xsub', choices={'xsub', 'xview'},
                    help='the work folder for storing results')
parser.add_argument('--method', default='sum', choices={'sum', 'multiply'})
arg = parser.parse_args()

protocol = arg.protocols
method = arg.method

# print(f'model_name:{model_name} \nattention: {attention} \nbenchmark:{benchmark} \nprotocol: {protocol} \nalpha: {alpha}')

def list2dict_label(list_of_tuples):
    datanames = list_of_tuples[0] 

    list_of_tuples_dict = dict()
    for i in range(len(datanames)):
        list_of_tuples_dict[list_of_tuples[0,i]] = list_of_tuples[1, i]

    return list_of_tuples_dict

def list2dict_prediction(list_of_prediction): 

    list_of_prediction_dict = dict()
    for dataname, prediction in list_of_prediction: 
        list_of_prediction_dict[dataname] = prediction
         
    return list_of_prediction_dict


### --imginput', default="fullbody", choices={'fullbody', 'body2parts', 'body3parts', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7'})
xsub_pickle_fullbody = '/project/lt200210-action/wtepsan/xsub_fullbody/evaluation/xsub_fullbody_tensor0.9127_best_evaluation.pkl'
xsub_pickle_body2parts = '/project/lt200210-action/wtepsan/xsub_body2parts/evaluation/xsub_body2parts_tensor9050_best_evaluation.pkl'
xsub_pickle_body3parts = "/project/lt200210-action/wtepsan/xsub_body3parts/evaluation/xsub_body3parts_tensor0.8961_best_evaluation.pkl"
xsub_pickle_roi3 = '/project/lt200210-action/wtepsan/xsub_roi3/evaluation/xsub_roi3_tensor0.8874_best_evaluation.pkl'
xsub_pickle_roi4 = '/project/lt200210-action/wtepsan/xsub_roi4/evaluation/xsub_roi4_tensor0.8847_best_evaluation.pkl'
xsub_pickle_roi5 = '/project/lt200210-action/wtepsan/xsub_roi5/evaluation/xsub_roi5_tensor0.9095_best_evaluation.pkl'
xsub_pickle_roi6 = '/project/lt200210-action/wtepsan/xsub_roi6/evaluation/xsub_roi6_tensor8901_best_evaluation.pkl'
xsub_pickle_roi7 = '/project/lt200210-action/wtepsan/xsub_roi7/evaluation/xsub_roi7_tensor9150_best_evaluation.pkl'

xview_pickle_fullbody = '/project/lt200210-action/wtepsan/xview_fullbody/evaluation/xview_fullbody_tensor0.9525_best_evaluation.pkl'
xview_pickle_body2parts = '/project/lt200210-action/wtepsan/xview_body2parts/evaluation/xview_body2parts_tensor0.9507_best_evaluation.pkl'
xview_pickle_body3parts = '/project/lt200210-action/wtepsan/xview_body3parts/evaluation/xview_body3parts_tensor0.9471_best_evaluation.pkl'
xview_pickle_roi3 = '/project/lt200210-action/wtepsan/xview_roi3/evaluation/xview_roi3_tensor0.9427_best_evaluation.pkl'
xview_pickle_roi4 = '/project/lt200210-action/wtepsan/xview_roi4/evaluation/xview_roi4_tensor0.9315_best_evaluation.pkl'
xview_pickle_roi5 = '/project/lt200210-action/wtepsan/xview_roi5/evaluation/xview_roi5_tensor0.9583_best_evaluation.pkl'
xview_pickle_roi6 = '/project/lt200210-action/wtepsan/xview_roi6/evaluation/xview_roi6_tensor0.9294_best_evaluation.pkl'
xview_pickle_roi7 = '/project/lt200210-action/wtepsan/xview_roi7/evaluation/xview_roi7_tensor0.9582_best_evaluation.pkl'

XSUB_PICKLE_LIST = [xsub_pickle_fullbody, xsub_pickle_body2parts, xsub_pickle_body3parts, xsub_pickle_roi3, xsub_pickle_roi4, xsub_pickle_roi5, xsub_pickle_roi6, xsub_pickle_roi7]
XVIEW_PICKLE_LIST = [xview_pickle_fullbody, xview_pickle_body2parts, xview_pickle_body3parts, xview_pickle_roi3, xview_pickle_roi4, xview_pickle_roi5, xview_pickle_roi6, xview_pickle_roi7]

GCN_PICKLE_PATH = "/home/wtepsan/NTU_MMNet_NewBaseLine/"

if __name__ == '__main__':

    if protocol == 'xsub':
        picklenamelist = XSUB_PICKLE_LIST
        print("XSUB XSUB XSUB XSUB XSUB XSUB XSUB XSUB XSUB 2RGB")
    else: 
        picklenamelist = XVIEW_PICKLE_LIST
        print("XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW 2RGB")
        
    for picklename in picklenamelist:
      for picklename2 in picklenamelist:
        print("NOW ", picklename, "\n", picklename2)

        if picklename.find('xsub') != -1:
            protocol = 'xsub'
            print(protocol)
        else:
            protocol = 'xview'
            print(protocol)
        label_pickle = "/home/wtepsan/NTU_MMNet_NewBaseLine/data/ntu_st_gcn/" + protocol + '/val_label.pkl'
        label = open(label_pickle, 'rb')
        label = np.array(pickle.load(label))
 
        datanames =  label[0]
        label_dict = list2dict_label(label)

        label_g3d_pickle = "/home/wtepsan/NTU_MMNet_NewBaseLine/data/ntu_st_gcn/" + protocol + '/val_label.pkl'
        label_g3d = open(label_g3d_pickle, 'rb')
        label_g3d = np.array(pickle.load(label_g3d))
        label_g3d_dict = list2dict_label(label_g3d)

        if protocol == 'xview':
            r1_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_withdataname/epoch1_test_score_withname.pkl"
        else:
            r1_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_withdataname_xsub/epoch1_test_score_withname.pkl"
        r1_gcn = open(r1_gcn_pickle, 'rb')
        r1_gcn = list(pickle.load(r1_gcn).items())
        r1_gcn_dict = list2dict_prediction(r1_gcn)

        if protocol == 'xview':
            r2_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            r2_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_withdataname_xsub/epoch1_test_score_withname.pkl"
        r2_gcn = open(r2_gcn_pickle, 'rb')
        r2_gcn = list(pickle.load(r2_gcn).items())
        r2_gcn_dict = list2dict_prediction(r2_gcn)

        if protocol == 'xview':
            r1_2s_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_motion_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            r1_2s_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_motion_withdataname_xsub/epoch1_test_score_withname.pkl"
        r1_2s = open(r1_2s_pickle, 'rb')
        r1_2s = list(pickle.load(r1_2s).items())
        r1_2s_dict = list2dict_prediction(r1_2s)

        if protocol == 'xview':
            r2_2s_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_motion_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            r2_2s_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_motion_withdataname_xsub/epoch1_test_score_withname.pkl"
        r2_2s = open(r2_2s_pickle, 'rb')
        r2_2s = list(pickle.load(r2_2s).items())
        r2_2s_dict = list2dict_prediction(r2_2s)

        if protocol == 'xview':
            r1_g3d_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p2_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            r1_g3d_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p2_withdataname_xsub/epoch1_test_score_withname.pkl"
        r1_g3d = open(r1_g3d_pickle, 'rb')
        r1_g3d = list(pickle.load(r1_g3d).items())
        r1_g3d_dict = list2dict_prediction(r1_g3d)
        
        if protocol == 'xview':
            r2_g3d_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p5_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            r2_g3d_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p5_withdataname_xsub/epoch1_test_score_withname.pkl"
        # r2_g3d_pickle = GCN_PICKLE_PATH+ + protocol + '/p5.pkl'
            
        r2_g3d = open(r2_g3d_pickle, 'rb')
        r2_g3d = list(pickle.load(r2_g3d).items())
        r2_g3d_dict = list2dict_prediction(r2_g3d)

        pickle_file = picklename
        r3_rgb = open(pickle_file, 'rb')
        r3_rgb = pickle.load(r3_rgb)
        r3_rgb_dict = list2dict_prediction(r3_rgb)

        pickle_file2 = picklename2
        r3_rgb2 = open(pickle_file2, 'rb')
        r3_rgb2 = pickle.load(r3_rgb2)
        r3_rgb2_dict = list2dict_prediction(r3_rgb2)

        # print(r3_rgb_dict)

        right_num_11_2s = right_num_11_2s_rgb = right_num_2s = right_num_2s_rgb = 0
        right_num_11_g3d = right_num_11_g3d_rgb = right_num_g3d = right_num_g3d_rgb = 0
        right_num_11_gcn = right_num_11_gcn_rgb = right_num_gcn = right_num_gcn_rgb = 0
        total_num = 0

        # print(set(datanames) - set(r3_rgb_dict.keys()))

        for dataname in tqdm(datanames):
            # print(dataname)
            # print(total_num)
            # print(dataname)

            total_num +=1
            l = label_dict[dataname]
            r33_rgb = r3_rgb_dict[dataname.split('.')[0]]
            r33_rgb2 = r3_rgb2_dict[dataname.split('.')[0]]

            r11_gcn = r1_gcn_dict[dataname.split('.')[0]]
            r22_gcn = r2_gcn_dict[dataname.split('.')[0]]

            r_gcn = r11_gcn + r22_gcn
            r11_gcn_rgb = r11_gcn + r33_rgb + r33_rgb2
            r_gcn_rgb = r11_gcn + r22_gcn + r33_rgb + r33_rgb2

            r11_gcn = np.argmax(r11_gcn)
            r11_gcn_rgb = np.argmax(r11_gcn_rgb)
            r_gcn = np.argmax(r_gcn)
            r_gcn_rgb = np.argmax(r_gcn_rgb)

            right_num_11_gcn += int(r11_gcn == int(l))
            right_num_11_gcn_rgb  += int(r11_gcn_rgb == int(l))
            right_num_gcn += int(r_gcn == int(l))
            right_num_gcn_rgb += int(r_gcn_rgb == int(l))

            r11_2s = r1_2s_dict[dataname.split('.')[0]]
            r22_2s = r2_2s_dict[dataname.split('.')[0]]
            r_2s = r11_2s + r22_2s
            r11_2s_rgb = r11_2s + r33_rgb + r33_rgb2
            r_2s_rgb = r11_2s + r22_2s + r33_rgb + r33_rgb2
            r11_2s = np.argmax(r11_2s)
            r11_2s_rgb = np.argmax(r11_2s_rgb)
            r_2s = np.argmax(r_2s)
            r_2s_rgb = np.argmax(r_2s_rgb)
            right_num_11_2s += int(r11_2s == int(l))
            right_num_11_2s_rgb  += int(r11_2s_rgb == int(l))
            right_num_2s += int(r_2s == int(l))
            right_num_2s_rgb += int(r_2s_rgb == int(l))

            r11_g3d = r1_g3d_dict[dataname.split('.')[0]]
            r22_g3d = r2_g3d_dict[dataname.split('.')[0]]
            r_g3d = r11_g3d + r22_g3d
            r11_g3d_rgb = r11_g3d + r33_rgb + r33_rgb2
            r_g3d_rgb = r11_g3d + r22_g3d + r33_rgb + r33_rgb2
            r11_g3d = np.argmax(r11_g3d)
            r11_g3d_rgb = np.argmax(r11_g3d_rgb)
            r_g3d = np.argmax(r_g3d)
            r_g3d_rgb = np.argmax(r_g3d_rgb)
            right_num_11_g3d += int(r11_g3d == int(l))
            right_num_11_g3d_rgb  += int(r11_g3d_rgb == int(l))
            right_num_g3d += int(r_g3d == int(l))
            right_num_g3d_rgb += int(r_g3d_rgb == int(l))

        print("Number of test:", total_num)

        acc_11_gcn = right_num_11_gcn / total_num
        acc_11_gcn_rgb = right_num_11_gcn_rgb / total_num
        acc_gcn = right_num_gcn / total_num
        acc_gcn_rgb = right_num_gcn_rgb / total_num
        print('ST-G3D   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_gcn,acc_11_gcn_rgb,acc_gcn,acc_gcn_rgb))

        acc_11_2s = right_num_11_2s / total_num
        acc_11_2s_rgb = right_num_11_2s_rgb / total_num
        acc_2s = right_num_2s / total_num
        acc_2s_rgb = right_num_2s_rgb / total_num
        print('2s-GCN   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_2s,acc_11_2s_rgb,acc_2s,acc_2s_rgb))

        acc_11_g3d = right_num_11_g3d / total_num
        acc_11_g3d_rgb = right_num_11_g3d_rgb / total_num
        acc_g3d = right_num_g3d / total_num
        acc_g3d_rgb = right_num_g3d_rgb / total_num
        print('MS-G3D   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_g3d,acc_11_g3d_rgb,acc_g3d,acc_g3d_rgb))

        print("\n-----------------\n\n")