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

# xsub_pickle_roi5 = '/project/lt200210-action/wtepsan/xsub_roi5/evaluation/xsub_roi5_tensor0.9095_best_evaluation.pkl'
xsub_pickle_roi5 = "/project/lt200210-action/wtepsan/xsub_roi5/evaluation/btwminxsub_roi5_btwmin_tensor9134_best_evaluation.pkl"

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

GCN_PICKLE_PATH = "/home/wtepsan/NTU_RGB_LAGCN/results/"

if __name__ == '__main__':

    if protocol == 'xsub':
        picklenamelist = XSUB_PICKLE_LIST
        print("XSUB XSUB XSUB XSUB XSUB XSUB XSUB XSUB XSUB 1RGB")
    else: 
        picklenamelist = XVIEW_PICKLE_LIST
        print("XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW 1RGB")
        
    for picklename in picklenamelist:
      for picklename2 in picklenamelist:        
        print("NOW ", picklename, picklename2)

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

        ## LOAD PREDICTION RESULT ##

        if protocol == 'xview':
            j_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_withdataname/epoch1_test_score_withname.pkl"
        else:
            j_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_withdataname_xsub/epoch1_test_score_withname.pkl"
        j_gcn = open(j_gcn_pickle, 'rb')
        j_gcn = list(pickle.load(j_gcn).items())
        j_gcn_dict = list2dict_prediction(j_gcn)

        # b_gcn_pickle = GCN_PICKLE_PATH+ + protocol + '/j.pkl'
        if protocol == 'xview':
            b_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            b_gcn_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_withdataname_xsub/epoch1_test_score_withname.pkl"
        b_gcn = open(b_gcn_pickle, 'rb')
        b_gcn = list(pickle.load(b_gcn).items())
        b_gcn_dict = list2dict_prediction(b_gcn)

        # jm_pickle = GCN_PICKLE_PATH+  + protocol + '/bm.pkl'
        if protocol == 'xview':
            jm_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_motion_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            jm_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/joint_motion_withdataname_xsub/epoch1_test_score_withname.pkl"
        jm = open(jm_pickle, 'rb')
        jm = list(pickle.load(jm).items())
        jm_dict = list2dict_prediction(jm)

        # bm_pickle = GCN_PICKLE_PATH+ protocol + '/jm.pkl'
        if protocol == 'xview':
            bm_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_motion_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            bm_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_motion_withdataname_xsub/epoch1_test_score_withname.pkl"
        bm = open(bm_pickle, 'rb')
        bm = list(pickle.load(bm).items())
        bm_dict = list2dict_prediction(bm)

        # bp2_pickle = GCN_PICKLE_PATH+ + protocol + '/p2.pkl'
        if protocol == 'xview':
            bp2_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p2_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            bp2_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p2_withdataname_xsub/epoch1_test_score_withname.pkl"
        bp2 = open(bp2_pickle, 'rb')
        bp2 = list(pickle.load(bp2).items())
        bp2_dict = list2dict_prediction(bp2)
        
        if protocol == 'xview':
            bp5_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p5_withdataname/epoch1_test_score_withname.pkl"
        elif  protocol == 'xsub':
            bp5_pickle  = "/project/lt200210-action/wtepsan_OLD/LAGCN/work_dir/bone_p5_withdataname_xsub/epoch1_test_score_withname.pkl"
        # bp5_pickle = GCN_PICKLE_PATH+ + protocol + '/p5.pkl'
            
        bp5 = open(bp5_pickle, 'rb')
        bp5 = list(pickle.load(bp5).items())
        bp5_dict = list2dict_prediction(bp5)


        ### RGB CHANNEL ### 
        pickle_file = picklename
        RGB = open(pickle_file, 'rb')
        RGB = pickle.load(RGB)
        RGB_dict = list2dict_prediction(RGB)


        pickle_file2 = picklename2
        RGB2 = open(pickle_file2, 'rb')
        RGB2 = pickle.load(RGB2)
        RGB2_dict = list2dict_prediction(RGB2)

        # print(r3_rgb_dict)

        right_Jm = right_Jm_rgb = right_Jm_Bm = right_Jm_Bm_rgb = 0
        right_B2 = right_B2_rgb = right_B2_B5 = right_B2_B5_rgb = 0
        right_J = right_J_rgb = right_J_B = right_J_B_rgb = 0
        right_J_B_Jm_Bm_B2_B5_rgb = 0
        right_J_B_Jm_Bm_rgb_rgb2 = 0
        total_num = 0

        # print('r1, r2', set(j_gcn_dict.keys()) - set(b_gcn_dict.keys()), set(b_gcn_dict.keys()) - set(j_gcn_dict.keys()))
        # print('r3, r1', set(r3_rgb_dict.keys()) - set(j_gcn_dict.keys()), set(j_gcn_dict.keys()) - set(b_gcn_dict.keys()))
        # print('r2, r3', set(r3_rgb_dict.keys()) - set(b_gcn_dict.keys()), set(b_gcn_dict.keys()) - set(r3_rgb_dict.keys()))

        # # print(set(r3_rgb_dict.keys()) - set(datanames))
        # # print(datanames)
        # break

        for dataname in tqdm(datanames):
          
            # print(dataname)
            # print(total_num)
            # print(dataname)

            total_num +=1
            l = label_dict[dataname]
            # print('OH YES', dataname, l)

            RGB = RGB_dict[dataname.split('.')[0]]
            RGB2 = RGB2_dict[dataname.split('.')[0]]

            j_gcn = j_gcn_dict[dataname.split('.')[0]]
            b_gcn = b_gcn_dict[dataname.split('.')[0]]

            J_B_gcn = j_gcn + b_gcn
            j_gcn_rgb = j_gcn + RGB + RGB2
            J_B_gcn_rgb = j_gcn + b_gcn + RGB+ RGB2

            j_gcn = np.argmax(j_gcn)
            j_gcn_rgb = np.argmax(j_gcn_rgb)
            J_B_gcn = np.argmax(J_B_gcn)
            J_B_gcn_rgb = np.argmax(J_B_gcn_rgb)
            right_J += int(j_gcn == int(l))
            right_J_rgb  += int(j_gcn_rgb == int(l))
            right_J_B += int(J_B_gcn == int(l))
            right_J_B_rgb += int(J_B_gcn_rgb == int(l))


            jm_gcn = jm_dict[dataname.split('.')[0]]
            bm_gcn = bm_dict[dataname.split('.')[0]]
            Jm_Bm = jm_gcn + bm_gcn
            jm_gcn_rgb = jm_gcn + RGB+ RGB2
            Jm_Bm_rgb = jm_gcn + bm_gcn + RGB+ RGB2
            jm_gcn = np.argmax(jm_gcn)
            jm_gcn_rgb = np.argmax(jm_gcn_rgb)
            Jm_Bm = np.argmax(Jm_Bm)
            Jm_Bm_rgb = np.argmax(Jm_Bm_rgb)
            right_Jm += int(jm_gcn == int(l))
            right_Jm_rgb  += int(jm_gcn_rgb == int(l))
            right_Jm_Bm += int(Jm_Bm == int(l))
            right_Jm_Bm_rgb += int(Jm_Bm_rgb == int(l))

            bp2_gcn = bp2_dict[dataname.split('.')[0]]
            bp5_gcn = bp5_dict[dataname.split('.')[0]]
            B2_B5 = bp2_gcn + bp5_gcn
            bp2_gcn_rgb = bp2_gcn + RGB+ RGB2
            B2_B5_rgb = bp2_gcn + bp5_gcn + RGB+ RGB2
            bp2_gcn = np.argmax(bp2_gcn)
            bp2_gcn_rgb = np.argmax(bp2_gcn_rgb)
            B2_B5 = np.argmax(B2_B5)
            B2_B5_rgb = np.argmax(B2_B5_rgb)
            right_B2 += int(bp2_gcn == int(l))
            right_B2_rgb  += int(bp2_gcn_rgb == int(l))
            right_B2_B5 += int(B2_B5 == int(l))
            right_B2_B5_rgb += int(B2_B5_rgb == int(l))


            J_B_Jm_Bm_rgb_rgb2 = np.argmax(j_gcn + b_gcn + jm_gcn + bm_gcn + RGB+ RGB2)
            right_J_B_Jm_Bm_rgb_rgb2 += int(J_B_Jm_Bm_rgb_rgb2==int(l))

        print("Number of test:", total_num)

        acc_11_gcn = right_J / total_num
        acc_11_gcn_rgb = right_J_rgb / total_num
        acc_gcn = right_J_B / total_num
        acc_gcn_rgb = right_J_B_rgb / total_num
        print('ST-G3D   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_gcn,acc_11_gcn_rgb,acc_gcn,acc_gcn_rgb))

        acc_11_2s = right_Jm / total_num
        acc_11_2s_rgb = right_Jm_rgb / total_num
        acc_2s = right_Jm_Bm / total_num
        acc_2s_rgb = right_Jm_Bm_rgb / total_num
        print('2s-GCN   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_2s,acc_11_2s_rgb,acc_2s,acc_2s_rgb))

        acc_11_g3d = right_B2 / total_num
        acc_11_g3d_rgb = right_B2_rgb / total_num
        acc_g3d = right_B2_B5 / total_num
        acc_g3d_rgb = right_B2_B5_rgb / total_num
        print('MS-G3D   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_g3d,acc_11_g3d_rgb,acc_g3d,acc_g3d_rgb))


        print(f'ALL SUM TOGETHER ACCURACY right_J_B_Jm_Bm_rgb_rgb2 from {total_num} samples: {right_J_B_Jm_Bm_rgb_rgb2 / total_num}')


        print("\n-----------------\n\n")