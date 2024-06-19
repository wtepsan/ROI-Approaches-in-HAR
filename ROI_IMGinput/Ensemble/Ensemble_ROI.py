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
        list_of_tuples_dict[list_of_tuples[0,i].split(".")[0]] = list_of_tuples[1, i]

    return list_of_tuples_dict

def list2dict_prediction(list_of_prediction): 

    list_of_prediction_dict = dict()
    for dataname, prediction in list_of_prediction: 
        dataname = dataname.split(".")[0]
        list_of_prediction_dict[dataname] = prediction
         
    return list_of_prediction_dict

def pickle2dict(predictionpickle):
    list_of_prediction_dict = dict()
    datanames = []
    for dataname, prediction in predictionpickle: 
        datanames.append(dataname)
        list_of_prediction_dict[dataname] = prediction

    return datanames, list_of_prediction_dict

def datanames2dictwithlabel(datanames):
    label_dict = dict() 
    for dataname in datanames:
        label = int(dataname[dataname.find("A")+1: dataname.find("A")+4])-1
        label_dict[dataname] = label
    
    # print(label_dict)
    return label_dict

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

if __name__ == '__main__':
    if protocol == 'xsub':
        print("XSUB XSUB XSUB XSUB XSUB XSUB XSUB XSUB XSUB")
        PICKLELIST = XSUB_PICKLE_LIST
    else:
        print("XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW XVIEW")
        PICKLELIST = XVIEW_PICKLE_LIST
    
    for pickle1_location in PICKLELIST:
            for pickle2_location in PICKLELIST: 
                print(pickle1_location, pickle2_location)
            
                pickle1 = open(pickle1_location, 'rb')
                pickle2 = open(pickle2_location, 'rb')

                prediction1 = pickle.load(pickle1)
                datanames, prediction1 = pickle2dict(prediction1) 

                prediction2 = pickle.load(pickle2)
                _, prediction2= pickle2dict(prediction2) 

                label_dict = datanames2dictwithlabel(datanames)

                total_num = 0
                ensemble_sum_total_correct = 0
                ensemble_multiply_total_correct = 0

                for dataname in tqdm(datanames): 

                    total_num +=1
                    l = label_dict[dataname]
                    roi_prediction = prediction1[dataname]
                    roiXopticalflow_prediction = prediction2[dataname] 

                    ensemble_sum = roi_prediction+roiXopticalflow_prediction
                    ensemble_sum_prediction = np.argmax(ensemble_sum)
                    ensemble_sum_total_correct += int(ensemble_sum_prediction == int(l))

                    ensemble_multiply = roi_prediction * roiXopticalflow_prediction
                    ensemble_multiply_prediction = np.argmax(ensemble_multiply)
                    ensemble_multiply_total_correct += int(ensemble_multiply_prediction == int(l))
            
                print("Total number of test:", total_num)
                print(f"Accuracy of SUM {ensemble_sum_total_correct/total_num} and MULTIPLY {ensemble_multiply_total_correct/total_num}")