###########################################
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
## Check if GPU  #
print("GPU Available: ", torch.cuda.is_available())

###########################################
## Set Random Seed #
random_seed = 123 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

###########################################
## Import Utils #
import sys        
sys.path.append('/home/wtepsan/TOOLS/')       
import Utils.ROI_dataset_UPDATE as ROI_dataset 
import Utils.Loss as Loss
import Utils.ROI_tools as ROI_tools
import Utils.RGBModels as RGBModels

###########################################
## Set Device #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################
## Set up Dataset #
transform_resize = transforms.Compose([ 
            transforms.Resize(size=(225,225)), #transforms.Resize(size=(225,45*self.temporal_rgb_frames)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

transform_noresize = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

###########################################
## TRAINING AND TESTING ##
if __name__ == '__main__':
    ###########################################
    ## Get Parser for Configurations ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_model_name', default='ResNet18', choices={'ResNet18', 'InceptionV3', 'EfficientNetB7', 'Densenet121'})
    parser.add_argument('--optim_name', default='SGD', choices={'SGD', 'Adam'}) 
    parser.add_argument('--loss_function_name', default='CrossEntropyLoss', choices={'CrossEntropyLoss', 'FocalLoss'}) 
    parser.add_argument('--num_epoch', default=80)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--benchmark', default='xsub', choices={'xsub', 'xview'})
    parser.add_argument('--google_pretrain', default="NO")
    parser.add_argument('--transform_resize', default="YES")
    parser.add_argument('--evaluation', default="True", choices={'True', 'False'})
    parser.add_argument('--random_interval', default="False", choices={'True', 'False'})
    parser.add_argument('--choosing_frame_method', default="random", choices={'fix', 'randombegin_randomlength', 'random_from_each_interval', 'random', 'min', 'max', 'mix','btwmin', 'btwmax'})
    parser.add_argument('--imginput', default="fullbody", choices={'fullbody', 'body2parts', 'body3parts', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7'})
    parser.add_argument('--attention', default="NONE", choices={'NONE', 'OpticalFlow'})
    parser.add_argument('--onthefly', default="NO", choices={'YES', 'NO'})
    parser.add_argument('--checkcode', default="NO", choices={'YES', 'NO'})

    arg = parser.parse_args()
    model_name = arg.roi_model_name 
    optim_name = arg.optim_name
    loss_function_name = arg.loss_function_name
    num_epoch = int(arg.num_epoch)
    batch_size = int(arg.batch_size)
    benchmark = arg.benchmark  
    google_pretrain = arg.google_pretrain
    transform_resize = arg.transform_resize
    evaluation = arg.evaluation
    random_interval = arg.random_interval
    choosing_frame_method = arg.choosing_frame_method
    imginput = arg.imginput
    attention = arg.attention
    onthefly = arg.onthefly
    checkcode = arg.checkcode

    ###########################################
    ## Set Up Configuration from Parser ##
    if arg.google_pretrain != "NO":
        google_pretrain = True
    else:
        google_pretrain = False

    if transform_resize == "YES":
        transform = transform_resize
    else:
        transform = transform_noresize

    # Set number of class #
    number_of_class = 60

    ###########################################
    ## Seup Data for Train and Test
    #### Get Datanames for Train/Test base on Benchmark (xsub/xview)


    PATH_FRAMEIMAGES = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"

    # Set only existed dataset in openpose folder  # 
    # datapath = "/project/lt200048-video/NTU_Optical/"
    datapath = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_opticalflow/"

    X_train_datanames, y_train_datanames, X_test_datanames, y_test_datanames = ROI_dataset.datanames_train_test_from_roifolder(benchmark=benchmark, datapath=PATH_FRAMEIMAGES)

    print(f"Device: {device}\n\
        model_name: {model_name}\n\
        optim_name: {optim_name}\n\
        loss_function_name: {loss_function_name}\n\
        num_epoch: {num_epoch}\n\
        batch_size: {batch_size}\n\
        benchmark :{benchmark}\n\
        train data number: {len(y_train_datanames)}\n\
        test data number: {len(y_test_datanames)}\n\
        google_pretrain: {google_pretrain}\n\
        Transform Resize: {transform_resize}\n\
        Frame Chosen Method: {choosing_frame_method}\n\
        Image INPUT: {imginput}\n\
        On the fly: {onthefly}\n\
        Attention: {attention}"
    )
    
    ################# SET STRING FOR SAVE ##################
    name_of_experiment = benchmark + "_" \
        +"_Input_" + imginput \
        + model_name+"_" \
        + optim_name+"_" \
        +loss_function_name \
        +"_Batch-"+ str(batch_size) \
        + "_Google-" + str(google_pretrain) \
        + "_Resize-" +  transform_resize
    
    MAINFOLDER = "/project/lt200210-action/wtepsan/"
    RESULT_SAVED_FOLDER = MAINFOLDER + benchmark + "_" + imginput
    if not os.path.exists(RESULT_SAVED_FOLDER):
        os.makedirs(RESULT_SAVED_FOLDER)
    #######################################################
    
    if arg.checkcode == "YES":
        num_epoch = 1
        X_train_datanames.sort()
        y_train_datanames.sort()
        X_test_datanames.sort()
        y_test_datanames.sort()
        X_train_datanames = X_train_datanames[0:64*2]
        y_train_datanames = y_train_datanames[0:64*2]
        X_test_datanames = X_test_datanames[0:64]
        y_test_datanames = y_test_datanames[0:64]
        saveroi = True
    else:
        saveroi = False
        print("NO SAVE IMAGES")


    ## DEFINE MODEL ##
    if benchmark == 'xsub': 
        # choices={'fullbody', 'body2parts', 'body3parts', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7'})
        if imginput=='fullbody':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_fullbody/xsub_fullbody_0.9130.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        if imginput=='body2parts':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_body2parts/EfficientNetB7/xsub__Input_body2parts_0.9083.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='body3parts':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_body3parts/xsub_body3parts_0.9001.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi3':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_roi3/EfficientNetB7/xsub__Input_roi3_0.8905.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi4':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_roi4/EfficientNetB7/xsub__Input_roi4_0.8869.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi5':
            if choosing_frame_method == 'btwmin':
                print("I AM HERE FORE THE TESTTTT")
                pretrain_path = "/project/lt200210-action/wtepsan/xsub_roi5/xsub__Input_roi5_FrameSelection_btwmin_epoch_54tensor_9171.pt"
            else: 
                pretrain_path = "/project/lt200210-action/wtepsan/xsub_roi5/xsub_roi5_0.9127.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi6':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_roi6/EfficientNetB7/xsub__Input_roi6_0.8907.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi7':
            pretrain_path = "/project/lt200210-action/wtepsan/xsub_roi7/EfficientNetB7/xsub__Input_roi7_0.9159.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        else:
            print(f"NO INPUT IMATE {imginput}")
    elif benchmark == 'xview': 
        # choices={'fullbody', 'body2parts', 'body3parts', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7'})
        if imginput=='fullbody':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_fullbody/EfficientNetB7/xview__Input_fullbody_0.9528.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        if imginput=='body2parts':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_body2parts/EfficientNetB7/xview__Input_body2parts_0.9520.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='body3parts':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_body3parts/EfficientNetB7/xview__Input_body3parts_0.9491.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi3':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_roi3/EfficientNetB7/xview__Input_roi3_0.9455.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi4':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_roi4/EfficientNetB7/xview__Input_roi4_0.9326.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi5':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_roi5/EfficientNetB7/xview__Input_roi5_0.9576.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi6':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_roi6/EfficientNetB7/xview__Input_roi6_0.9315.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        elif imginput=='roi7':
            pretrain_path = "/project/lt200210-action/wtepsan/xview_roi7/EfficientNetB7/xview__Input_roi7_0.9600.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    
        else:
            print(f"NO INPUT IMATE {imginput}")

    ## Define Optimizer ##
    if optim_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    elif optim_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    ## Define Loss ##
    if loss_function_name == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
    elif loss_function_name == "FocalLoss":
        loss_function = Loss.FocalLoss()

    ## Send model to device ##
    model = model.to(device)
    loss_function = loss_function.to(device) 

    ## DATASET ##
    random_interval = random_interval
    test_dataset = ROI_dataset.ROI_dataset_on_the_fly_openpose_genIMG(X_test_datanames, y_test_datanames, transform=transform, evaluation=True, choosing_frame_method=choosing_frame_method, imginput=imginput, saveroi=saveroi) # datanames, datanamelabels, transform=None, evaluation =True, random_interval=False,  random_flip=False, saveroi=False, temporal_rgb_frames=5, choosing_frame_method='fix', joints=[0, 4, 7, 10, 13], w=48, h=48)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 64)

    ## Testing  ##
    start_testing_time = time.time()
    Test_Num_of_Correct, Test_Num_of_Data, Test_Accuracty, Test_Losses, Test_ConfusionMatrix, picklelists = RGBModels.evaluation(model, test_dataloader, loss_function, device)

    MAINFOLDER = "/project/lt200210-action/wtepsan/"
    RESULT_SAVED_FOLDER = MAINFOLDER + benchmark + "_" + imginput + "/evaluation/" + "/" + choosing_frame_method 
    if not os.path.exists(RESULT_SAVED_FOLDER):
        os.makedirs(RESULT_SAVED_FOLDER)

    ################################################
    ################################################
    print(f'Testing correct {Test_Num_of_Correct} from {Test_Num_of_Data}\
            \nTesting Loss: {np.mean(np.array(Test_Losses, dtype=np.float32)):10.8f}\
            \nTesting Accuracy: {Test_Accuracty:10.8f}\n')
    
    print(f'Time for Testing: {time.time() - start_testing_time:.0f} seconds')
    
    name_of_experiment = benchmark+ "_" + imginput +"_"+ choosing_frame_method
    namesaveresults = name_of_experiment + "_" +str(Test_Accuracty)


    ## SAVE PICKLE ## 
    pickle_save_location = RESULT_SAVED_FOLDER + namesaveresults + '_best_evaluation.pkl'
    with open(pickle_save_location, 'wb') as fp:
        pickle.dump(picklelists, fp)
    print(f"PICKLE is saved at: {pickle_save_location}") 
    
    ## SAVE CONFUSION MATRIX ##
    confMatrix_save_location = RESULT_SAVED_FOLDER + namesaveresults + "_best_evaluation.csv"
    Test_ConfusionMatrix_DF = pd.DataFrame(Test_ConfusionMatrix)
    Test_ConfusionMatrix_DF.to_csv(confMatrix_save_location) 
    print(f"the confusion matrix is saved at: {confMatrix_save_location}") 


    ## PRINT RESUTLS ##
    print(f'Testing correct {Test_Num_of_Correct} from {Test_Num_of_Data}\
        \nTesting Loss: {np.mean(np.array(Test_Losses, dtype=np.float32)):10.8f}\
        \nTesting Accuracy: {Test_Accuracty:10.8f}\n')

    print("---------------------- COMPLETED ----------------------")