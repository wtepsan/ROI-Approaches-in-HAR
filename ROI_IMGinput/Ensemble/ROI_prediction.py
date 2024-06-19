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
import Utils.ROI_dataset as ROI_dataset 
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

# train_transform = transforms.Compose([
#         transforms.RandomRotation(10),      # rotate +/- 10 degrees
#         transforms.RandomHorizontalFlip(),  # reverse 50% of images
#         transforms.Resize(480),              ##### resize shortest side to 224 pixels
#         transforms.CenterCrop(224),         ##### crop longest side to 224 pixels at center
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

###########################################
## TRAINING AND TESTING ##
if __name__ == '__main__':
    ###########################################
    ## Get Parser for Configurations ##
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_model_name', default='ResNet18', choices={'ResNet18', 'InceptionV3', 'EfficientNetB7', 'EfficientNetB7_double', 'Densenet121'})
    parser.add_argument('--optim_name', default='SGD', choices={'SGD', 'Adam'}) 
    parser.add_argument('--loss_function_name', default='CrossEntropyLoss', choices={'CrossEntropyLoss', 'FocalLoss'}) 
    parser.add_argument('--num_epoch', default=80)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--benchmark', default='xsub', choices={'xsub', 'xview'})
    parser.add_argument('--google_pretrain', default="NO")
    parser.add_argument('--resize', default="YES")
    parser.add_argument('--evaluation', default="True", choices={'True', 'False'})
    parser.add_argument('--random_interval', default="False", choices={'True', 'False'})
    parser.add_argument('--attention', default="NONE", choices={'NONE', 'OpticalFlow'})
    parser.add_argument('--onthefly', default="NO", choices={'YES', 'NO'})
    parser.add_argument('--checkcode', default="NO", choices={'YES', 'NO'})


    # python3 _roi_predict.py --roi_model_name ResNet18 --optim_name SGD --loss_function_name CrossEntropyLoss num_epoch 10 batch_size 64 benchmark xsub

    arg = parser.parse_args()
    model_name = arg.roi_model_name 
    optim_name = arg.optim_name
    loss_function_name = arg.loss_function_name
    num_epoch = int(arg.num_epoch)
    batch_size = int(arg.batch_size)
    benchmark = arg.benchmark  
    google_pretrain = arg.google_pretrain
    resize = arg.resize
    evaluation = arg.evaluation
    random_interval = arg.random_interval
    attention = arg.attention
    onthefly = arg.onthefly
    checkcode = arg.checkcode
    
    settings = ""
    name_of_experiment = ""
    for setting, value in vars(arg).items(): 
        settings += str(setting)+"_"
        name_of_experiment += str(value)+"_"
    
    print(f"{setting}:{name_of_experiment}")

    ###########################################
    ## Set Up Configuration from Parser ##
    if arg.google_pretrain != "NO":
        google_pretrain = True
    else:
        google_pretrain = False

    if resize == "YES":
        transform = transform_resize
    else:
        transform = transform_noresize

    # Set number of class #
    number_of_class = 60

    ###########################################
    ## Seup Data for Train and Test
    #### Get Datanames for Train/Test base on Benchmark (xsub/xview)


    # PATH_FRAMEIMAGES = "/project/lt200048-video/DatasetGen/Videos2FoldersOfFrameFullSizeImages/"
    # Set only existed dataset in openpose folder  # 
    # datapath = "/project/lt200048-video/NTU_Optical/"
    datapath = "/project/lt200048-video/DatasetGen/roi_fixed_frame_X_attention_opticalflow/"
    X_train_datanames, y_train_datanames, X_test_datanames, y_test_datanames = ROI_dataset.datanames_train_test_from_roifolder(benchmark=benchmark, datapath=datapath)

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
        Transform Resize: {transform}\n\
        On the fly: {onthefly}\n\
        Attention: {attention}"
    )
    

    if arg.checkcode == "YES":
        X_train_datanames = X_train_datanames[0:64*2]
        y_train_datanames = y_train_datanames[0:64*2]
        X_test_datanames = X_test_datanames[0:64]
        y_test_datanames = y_test_datanames[0:64]

    ## Define MODEL ##
    if benchmark == 'xsub':
        if attention == "OpticalFlow":
            print(attention)
            pretrain_path = "/home/wtepsan/ROI_Attentions/__BATCH-RESULTS__/xsub_EfficientNetB7_SGD_CrossEntropyLoss_Batch-8_Google-True_Resize-NO_Attention_OpticalFlow__epoch_52tensor_0.8408.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)
        else:
            print(attention)
            pretrain_path = "/home/wtepsan/ROI_Attentions/__BATCH-RESULTS__/xsub_EfficientNetB7_SGD_CrossEntropyLoss_Batch-8_Google-True_Resize-NO_Attention_NONE__epoch_77tensor_0.8913.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    

    else:
        if attention == "OpticalFlow":
            print(attention)
            pretrain_path = "/home/wtepsan/ROI_Attentions/__BATCH-RESULTS__/xview_EfficientNetB7_SGD_CrossEntropyLoss_Batch-8_Google-True_Resize-NO__epoch_74tensor_0.9054.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)
        else:
            print(attention)
            pretrain_path = "/home/wtepsan/ROI_Attentions/__BATCH-RESULTS__/xview_EfficientNetB7_SGD_CrossEntropyLoss_Batch-8_Google-True_Resize-NO_Attention_NONE__epoch_64tensor_0.9409.pt"
            model = RGBModels.gen_model(model_name, number_of_class, google_pretrain = google_pretrain, pretrain_path=pretrain_path)    

    # model = nn.DataParallel(model, device_ids=[0,1])

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

    #### Start Training Loop #### 
    # evaluation = evaluation
    # random_interval = random_interval

    ## Test dataset for inputing into model ##
        ## Train dataset for inputing into model ##
    evaluation = False
    random_interval = random_interval 
    saveroi = False

    if model_name == 'EfficientNetB7_double':
        print(1)
        test_dataset = ROI_dataset.ROI_dataset_on_gen_roi_openpose_update_fix_frames_double(X_test_datanames, y_test_datanames, transform=transform)
    else:
        if onthefly == "YES":
            
            if attention == 'NONE':
                print(2)
                test_dataset = ROI_dataset.ROI_dataset_on_the_fly_openpose_update_fix_frames(X_test_datanames, y_test_datanames, transform=transform)
            elif attention == 'OpticalFlow':
                print(3)
                test_dataset = ROI_dataset.ROI_dataset_on_the_fly_openpose_update_fix_frames_opticalflow(X_test_datanames, y_test_datanames, transform=transform)
        else:
            if attention == 'NONE':
                print(4)
                print("Train/Test from Gen ROI with No Attention")
                test_dataset = ROI_dataset.ROI_dataset_on_gen_roi_openpose_update_fix_frames(X_test_datanames, y_test_datanames, transform=transform)
            elif attention == 'OpticalFlow':
                print(5)
                print("Train/Test from Gen ROI X OpticalFlow")
                test_dataset = ROI_dataset.ROI_dataset_on_gen_roi_openpose_update_fix_frames_opticalflow(X_test_datanames, y_test_datanames, transform=transform)        

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 16)

    ## Testing  ##
    start_testing_time = time.time()
    Test_Num_of_Correct, Test_Num_of_Data, Test_Accuracty, Test_Losses, Test_ConfusionMatrix, picklelists = RGBModels.evaluation(model, test_dataloader, loss_function, device)

    ################################################
    ################################################
    print(f'Testing correct {Test_Num_of_Correct} from {Test_Num_of_Data}\
            \nTesting Loss: {np.mean(np.array(Test_Losses, dtype=np.float32)):10.8f}\
            \nTesting Accuracy: {Test_Accuracty:10.8f}\n')
    
    print(f'Time for Testing: {time.time() - start_testing_time:.0f} seconds')
    
    name_of_experiment = benchmark+"_" + attention 
    namesaveresults = name_of_experiment + "_" +str(Test_Accuracty)

    ## SAVE PICKLE ##
    name = namesaveresults+"_"+str(Test_Accuracty)
    save_location = "/home/wtepsan/ROI_Attentions/Ensemble/pickles/"
    with open(save_location+name+'.pkl', 'wb') as fp:
        pickle.dump(picklelists, fp)

    Test_ConfusionMatrix_DF = pd.DataFrame(Test_ConfusionMatrix)
    
    confusionmatrixsavepath =  "/home/wtepsan/ROI_Attentions/Ensemble/confusionmatrix/"
    save_path_cfmatrix = confusionmatrixsavepath + model_name + "_TestWithImages_best_evaluation/"
    if not os.path.exists(save_path_cfmatrix):
        os.mkdir(save_path_cfmatrix)

    save_name_cfmatrix = save_path_cfmatrix+name+".csv"
    Test_ConfusionMatrix_DF.to_csv(save_name_cfmatrix) 
    print(f"the confusion matrix is saved at: {save_name_cfmatrix}") 

    print(f'Testing correct {Test_Num_of_Correct} from {Test_Num_of_Data}\
        \nTesting Loss: {np.mean(np.array(Test_Losses, dtype=np.float32)):10.8f}\
        \nTesting Accuracy: {Test_Accuracty:10.8f}\n')

    print("---------------------- COMPLETED ----------------------")