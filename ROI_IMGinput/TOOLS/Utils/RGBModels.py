## Define MODELS ##
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

from tqdm import tqdm

############################################
### ResNet18 ###
############################################
class ResNet18(nn.Module):
    def __init__(self, numclasses, google_pretrain = False):
        super(ResNet18, self).__init__()
        # LOAD CORE MODEL #
        self.model = models.resnet18() 
        if google_pretrain:
            print("ResNet LOADING GOOGLE PRETRAIN")
            self.model.load_state_dict(torch.load('/home/wtepsan/NTU_RGB_train/Utils/GooglePretrain/ResNet18/resnet18-f37072fd.pth'))
        else:
            print("ResNet No GOOGLE PRETRAIN")
        # Model Figuration #
        numftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(numftrs, numclasses)

    def forward(self, x):
        out = self.model(x)
        return out

############################################
### DenseNet121 ###
############################################
class DenseNet121(nn.Module):
    def __init__(self, numclasses, google_pretrain = False):
        super(DenseNet121, self).__init__()
        # LOAD CORE MODEL #
        self.model = models.densenet121()
        if google_pretrain:
            print("DenseNet121 LOADING GOOGLE PRETRAIN")
            self.model.load_state_dict(torch.load('/home/wtepsan/NTU_RGB_train/Utils/GooglePretrain/DenseNet121/densenet121-a639ec97.pth'))
        else:
            print("DenseNet121 No GOOGLE PRETRAIN")

        # Model Figuration #
        self.model.classifier = nn.Linear(self.model.classifier.in_features, numclasses)

    def forward(self, x):
        out = self.model(x)
        return out
    
############################################
### EfficientNetB7 ###
############################################
class EfficientNetB7(nn.Module):
    def __init__(self, numclasses, google_pretrain=False):
        super(EfficientNetB7, self).__init__()
        # LOAD CORE MODEL #
        self.model = models.efficientnet_b7()
        if google_pretrain:
            print("EfficientNetB7 LOADING GOOGLE PRETRAIN")
            self.model.load_state_dict(torch.load('/home/wtepsan/NTU_MMNet_NewBaseLine/Utils/GooglePretrain/EfficientNetB7/efficientnet_b7_lukemelas-dcc49843.pth'))
        else:
            print("EfficientNetB7 No GOOGLE PRETRAIN")
        
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, numclasses)

    def forward(self, x):
        out = self.model(x)
        return out
    
### EfficientNetB7 Double ###
############################################
class EfficientNetB7_double(nn.Module):
    def __init__(self, numclasses, google_pretrain=False):
        super(EfficientNetB7_double, self).__init__()
        # LOAD CORE MODEL #
        self.model1 = models.efficientnet_b7()
        self.model2 = models.efficientnet_b7()

        if google_pretrain:
            print("EfficientNetB7 LOADING GOOGLE PRETRAIN")
            self.model1.load_state_dict(torch.load('/home/wtepsan/NTU_MMNet_NewBaseLine/Utils/GooglePretrain/EfficientNetB7/efficientnet_b7_lukemelas-dcc49843.pth'))
            self.model2.load_state_dict(torch.load('/home/wtepsan/NTU_MMNet_NewBaseLine/Utils/GooglePretrain/EfficientNetB7/efficientnet_b7_lukemelas-dcc49843.pth'))
        else:
            print("EfficientNetB7 No GOOGLE PRETRAIN")
        
        self.model1.classifier[1] = nn.Linear(self.model1.classifier[1].in_features, numclasses)
        self.model2.classifier[1] = nn.Linear(self.model2.classifier[1].in_features, numclasses)

    def forward(self, x_roi, x_attention):
        out1 = self.model1(x_roi)
        out2 = self.model2(x_attention)

        return out1+out2

############################################
### InceptionV3 ###
############################################
class InceptionV3(nn.Module):
    def __init__(self, numclasses, google_pretrain=False):
        super(InceptionV3, self).__init__()
        # LOAD CORE MODEL #
        self.model = models.inception_v3()
        if google_pretrain:
            print("InceptionV3 LOADING GOOGLE PRETRAIN")
            self.model.load_state_dict(torch.load('/home/wtepsan/NTU_RGB_train/Utils/GooglePretrain/InceptionV3/inception_v3_google-0cc3c7bd.pth'))
        else:
            print("InceptionV3 No GOOGLE PRETRAIN")

        numftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(numftrs, numclasses) 

    def forward(self, x):
        out = self.model(x)
        return out


    
############################################
### Gen Model ###
############################################
def gen_model(modelname, numclasses, google_pretrain = False, pretrain_path=None):
    
    print(f"Model: {modelname}\nNum Classes: {numclasses}\n google_pretrain: {google_pretrain}\n pretrain_path: {pretrain_path}")

    if modelname == "ResNet18":
        model = ResNet18(numclasses, google_pretrain) 
        if pretrain_path is not None:
            model.load_state_dict(torch.load(pretrain_path))
        else:
            print("NO PRETRAINED")

    elif modelname == "InceptionV3":
        model = InceptionV3(numclasses, google_pretrain) 
        if pretrain_path is not None:
            model.load_state_dict(torch.load(pretrain_path))
        else:
            print("NO PRETRAINED")

    elif modelname == "EfficientNetB7":
        model = EfficientNetB7(numclasses, google_pretrain) 
        if pretrain_path is not None:
            model.load_state_dict(torch.load(pretrain_path))
        else:
            print("NO PRETRAINED")

    elif modelname == "EfficientNetB7_double":
        model = EfficientNetB7_double(numclasses, google_pretrain) 
        if pretrain_path is not None:
            model.load_state_dict(torch.load(pretrain_path))
        else:
            print("NO PRETRAINED")

    elif modelname == "DenseNet121":
        model = DenseNet121(numclasses, google_pretrain) 
        if pretrain_path is not None:
            model.load_state_dict(torch.load(pretrain_path))
        else:
            print("NO PRETRAINED")
    else:
        print(f"No {modelname}")
        model = ''

    return model


############################################
### Train/ Prediction ###
############################################
def traintest(model, dataloader, criterion, device, optimizer, mode= 'train', savemodel=False): 

    losses=[]
    yall = []
    ypredictlabelall = []
    correct = 0
    numberofdata = 0
    
    if mode == 'train': 
        for _, (X, y, names) in tqdm(enumerate(dataloader), total = len(dataloader)):
            ## Model prediction ##
            X = X.to(device)
            y = y.to(device)
            ypredict  = model(X) 
            ypredictlabel  = torch.argmax(ypredict, 1) 

            ## Measure the accuracy ##
            correct += torch.sum(ypredictlabel == y) 
            numberofdata += y.shape[0] 
            loss = criterion(ypredict, y)
            losses.append(loss.item())

            ## OPTIMIZER ##
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yall = yall + list(y.cpu().detach().numpy())
            ypredictlabelall = ypredictlabelall + list(ypredictlabel.cpu().detach().numpy())
    else: 
        with torch.no_grad():
            for _, (X, y, names) in tqdm(enumerate(dataloader), total = len(dataloader)):
                ## Model prediction ##
                X = X.to(device)
                y = y.to(device)
                ypredict  = model(X) 
                ypredictlabel  = torch.argmax(ypredict, 1) 

                ## Measure the accuracy ##
                correct += torch.sum(ypredictlabel == y) 
                numberofdata += y.shape[0] 
                loss = criterion(ypredict, y)
                loss = loss.cpu()
                losses.append(loss.item()) 

                yall = yall + list(y.cpu().detach().numpy())
                ypredictlabelall = ypredictlabelall + list(ypredictlabel.cpu().detach().numpy())


    confusionmatrix = confusion_matrix(yall, ypredictlabelall)

    accuracy = correct/numberofdata

    return correct, numberofdata, accuracy, losses, confusionmatrix

############################################
### Train/ Prediction ###
############################################
def traintest_double(model, dataloader, criterion, device, optimizer, mode= 'train', savemodel=False): 

    losses=[]
    yall = []
    ypredictlabelall = []
    correct = 0
    numberofdata = 0
    
    if mode == 'train': 
        for _, (X1, X2, y, names) in tqdm(enumerate(dataloader), total = len(dataloader)):
            ## Model prediction ##
            X1 = X1.to(device)
            X2 = X2.to(device)
            y = y.to(device)
            ypredict  = model(X1, X2) 
            ypredictlabel  = torch.argmax(ypredict, 1) 

            ## Measure the accuracy ##
            correct += torch.sum(ypredictlabel == y) 
            numberofdata += y.shape[0] 
            loss = criterion(ypredict, y)
            losses.append(loss.item())

            ## OPTIMIZER ##
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            yall = yall + list(y.cpu().detach().numpy())
            ypredictlabelall = ypredictlabelall + list(ypredictlabel.cpu().detach().numpy())
    else: 
        with torch.no_grad():
            for _, (X1, X2, y, names) in tqdm(enumerate(dataloader), total = len(dataloader)):
                ## Model prediction ##
                X1 = X1.to(device)
                X2 = X2.to(device)
                y = y.to(device)
                ypredict  = model(X1, X2) 
                ypredictlabel  = torch.argmax(ypredict, 1)

                ## Measure the accuracy ##
                correct += torch.sum(ypredictlabel == y) 
                numberofdata += y.shape[0] 
                loss = criterion(ypredict, y)
                loss = loss.cpu()
                losses.append(loss.item()) 

                yall = yall + list(y.cpu().detach().numpy())
                ypredictlabelall = ypredictlabelall + list(ypredictlabel.cpu().detach().numpy())


    confusionmatrix = confusion_matrix(yall, ypredictlabelall)

    accuracy = correct/numberofdata

    return correct, numberofdata, accuracy, losses, confusionmatrix

############################################
### EVALUATION ###
############################################
def evaluation(model, dataloader, criterion, device): 

    losses=[]
    yall = []
    ypredictlabelall = []
    picklelists = []
    correct = 0
    numberofdata = 0
    
    with torch.no_grad():
        for _, (X, y, names) in tqdm(enumerate(dataloader), total = len(dataloader)):
            ## Model prediction ##
            X = X.to(device)
            y = y.to(device)
            ypredict  = model(X) 
            # ypredicttorch = torch.cat((ypredicttorch, ypredict))
            # allnames += names 
            ypredictlabel  = torch.argmax(ypredict, 1)

            ## Measure the accuracy ##
            correct += torch.sum(ypredictlabel == y) 
            numberofdata += y.shape[0] 
            loss = criterion(ypredict, y)
            loss = loss.cpu()
            losses.append(loss.item()) 

            yall = yall + list(y.cpu().detach().numpy())
            ypredictlabelall = ypredictlabelall + list(ypredictlabel.cpu().detach().numpy())


            ### GEN PICKLE DICT ### ('name', array)
            for k in range(len(names)):
                name = names[k][0:20]
                picklelists.append((name, ypredict[k].cpu().detach().numpy()))

    confusionmatrix = confusion_matrix(yall, ypredictlabelall)
    accuracy = correct/numberofdata

    return correct, numberofdata, accuracy, losses, confusionmatrix, picklelists# 

if __name__ == '__main__':

    model_names_list = ["ResNet18", "DenseNet121", "EfficientNetB7",  "EfficientNetB7"]
    number_of_class = 60
    for model_name in model_names_list:
        model = gen_model(model_name, number_of_class)
        x = torch.rand(16,3,400,400)
        y = model(x)
        print(x.shape, y.shape)