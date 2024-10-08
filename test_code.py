#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import re
import numpy as np
import os
import random
import pickle
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve, auc
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    
class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):
    
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    return cost

def init_weight(net,restore):
    net.load_state_dict(torch.load(restore))
    print("Restore model from: {}".format(restore))
    return net

def save_model(net, filename):
    """Save trained model."""
    torch.save(net.state_dict(),filename)
    print("finished save pretrained model to: {}".format(filename))
    
class add_attn(nn.Module):
    def __init__(self, x_channels, g_channels=256):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv1d(x_channels,
                      x_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm1d(x_channels))
        self.theta = nn.Conv1d(x_channels,
                               x_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)
        self.phi = nn.Conv1d(g_channels,
                             x_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.psi = nn.Conv1d(x_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        
        self.upsampling1=nn.Upsample(size=32,mode='nearest')
        self.upsampling2=nn.Upsample(size=64,mode='nearest')
        self.sigmoid1=nn.Sigmoid()
        self.relu1=nn.ReLU()

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        
        phi_g = self.upsampling1(self.phi(g))

        f = self.relu1(theta_x + phi_g)

        sigm_psi_f = self.sigmoid1(self.psi(f))
        sigm_psi_f = self.upsampling2(sigm_psi_f)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y

def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,padding=(size-stride)//2, bias=False)

class BasicBlock1d(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.inplanes=inplanes
        self.planes=planes
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU()
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.res = res
        self.se = nn.Sequential(
            nn.Linear(planes, planes//4),
            nn.ReLU(),
            nn.Linear(planes//4, planes),
            nn.Sigmoid())
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes))
        

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)   
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out) 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out) 

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            if self.inplanes!=self.planes:
                residual = self.shortcut(residual)
            out += residual
        #out = self.relu(out)
        b, c, _ = out.size()
        y = nn.AdaptiveAvgPool1d(1)(out)
        y = y.view(b,c)
        y = self.se(y).view(b, c, 1)
        y = out * y.expand_as(out)
        
        out = y + residual
        return out

class featureextractor(nn.Module):
    def __init__(self):
        super(featureextractor, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=60, stride=2, padding=29,bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=20, stride=2, padding=9,bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        layers = []
        layers.append(BasicBlock1d(32, 32, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(32, 32, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(BasicBlock1d(32, 64, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(64, 64, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(BasicBlock1d(64, 128, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(128, 128, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        layers.append(BasicBlock1d(128, 256, stride=1, downsample=None, size=5, res=True))
        layers.append(BasicBlock1d(256, 256, stride=1, downsample=None, size=5, res=True))
        layers.append(nn.MaxPool1d(2))
        self.layers1=nn.Sequential(*layers)
        
        self.center = nn.Sequential(
                nn.Conv1d(
                    256,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(
                    512,
                    512,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU())
        
        self.gating = nn.Sequential(
                nn.Conv1d(
                    512,
                    256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        self.attn_1 = add_attn(x_channels=256)
        
        self.lstm1 = nn.LSTM(input_size=256,hidden_size=256,batch_first=True,bidirectional=True)
        self.sigmoid=nn.Sigmoid()
        self.maxpool = nn.MaxPool1d(2)
    
    def forward(self, x0):
        batch_size = x0.size()[0]
        x0 = self.conv1(x0)     
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.conv2(x0)
        
        conv_out=self.layers1(x0)
        center_out = self.center(self.maxpool(conv_out))
        gating_out = self.gating(center_out)
    
        attn_1_out = self.attn_1(conv_out, gating_out)
            
        attn_1_out = attn_1_out.permute(0, 2, 1)  
        lstm_output, (final_hidden_state, final_cell_state) = self.lstm1(attn_1_out)
        lstm_output = lstm_output.permute(0, 2, 1)
            
        return lstm_output.contiguous().view(batch_size,512*64)

class ECGNet_transfer(nn.Module):
    def __init__(self, method_name="LMMD"):
        super(ECGNet_transfer, self).__init__()
        self.method_name=method_name
        self.feature_layers = featureextractor()
        self.bottle = nn.Sequential(
            nn.Linear(512*64, 64),
            nn.ReLU())
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(512*64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        
        self.fc = nn.Linear(64, 9)
        self.grl_layer = AdversarialLayer(high=1.0)
        
    def forward(self, source, target, s_label, alpha):
        source=self.feature_layers(source)
        source1 = self.bottle(source)
        s_pred = self.fc(source1)
            
        reverse_feature = self.grl_layer(source)
        loss_transfer = self.domain_classifier(reverse_feature)
            
        return s_pred, loss_transfer,source
    
    def predict(self, x):
        x = self.feature_layers(x)
        x1 = self.bottle(x)
        return self.fc(x1),x1
    
#'窦性心律','窦性心动过速','窦性心动过缓','房性早搏','心房颤动','室性早搏','窦性心律不齐','室上性心动过速','室性心动过速'
#CPSC #N:0, A:1, V:2, AF:3
#CODE #0:Normal 1:ST 2:SB 3:AF
def process_cpscresult(y_data):
    y_data1=np.zeros((len(y_data),9))
    for i in range(len(y_data)):
        if y_data[i][0]==1:
            y_data1[i][0]=1
        
        if y_data[i][1]==1:
            y_data1[i][3]=1
            
        if y_data[i][2]==1:
            y_data1[i][5]=1
            
        if y_data[i][3]==1:
            y_data1[i][4]=1
    return y_data1

def process_coderesult(y_data):
    y_data1=np.zeros((len(y_data),9))
    for i in range(len(y_data)):
        if y_data[i][0]==1:
            y_data1[i][0]=1
        
        if y_data[i][1]==1:
            y_data1[i][1]=1
            
        if y_data[i][2]==1:
            y_data1[i][2]=1
            
        if y_data[i][3]==1:
            y_data1[i][4]=1
    return y_data1

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, train_flag=True, factor='Hospital'):
        if factor=="Hospital":
            with h5py.File("../data/data_trainall1.hdf5", 'r') as f:
                x_data = (np.array(f['data']))*0.8317
                y_data = np.array(f['label'])
        elif factor=="CPSC":
            with open("../data/cspc2018/data_original.pkl",  'rb') as inp:
                x_data = pickle.load(inp)
                y_data1 = pickle.load(inp)
            y_data=process_cpscresult(y_data1)
        else:
            with h5py.File("../data/CODE/data_combineall.hdf5", 'r') as f:
                x_data = (np.array(f['data']))
                y_data1 = np.array(f['label'])
                otherdata = np.array(f['otherdata'])
            x_data=x_data.transpose((0,2,1))
            y_data=process_coderesult(y_data1)
        
        x_train,x_test1,y_train,y_test1 = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
        x_valid,x_test,y_valid,y_test = train_test_split(x_test1,y_test1,test_size=0.5,random_state=1)
        del x_data,y_data,x_test1,y_test1
        
        if train_flag:
            self.dataset=torch.from_numpy(x_train)
            self.labels=torch.from_numpy(y_train)
            self.n_data = len(x_train)
        else:
            self.dataset=torch.from_numpy(x_valid)
            self.labels=torch.from_numpy(y_valid)
            self.n_data = len(x_valid)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return self.n_data


class GetLoader1(torch.utils.data.Dataset):
    def __init__(self,factor='Hospital'):
        if factor=="Hospital":
            with h5py.File("../data/data_trainall1.hdf5", 'r') as f:
                x_data = (np.array(f['data']))*0.8317
                y_data = np.array(f['label'])
        elif factor=="CPSC":
            with open("../data/cspc2018/data_original.pkl",  'rb') as inp:
                x_data = pickle.load(inp)
                y_data1 = pickle.load(inp)
            y_data=process_cpscresult(y_data1)
        else:
            with h5py.File("../data/CODE/data_combineall.hdf5", 'r') as f:
                x_data = (np.array(f['data']))
                y_data1 = np.array(f['label'])
                otherdata = np.array(f['otherdata'])
            x_data=x_data.transpose((0,2,1))
            y_data=process_coderesult(y_data1)
        
        x_train,x_test1,y_train,y_test1 = train_test_split(x_data,y_data,test_size=0.2,random_state=1)
        x_valid,x_test,y_valid,y_test = train_test_split(x_test1,y_test1,test_size=0.5,random_state=1)
        del x_data,y_data,x_test1,y_test1
        
        self.dataset=torch.from_numpy(x_test)
        self.labels=torch.from_numpy(y_test)
        self.n_data = len(x_test)
        
    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return self.n_data

def load_data(batch_size=128,factor='Hospital',factor1="CPSC"):
    dataset1 = GetLoader(train_flag=True,factor=factor)
    loader_src = torch.utils.data.DataLoader(dataset=dataset1,batch_size=batch_size,shuffle=True,drop_last=True)
    
    dataset2 = GetLoader(train_flag=True,factor=factor1)
    loader_tar = torch.utils.data.DataLoader(dataset=dataset2,batch_size=batch_size,shuffle=True,drop_last=True)
    
    dataset3 = GetLoader1(factor=factor1)
    loader_tar_test = torch.utils.data.DataLoader(dataset=dataset3,batch_size=batch_size,shuffle=False)
    
    return loader_src, loader_tar, loader_tar_test

def round_compute(prev):
    prev1=np.zeros((len(prev),len(prev[0])))
    for i in range(len(prev)):
        for j in range(len(prev[i])):
            if prev[i][j]>=0.5:
                prev1[i][j]=1
            else:
                prev1[i][j]=0
    return prev1

def result_process(y_data):
    for i in range(len(y_data)):
        if np.sum(y_data[i][1:])>0 and y_data[i][0]==1:
            y_data[i][0]=0
        if np.sum(y_data[i])==0:
            y_data[i][0]=1
    return y_data
    
def test_model1(model,dataloader1,factor,factor1,method_name):
    model.eval()
    true_value=np.zeros((1,9))
    pred_value=np.zeros((1,9))
    pred_value1=np.zeros((1,9))
    source_value=np.zeros((1,64))
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader1):
            t_img, t_label = t_img.type(torch.FloatTensor),t_label.type(torch.FloatTensor)
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output,source = model.predict(t_img)
            class_output=nn.Sigmoid()(class_output)
            class_output_np=class_output.to(torch.device('cpu')).numpy()
            label_np=t_label.to(torch.device('cpu')).numpy()
            source_value=np.concatenate((source_value,source.to(torch.device('cpu')).numpy()))
            true_value=np.concatenate((true_value, label_np))
            pred_value=np.concatenate((pred_value, round_compute(class_output_np)))
            pred_value1=np.concatenate((pred_value1, class_output_np))
            
    true_value=true_value[1:]
    pred_value=pred_value[1:]
    pred_value1=pred_value1[1:]
    source_value=source_value[1:]
    F1_macro=f1_score(true_value,pred_value,average='macro')
    F1s=f1_score(true_value,pred_value,average=None)
    acc1=accuracy_score(true_value,pred_value)
    F1_macro1=np.mean(np.array(F1s)[1:])
    print('Acc: {:.6f},F1_macro: {:.6f}'.format(acc1,  F1_macro*9/4))
    print(F1s)
    print(F1_macro1*8/3)

def test_model4(model,dataloader1):
    model.eval()
    true_value=np.zeros((1,9))
    pred_value=np.zeros((1,9))
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader1):
            t_img, t_label = t_img.type(torch.FloatTensor),t_label.type(torch.FloatTensor)
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output,source = model.predict(t_img)
            class_output=nn.Sigmoid()(class_output)
            class_output_np=class_output.to(torch.device('cpu')).numpy()
            label_np=t_label.to(torch.device('cpu')).numpy()
            true_value=np.concatenate((true_value, label_np))
            pred_value=np.concatenate((pred_value, round_compute(class_output_np)))
            
    true_value=true_value[1:]
    pred_value=pred_value[1:]
    F1_macro=f1_score(true_value,pred_value,average='macro')
    F1s=f1_score(true_value,pred_value,average=None)
    acc1=accuracy_score(true_value,pred_value)
    F1_macro1=np.mean(np.array(F1s)[1:])
    print('Acc: {:.6f},F1_macro: {:.6f}'.format(acc1,  F1_macro*9/4))
    print(F1s)
    print(F1_macro1*8/3)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    
    # 添加颜色条
    cbar = ax.figure.colorbar(ax.imshow(cm, cmap='Blues'))
    cbar.ax.set_ylabel('Number', rotation=-90, va="bottom")
    
    # 添加文本
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > np.max(cm) / 2 else "black")
    
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix')
    plt.show()

def plot_ruc(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr(y_test, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.show()

def get_best_model_name(filename):
    pathDirs=os.listdir(filename)
    index_list=[]
    best_filename=''
    for pathDir in pathDirs:
        index_list.append(int(re.split('_',pathDir)[2]))
    if len(index_list)>0:
        a=index_list.index(max(index_list))
        best_filename=filename+pathDirs[a]
    else:
        print(filename+' no pth!')
    return best_filename

def get_best_model_name1(filename):
    pathDirs=os.listdir(filename)
    index_list=[]
    best_filename=''
    for pathDir in pathDirs:
        index_list.append(int(re.split('_',pathDir)[1]))
    if len(index_list)>0:
        a=index_list.index(max(index_list))
        best_filename=filename+pathDirs[a]  
    else:
        print(filename+' no pth!')
    return best_filename

def get_best_model_name2(filename):
    pathDirs=os.listdir(filename)
    index_list=[]
    best_filename=''
    for pathDir in pathDirs:
        index_list.append(int(re.split('_',pathDir)[2]))
    if len(index_list)>0:
        a=index_list.index(min(index_list))
        best_filename=filename+pathDirs[a]
    else:
        print(filename+' no pth!')
    return best_filename

if __name__ == '__main__':
    factors=["Hospital"]
    method_names=["U_ECGNet"]
    for factor in factors:
        factors1=["CODE","CPSC"]
        for factor1 in factors1:
            print("original:"+factor+" to "+factor1)
            loader_src, loader_tar, loader_tar_test = load_data(batch_size=128,factor=factor,factor1=factor1)
            model = ECGNet_transfer(method_name="").to(DEVICE)
            model2_dict = model.state_dict()
            save_filename=get_best_model_name('./model/ECGNet/')
            pretext_model = torch.load(save_filename)
                        
            state_dict1 = {k:v for k,v in pretext_model.items() if k in model2_dict.keys()}
            state_dict1_1 = {'feature_layers.'+k:v for k,v in pretext_model.items() if 'feature_layers.'+k in model2_dict.keys()}
            state_dict1.update(state_dict1_1)
            model2_dict.update(state_dict1)
            model.load_state_dict(model2_dict)
            test_model4(model,loader_tar_test)
            del model
            for method_name in method_names:
                print("transfer:"+method_name+" "+factor+" to "+factor1)
                filename=get_best_model_name1('./model_transfer/dataset/'+method_name+'/'+factor+"_"+factor1+'/')
                model = ECGNet_transfer(method_name=method_name).to(DEVICE)
                model=init_weight(model,filename)
                test_model1(model,loader_tar_test,factor,factor1,method_name)
                del model
            del loader_src,loader_tar,loader_tar_test

        
