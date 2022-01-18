import pdb
import torch
import math
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import evaluate
import time
import numpy as np
import shutil
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import models
import TreeEvaluationDatasets
from torchsummary import summary
from torch.nn.parameter import Parameter
from thop import profile
import scipy.io as sio

parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--pretrained', dest='pretrained', metavar='PATH', help='use pre-trained model')
parser.add_argument('--crop_size', default=[224, 224], type=int, nargs='+')
parser.add_argument('--extract_features_folder', type=str, default='features/triplet_features')


paths = {
        'gender': ['gender'], 
        'down': ['gender', 'down'], 'clothes': ['gender', 'clothes'], 
        'age': ['gender', 'down', 'age'], 'backpack2': ['gender', 'down', 'backpack2'], 'down1': ['gender', 'clothes', 'down1'],
        'bag1':['gender', 'clothes', 'bag1'], 'bag2': ['gender', 'clothes', 'down1', 'bag2'],
        'downcolor': ['gender', 'down', 'age', 'downcolor'], 'backpack': ['gender', 'down', 'age', 'downcolor', 'backpack'],
        'backpack1': ['gender', 'down', 'age', 'downcolor', 'backpack1'],
        'bag': ['gender', 'down', 'age', 'bag'], 'downcolor1': ['gender', 'down', 'age', 'bag', 'downcolor1'],
        'downcolor2': ['gender', 'down', 'backpack2', 'downcolor2'], 'downcolor3': ['gender', 'down', 'backpack2', 'downcolor3'],
        'upcolor': ['gender', 'down', 'backpack2', 'downcolor2', 'upcolor'],
    }

path_decisions = {
    'down': [0],
    'clothes': [1],
    'age':[0,0],
    'backpack2':[0,1],
    'bag1':[1,0],
    'down1':[1,1],
    'bag2':[1,1,1],
    'downcolor':[0,0,1],
    'downcolor2':[0,1,0],
    'downcolor3':[0,1,1],
    'upcolor':[0,1,0,0],
    'backpack':[0,0,1,0],
    'backpack1':[0,0,1,8],
    'bag':[0,0,2],
    'downcolor1':[0,0,2,0]
}

leaf_DNNs = {
    'downcolor1' : [1,2,3,4,5,6,7,8,9],
    'bag': [2],
    'backpack': [1, 2],
    'backpack1': [1, 2],
    'age':[1,4],
    'downcolor2': [2,3,4,5,6,7,8,9],
    'downcolor': [2,3,4,5,6,7,8],
    'bag1': [1, 2],
    'down1':[1],
    'bag2': [1, 2],
    'upcolor': [1,2,3,4,5,6,7,8]
}

tree_path = {
    'gender': {},
    'down': {'gender': 1},
    'clothes': {'gender': 2},
    'down1': {'gender': 2, 'clothes': 2},
    'bag2': {'gender': 2, 'clothes': 2, 'down':2},
    'bag1': {'gender': 2, 'clothes': 1},
    'age': {'gender': 1, 'down': 1},
    'backpack2': {'gender': 1, 'down': 2},
    'downcolor': {'gender': 1, 'down': 1, 'age': 2},
    'downcolor2': {'gender': 1, 'down': 2, 'backpack': 1},
    'upcolor': {'gender': 1, 'down': 2, 'backpack': 1, 'downcolor':1},
    'downcolor3': {'gender': 1, 'down': 2, 'backpack': 2},
    'backpack': {'gender': 1, 'down': 1, 'age': 2, 'downcolor': 1},
    'backpack1': {'gender': 1, 'down': 1, 'age': 2, 'downcolor': 9},
    'bag': {'gender': 1, 'down': 1, 'age': 3},
    'downcolor1': {'gender': 1, 'down': 1, 'age': 3, 'bag':1},
}

hierarchy = {
    'gender': ['down', 'clothes'],
    'down': ['age', 'backpack2'],
    'age': [None, 'downcolor', 'bag', None],
    'downcolor': ['backpack', None, None, None, None, None, None, None, 'backpack1'],
    'bag': ['downcolor1', None],
    'backpack2': ['downcolor2', 'downcolor3'],
    'downcolor2': ['upcolor', None, None, None, None, None, None, None, None],
    'clothes': ['bag1', 'down1'],
    'down1': [None, 'bag2']
}


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for layer_name, param in state_dict.items():
        layer_name = layer_name.replace('module.','')
        if 'layer_fc' not in layer_name:
            layer_name = layer_name.replace('.1', '_1').replace('.2','_2')
        if isinstance(param, Parameter):
            param = param.data
        if layer_name in own_state:
            try:
                own_state[layer_name].copy_(param)
            except:
                return

def load_checkpoint(model, resume):
    checkpoint = torch.load(resume)
    try:
        args.start_iteration = checkpoint['iterations']
    except:
        args.start_iteration = 0
    try:
        best_acc = checkpoint['best_semantic_acc']
    except:
        best_acc = 0.
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        load_my_state_dict(model, checkpoint['state_dict'])
    best_loss = checkpoint['best_loss']
    return model, best_loss, best_acc



def get_gallery_attributes(gallery_labels):
    attributes = loadmat('attributes/market_attribute.mat')['market_attribute']['test'] # save attributes to attributes directory
    image_indices = attributes['image_index'].astype(np.int)
    gallery_attribute = {}
    for count, image_index in enumerate(image_indices):
        index = np.where(gallery_labels == image_index)[0]
        current_attribute = 'gender' # attribute identified at root
        path = {}
        while current_attribute is not None:
            image_attribute = attributes[''.join(filter(lambda x: x.isalpha(), current_attribute))][count]
            path[current_attribute] = image_attribute-1
            if current_attribute not in hierarchy:
                current_attribute = None
            else:
                current_attribute = hierarchy[current_attribute][image_attribute-1]

        for i in index:
            gallery_attribute[i] = path
    
    return gallery_attribute

def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def extract_features_gallery(dataloader, gallery_attributes):
    features = torch.FloatTensor()
    features_list = []
    tree_outputs = []
    count = 0 
    gallery_path = []
    for i, input in enumerate(dataloader):
        img, label = input
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n,128).zero_().cuda()
        for x in range(2):
            currentDNN = 'gender'
            sem = {}
            if(x==1):
                img = fliplr(img)
            out = Variable(img.cuda())

            while currentDNN is not None:
                model = getattr(models, "get_"+currentDNN)().cuda()
                model, _, _ = load_checkpoint(model, 'checkpoint/'+currentDNN+'/model_3_best.pth.tar')
                model.eval()
                out, att, feat = model.evaluate(out)        
                if i not in gallery_attributes:
                    pred = att.max(1, keepdim=True)[1]
                    sem = (currentDNN, pred.cpu().item())
                else:
                    pred = gallery_attributes[i][currentDNN]
                    sem = (currentDNN, pred.item())

                if currentDNN in hierarchy:
                        currentDNN = hierarchy[currentDNN][pred]
                else:
                    currentDNN = None

            ff += feat
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.data.cpu()), 0)
        del ff
        gallery_path.append(sem)
        print(i, "/", len(dataloader))
    return features, gallery_path


def extract_features_query(dataloader):
    features = torch.FloatTensor()
    features_list = []
    tree_outputs = []
    count = 0 
    query_path = []
    for i, input in enumerate(dataloader):
        img, label = input
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n,128).zero_().cuda()
        for x in range(2):
            currentDNN = 'gender'
            sem = {}
            if(x==1):
                img = fliplr(img)
            out = Variable(img.cuda())

            while currentDNN is not None:
                model = getattr(models, "get_"+currentDNN)().cuda()
                model, _, _ = load_checkpoint(model, 'checkpoint/'+currentDNN+'/model_3_best.pth.tar')
                model.eval()

                out, att, feat = model.evaluate(out)

                pred = att.max(1, keepdim=True)[1]

                sem = (currentDNN, pred.cpu().item())
                if currentDNN in hierarchy:
                    currentDNN = hierarchy[currentDNN][pred]
                else:
                    currentDNN = None

            ff += feat
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff.data.cpu()), 0)
        del ff
        query_path.append(sem)
        print(i, "/", len(dataloader))
    return features, query_path


def loadmat(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



def main():
    global args

    args = parser.parse_args()
    
    for arg in vars(args):
        print(arg, getattr(args, arg))
    writer = SummaryWriter()

    is_tencrop=False
    gen_stage_features = False
    path = {}
    semantic = None

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_image_size = [int(x * 1.125) for x in args.crop_size]

    print('Begin dataloader setup')
    path = {}
    data_transforms = transforms.Compose([
        transforms.Resize(scale_image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    image_datasets = {x: TreeEvaluationDatasets.ImageFolder(os.path.join(args.data,x), data_transforms, tree_path = path, data_type = x) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=1) for x in ['gallery','query']}
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    gallery_attributes = get_gallery_attributes(gallery_label)

    gallery_feature, gallery_semantic = extract_features_gallery(dataloaders['gallery'], gallery_attributes)
    query_feature, query_semantic = extract_features_query(dataloaders['query'])

    result = {
                'gallery_f':gallery_feature.numpy(),
                'gallery_label':gallery_label,
                'gallery_cam':gallery_cam,
                'query_f':query_feature.numpy(),
                'query_label':query_label,
                'query_cam':query_cam,
                'gallery_semantic':gallery_semantic,
                'query_semantic':query_semantic
        }
    sio.savemat('pytorch_result.mat',result)


if __name__ == '__main__':
    main()