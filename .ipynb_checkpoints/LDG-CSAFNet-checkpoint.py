import random
from random import sample
import argparse
import numpy as np
import os
import pickle
import time
from torch import Tensor
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from scipy.ndimage import uniform_filter
from skimage.segmentation import mark_boundaries
from numpy import ndarray
from skimage import measure
from sklearn.metrics import auc
import pandas as pd
from tqdm import tqdm
from module.backbone.ResNet import ResNet
import torchvision.models as models
from module.cbam import ChannelGate, SpatialGate
import matplotlib.pyplot as plt
import matplotlib
import random
from torch.utils.data import DataLoader, Subset
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import wide_resnet50_2, resnet18,resnet50
import datasets.mvtec as mvtec
import pdb
# pip install scikit-learns
# pip install scikit-image
# pip install scikit-learn

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('LDG-CSAFNe')
    parser.add_argument('--data_path', type=str, default='datasets/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='/tmp/mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2','resnet50'], default='resnet50')
    parser.add_argument('--data_percent', type=float, default=0.1,
                    help='Percentage of training data to use (0~1)')
    parser.add_argument('--data_percentest', type=float, default=0.1,
                    help='Percentage of training data to use (0~1)')
    return parser.parse_args()

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LDCF(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(LDCF, self).__init__()
        self.deConv = nn.ConvTranspose2d(channels_high, channels_low, 2, stride=2)

        self.ChannelGate = ChannelGate(channels_high)
        self.SpatialGate = SpatialGate()
        self.conv1 = conv1x1(channels_high, channels_low)
        self.conv2 = conv3x3(channels_low, channels_low)
        self.relu = nn.ReLU(inplace=True)
        
        # Layer Difference attention branch
        self.diff_conv = nn.Sequential(
            conv3x3(channels_low, channels_low),
            nn.ReLU(inplace=True),
            conv1x1(channels_low, 1),
            nn.Sigmoid()
        )


    def forward(self, fms_high, fms_low):

        x1 = self.deConv(fms_high)
        x2 = self.ChannelGate(fms_high)
        x2 = self.deConv(x2)
        x2 = self.SpatialGate(x1)
        x3 = self.conv2(fms_low)
       
        diff = torch.abs(x1 - x3) 
        attn = self.diff_conv(diff) 
        x2 =  x2 *attn 
        out = x1 + x2 
        out = self.relu(out)
        return out


def main():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=True, progress=True)
        t_d = 512
        d =512
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    #idx = torch.tensor(sample(range(0, 64), 64))
    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[0].conv1.register_forward_hook(hook)#xw
    model.layer1[0].bn1.register_forward_hook(hook) #xws
 
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.layer4[-1].register_forward_hook(hook)


    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(args, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        # train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])#org
        # test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])##org

        train_outputs = OrderedDict([
            ('layer1_conv1', []),  # layer1的第一个卷积层conv1 #xw
            ('layer1_bn1', []),  # layer1的第一个批归一化层bn1
            ('layer1', []),  # layer1的最后一个子模块
            ('layer2', []),  # layer2的最后一个子模块
            ('layer3', []),  # layer2的最后一个子模块
            ('layer4', []),  # layer2的最后一个子模块

        ])
        test_outputs = OrderedDict([
            ('layer1_conv1', []),  # layer1的第一个卷积层conv1 #xw
            ('layer1_bn1', []),  # layer1的第一个批归一化层bn1
            ('layer1', []),  # layer1的最后一个子模块
            ('layer2', []),  # layer2的最后一个子模块
            ('layer3', []),  # layer2的最后一个子模块
            ('layer4', []),  # layer2的最后一个子模块
 
        ])

        # 获取原始数据集
        full_dataset = train_dataloader.dataset
        data_len = len(full_dataset)

        # 按百分比采样索引
        used_len = int(data_len * args.data_percent)
        indices = random.sample(range(data_len), used_len)

        # 构建新 dataloader
        subset = Subset(full_dataset, indices)
        train_dataloader= torch.utils.data.DataLoader(subset, batch_size=train_dataloader.batch_size, shuffle=False)

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(device))
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in train_outputs.items():
                train_outputs[k] = torch.cat(v, 0).to('cpu')

            # Embedding concat
            # embedding_vectors = train_outputs['layer1_conv1']
            # for layer_name in ['layer1_bn1', 'layer1','layer2']:
            #     embedding_vectors = embedding_add(embedding_vectors, train_outputs[layer_name])

            #xw2024 Embedding concat
            ## 特定层权重

            bottom_ch = 512
            ldcf3 = LDCF(bottom_ch, 256)
            ldcf2 = LDCF(bottom_ch // 2, 128)
            ldcf1 = LDCF(bottom_ch // 4, 64)

      

            weights = model.layer3[0].conv1.weight.data.to('cpu') #512-256
            weights_1 = model.layer2[1].conv1.weight.data.to('cpu')#512-128
            weights_2 = model.layer2[0].conv1.weight.data.to('cpu')#256-128
            weights_3 = model.layer4[0].conv2.weight.data.to('cpu')  # 512-512
            
            
            fmap1 = train_outputs['layer1_conv1'] #64 [105, 64, 56, 56]
            fmap2 = train_outputs['layer1_bn1'] #64 [105, 64, 56, 56]
            fmap3 = train_outputs['layer1'] #256 [105, 256, 56, 56]
            fmap4 = train_outputs['layer2'] #512  [105, 512, 28, 28]

           
           
            
            fmap1_resized = F.interpolate(fmap1, size=(224, 224), mode='bilinear', align_corners=False)
            fmap_cat12 = embedding_concat(fmap1, fmap2).to('cpu') #128 [105, 128, 56, 56]
            fmap_cat12_resized = F.interpolate(fmap_cat12, size=(112, 112), mode='bilinear', align_corners=False)
            
 
            
            ############# LDCF
            f_ldcf3 = ldcf3(fmap4 , fmap3)  # 1/16 [105, 256, 56, 56]
            f_ldcf2 = ldcf2(f_ldcf3,  fmap_cat12_resized)  # 1/8 [105, 128, 112, 112]
            f_ldcf1 = ldcf1(f_ldcf2, fmap1_resized)  # 1/4 [105, 64, 224, 224]
            #conv1_1 = conv1x1(64, 64)
            #fmaf1 = conv1_1(fmaf1)
            downsample = nn.MaxPool2d(kernel_size=4, stride=4)
            f_ldcf1= downsample(f_ldcf1)

            fmap1 = embedding_add(fmap1,f_ldcf1)
                    
            ############fcat12
            size_fMap1 = fmap1.size()[2:]  # 取[2:]是因为前两个维度是批次和通道
            fmap_cat12 = embedding_concat(fmap1,fmap2).to('cpu') #128 [105, 128, 56, 56]
            fmap_cat12_resized = F.interpolate(fmap_cat12, size=(112, 112), mode='bilinear', align_corners=False)
            
           
            # 调整 fMap4 的尺寸以匹配 fMap1
            fmap4_resized = F.interpolate(fmap4, size=size_fMap1, mode='nearest').to('cpu') #512
            fmap4_resized_conv1x1 = conv_custom(fmap4_resized, weights, bias=None).to('cpu') #256 #
            fmap_add34 = embedding_add(fmap3,fmap4_resized_conv1x1).to('cpu')#256
            fmap_add34_conv1x1 = conv_custom(fmap_add34, weights_2, bias=None).to('cpu') #128 #
            fmap_cat34 = embedding_concat(fmap3, fmap4_resized_conv1x1).to('cpu')#512

            fmap_cat34_conv1x1 = conv_custom(fmap_cat34, weights, bias=None).to('cpu')#256
            fmap_cat12add34 = embedding_concat(fmap_cat12, fmap_add34_conv1x1).to('cpu')#256
            fmap_cat1234=embedding_concat(fmap_cat12add34, fmap_cat34_conv1x1).to('cpu') #512
            #fmap_cat1234=embedding_concat(fmap_cat1234,fmaf1).to('cpu')
            

            # 创建层归一化模块，指定在通道维度上进行归一化
            conv = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=1).to('cpu')
            bn = torch.nn.BatchNorm2d(512).to('cpu')
            relu = torch.nn.ReLU(inplace=True)

            fmap_cat1234_conv = conv_custom(fmap_cat1234, weights_3, bias=None).to('cpu')  # 512
            fmap_cat1234_bn = bn(fmap_cat1234_conv)
            embedding_vectors = fmap_cat1234_conv +fmap_cat1234_bn
    
           
            # embedding_vectors = fmap_cat1234
           
            # num_channels = fmap_cat1234.size(1)
            # batch_norm = torch.nn.BatchNorm2d(num_features=fmap_cat1234.size(1)).to('cpu')
            # fmap_cat1234_bn = batch_norm(fmap_cat1234).to('cpu')
            # embedding_vectors =fmap_cat1234_bn

            
                        
            #######################################################################################################################

            # embedding_vectors = train_outputs['layer1']
            # for layer_name in ['layer2', 'layer3']:
            #     embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
       
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            #embedding_vectors = embedding_concat(embedding_vectors, fmaf1).to('cpu')  # 512
            # calculate multivariate Gaussian distribution
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W)
            mean = torch.mean(embedding_vectors, dim=0).detach().numpy()
            cov = torch.zeros(C, C, H * W).numpy()
            I = np.identity(C)
            for i in range(H * W):
                # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i].detach().numpy(), rowvar=False) + 0.01 * I
            # save learned distribution
            train_outputs = [mean, cov]
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

         # 获取原始数据集
        full_dataset = test_dataloader.dataset
        data_len = len(full_dataset)

        # 按百分比采样索引
        used_len = int(data_len * args.data_percentest)
        indices = random.sample(range(data_len), used_len)

        # 构建新 dataloader
        subset = Subset(full_dataset, indices)
        test_dataloader= torch.utils.data.DataLoader(subset, batch_size=test_dataloader.batch_size, shuffle=False)

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0).to('cpu')
        
        # Embedding concat
        # embedding_vectors = test_outputs['layer1_conv1']
        # for layer_name in ['layer1_bn1', 'layer1','layer2']:
        #     embedding_vectors = embedding_add(embedding_vectors, test_outputs[layer_name])
        bottom_ch = 512
        ldcf3 = LDCF(bottom_ch, 256)
        ldcf2 = LDCF(bottom_ch // 2, 128)
        ldcf1 = LDCF(bottom_ch // 4, 64)
        
        ## 特定层权重
        weights = model.layer3[0].conv1.weight.data.to('cpu')  # 512-256
        weights_1 = model.layer2[1].conv1.weight.data.to('cpu')  # 512-128
        weights_2 = model.layer2[0].conv1.weight.data.to('cpu')  # 256-128
        weights_3 = model.layer4[0].conv2.weight.data.to('cpu')  # 512-512

        fmap1 = test_outputs['layer1_conv1']
        fmap2 = test_outputs['layer1_bn1']
        fmap3 = test_outputs['layer1']
        fmap4 = test_outputs['layer2']
        
        
        fmap1_resized = F.interpolate(fmap1, size=(224, 224), mode='bilinear', align_corners=False)
        fmap_cat12 = embedding_concat(fmap1, fmap2).to('cpu')
        size_fMap1 = fmap1.size()[2:]  # 取[2:]是因为前两个维度是批次和通道
        fmap_cat12_resized = F.interpolate(fmap_cat12, size=(112, 112), mode='bilinear', align_corners=False)
               
        ############# LDCF
        f_ldcf3 = ldcf3(fmap4 , fmap3)  # 1/16 [105, 256, 56, 56]
        f_ldcf2 = ldcf2(f_ldcf3,  fmap_cat12_resized)  # 1/8 [105, 128, 112, 112]
        f_ldcf1 = ldcf1(f_ldcf2, fmap1_resized)  # 1/4 [105, 64, 224, 224]
        #conv1_1 = conv1x1(64, 64)
        #fmaf1 = conv1_1(fmaf1)
        downsample = nn.MaxPool2d(kernel_size=4, stride=4)
        f_ldcf1= downsample(f_ldcf1)

        fmap1 = embedding_add(fmap1,f_ldcf1)
            
        ############fcat12
        fmap_cat12 = embedding_concat(fmap1, fmap2).to('cpu')
        size_fMap1 = fmap1.size()[2:]  # 取[2:]是因为前两个维度是批次和通道

        fmap_cat12_resized = F.interpolate(fmap_cat12, size=(112, 112), mode='bilinear', align_corners=False)
       
       
       
        # 调整 fMap4 的尺寸以匹配 fMap1
        fmap4_resized = F.interpolate(fmap4, size=size_fMap1, mode='nearest').to('cpu')  # 512
        fmap4_resized_conv1x1 = conv_custom(fmap4_resized, weights, bias=None).to('cpu')  # 256 #
        fmap_add34 = embedding_add(fmap3, fmap4_resized_conv1x1).to('cpu')  # 256
        fmap_add34_conv1x1 = conv_custom(fmap_add34, weights_2, bias=None).to('cpu')  # 128 #
        fmap_cat34 = embedding_concat(fmap3, fmap4_resized_conv1x1).to('cpu')  # 512
        fmap_cat34_conv1x1 = conv_custom(fmap_cat34, weights, bias=None).to('cpu')  # 256
        fmap_cat12add34 = embedding_concat(fmap_cat12, fmap_add34_conv1x1).to('cpu')  # 256
        fmap_cat1234 = embedding_concat(fmap_cat12add34, fmap_cat34_conv1x1).to('cpu')  # 512
        #fmap_cat1234=embedding_concat(fmap_cat1234,fmaf1).to('cpu')

        conv = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=1).to('cpu')
        bn = torch.nn.BatchNorm2d(512).to('cpu')
        relu = torch.nn.ReLU(inplace=True)

        fmap_cat1234_conv = conv_custom(fmap_cat1234, weights_3, bias=None).to('cpu')  # 512
        fmap_cat1234_bn = bn(fmap_cat1234_conv)
        embedding_vectors =fmap_cat1234_conv +fmap_cat1234_bn
        
       
  
        # embedding_vectors = fmap_cat1234
        # # 创建层归一化模块，指定在通道维度上进行归一化
        # num_channels = fmap_cat1234.size(1)
        # batch_norm = torch.nn.BatchNorm2d(num_features=fmap_cat1234.size(1)).to('cpu')
        # fmap_cat1234_bn = batch_norm(fmap_cat1234).to('cpu')
        # embedding_vectors =fmap_cat1234_bn
        
        
       
        
        
        ##################################################################################################
        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
  
        #embedding_vectors = embedding_concat(embedding_vectors, fmaf1).to('cpu')  # 512
        # calculate distance matrix
        inference_start = time.time()
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).detach().numpy()
        dist_list = []
        for i in range(H * W):
            mean = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=2)
        #推理时间
        inference_time = time.time() - inference_start
        print('{} inference time: {:.3f}'.format(class_name, inference_time))
        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        pdb.set_trace()
        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        total_roc_auc.append(img_roc_auc)
        print('image ROCAUC: %.3f' % (img_roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc),linewidth=4)
        
        # get optimal threshold
        gt_mask = np.asarray(gt_mask_list)
        # gt_mask = (gt_mask > 0.5).astype(np.uint8)      ######### xw2025.5.12
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

 
        predictions = (scores > threshold).astype(int)
        f1 = calculate_f1_score(gt_mask.flatten(), predictions.flatten())
        print('F1 Score:', f1)
        # pro_auc = calc_pro_auc(scores, gt_mask, num_th=500)
        # print('PRO AUC: %.3f' % (pro_auc))    #xw
        
        gt_mask_np = np.asarray(gt_mask_list).squeeze(1).astype(np.uint8)
        aupro = compute_pro(gt_mask_np, scores)
        print('PRO AUC: {:.3f}'.format(aupro))
       

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc),linewidth=4)
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.set_title('Average image ROCAUC: %.3f' % np.mean(total_roc_auc), fontsize=28)
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.set_title('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc), fontsize=28)
    fig_pixel_rocauc.legend(loc="lower right", fontsize=28)

    fig_img_rocauc.legend(loc="lower right", fontsize=28)
    fig_pixel_rocauc.legend(loc="lower right", fontsize=28)
    
    fig_img_rocauc.set_yticks([0, 0.5,1])
    fig_pixel_rocauc.set_yticks([0,0.5, 1])
    
    fig_img_rocauc.set_xlabel('False Positive Rate', fontsize=28)
    fig_img_rocauc.set_ylabel('True Positive Rate', fontsize=28)
    fig_pixel_rocauc.set_xlabel('False Positive Rate', fontsize=28)
    fig_pixel_rocauc.set_ylabel('True Positive Rate', fontsize=28)

    # 调整坐标轴刻度标签的字体大小
    fig_img_rocauc.tick_params(axis='both', which='major', labelsize=28)
    fig_img_rocauc.tick_params(axis='both', which='minor', labelsize=28)
    fig_pixel_rocauc.tick_params(axis='both', which='major', labelsize=28)
    fig_pixel_rocauc.tick_params(axis='both', which='minor', labelsize=28)

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def embedding_add(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()

    # 确保x和y的通道数相同
    assert C1 == C2, "Channel numbers of x and y must be the same for addition."

    # 如果空间维度相同，则直接相加
    if H1 == H2 and W1 == W2:
        return x + y

    # 如果x的空间维度大于y，则需要调整y的尺寸
    if H1 > H2 or W1 > W2:
        y_resized = F.interpolate(y, size=(H1, W1), mode='nearest')
        return x + y_resized

    # 如果y的空间维度大于x，则需要调整x的尺寸
    x_resized = F.interpolate(x, size=(H2, W2), mode='nearest')
    return x_resized + y

def calculate_f1_score(predictions, ground_truth):

    f1 = f1_score( predictions,ground_truth)
    return f1

def calc_pro(binary_maps: ndarray, gt_masks: ndarray):
    pros = []
    for binary_map, gt_mask in zip(binary_maps, gt_masks):
        for region in measure.regionprops(measure.label(gt_mask)):
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
            pros.append(tp_pixels / region.area)
    return np.array(pros).mean()

def conv_custom(input_tensor, weights, bias=None):
    out_channels, in_channels, kh, kw = weights.shape
    
    # 根据卷积核大小设置padding，1x1卷积核不需要padding，3x3卷积核设置padding为1
    if kh == 1 and kw == 1:
        padding = 0
    elif kh == 3 and kw == 3:
        padding = 1
    else:
        raise ValueError("Unsupported kernel size. Only 1x1 and 3x3 are supported.")
    
    # 创建卷积层，注意bias设置为是否有bias传递进来
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(kh, kw), stride=1, padding=padding, bias=(bias is not None))
    
    # 将提取的权重和偏置赋给卷积层
    conv_layer.weight = nn.Parameter(weights)
    if bias is not None:
        conv_layer.bias = nn.Parameter(bias)
    
    # 应用卷积
    output_tensor = conv_layer(input_tensor)
    return output_tensor


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """
    Compute the area under the curve of per-region overlapping (PRO) with FPR from 0 to 0.3.

    Args:
        masks (ndarray): All binary masks in test. shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds. Default is 200.

    Returns:
        float: PRO AUC score
    """

    # 1. 参数检查
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(np.unique(masks)).issubset({0, 1}), "masks must be binary {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    # 2. 初始化
    df = pd.DataFrame(columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th

    # 3. 遍历阈值
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        # 计算每个 mask 区域的 PRO
        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                coords = region.coords
                tp_pixels = binary_amap[coords[:, 0], coords[:, 1]].sum()
                pros.append(tp_pixels / region.area)

        # 计算 FPR
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # 添加到 DataFrame
        df.loc[len(df)] = {"pro": np.mean(pros), "fpr": fpr, "threshold": th}

    # 4. 正规化 FPR（0~1 -> 0~0.3）
    df = df[df["fpr"] <= 0.3].copy()
    df["fpr"] = df["fpr"] / df["fpr"].max()

    # 5. 计算 AUC
    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

if __name__ == '__main__':
    main()
