import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from digr_res50 import IRFF

from data import test_dataset
from decode_csv import decode_csv


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./dataset_dut/RGBD_for_test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = IRFF()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./IRFFNet_cpts/') when training.
model.load_state_dict(torch.load('./DIGR-Net_cpts/IRFFNet_epoch_best.pth'))
model.cuda()
model.eval()

#test
test_datasets = ['NJU2K','LFSD','DES', 'SSD', 'SIP', 'STERE', 'DUT', 'NLPR']
# lfsd_score=decode_csv("./score_folder/test_lfsd.csv")
# des_score=decode_csv("./score_folder/test_des.csv")
# ssd_score=decode_csv("./score_folder/test_ssd.csv")
# sip_score=decode_csv("./score_folder/test_sip.csv")

for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/depth/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
    scores = decode_csv("./score_folder/test_"+dataset.lower()+".csv")
    # print(score)
    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()

        score = scores[i]
        score = torch.from_numpy(np.array([score]))
        score = score.cuda().unsqueeze(1).unsqueeze(2).unsqueeze(3).type_as(depth)

        _,_,_,_,_,_,res = model(image, depth, score)
        # res = model(image, depth)
        # _,res = model(image, depth, score)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path+name, res*255)
    print('Test Done!')
