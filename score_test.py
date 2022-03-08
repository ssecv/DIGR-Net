import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img
from tensorflow.keras import backend as K
import os
import torch
import torch.nn.functional as F
import sys

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from IRFFNet.data import get_loader

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def get_score(test_images):
    print(test_images.shape)
    channel_repeat_image=np.zeros((test_images.shape[0],224,224,3))
    for i in range(test_images.shape[0]):
        channel_repeat_image[i]=test_images[i].repeat(3,axis=2)
    IMG_SHAPE = (224, 224, 3)
    input_layer = tf.keras.layers.Input((224, 224, 3))
    base_model = tf.keras.applications.DenseNet121(include_top=False, input_shape=IMG_SHAPE, weights='imagenet')
    x = base_model(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=predictions)

    model.load_weights('Densenet121_M.0.0024（1）.hdf5',by_name=False)
    score_pre=model.predict(channel_repeat_image,batch_size=1, verbose=0)
    return score_pre

# set the path
# image_root = opt.rgb_root
# gt_root = opt.gt_root
# depth_root = opt.depth_root
# test_image_root = opt.test_rgb_root
# test_gt_root = opt.test_gt_root
# test_depth_root = opt.test_depth_root
# save_path = opt.save_path

# get_score('merge_train_test_data/based_on_cosine/twice/nlpr500_test.csv')
    # load score
# train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
# depths_train_list=[]
# for i, (_, _, depths) in enumerate(train_loader, start=1):
#     depths=depths.cuda()
#     depths1=F.upsample(depths, size=(224, 224), mode='bilinear', align_corners=False)
#     depths1 = depths1.view(depths1.shape[0], 224, 224, -1)
#     depths1 = depths1.data.cpu().numpy()
#     print(depths1.shape)
#     depths_train_list.append(depths1)
# depths_total_train=np.zeros((2185,224,224,1))
# for i in range(2185):
#     for j in range(len(depths_train_list)):
#         for k in range(len(depths_train_list[j])):
#             depths_total_train[i]=depths_train_list[j][k]
# print(depths_total_train.shape)
#
#
# test_datasets = ['LFSD','NJU2K_Test','NLPR_test','STERE', 'DES', 'SSD','SIP',"DUT"]
# dataset_path = opt.test_path
# for dataset in test_datasets:
#     depth_root=dataset_path +dataset +'/depth/'
#     test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)
#     for i in range(test_loader.size):
#         image, gt,depth, name, image_for_post = test_loader.load_data()
#         depths = depth.cuda()
#         depths1 = F.upsample(depths, size=(224, 224), mode='bilinear', align_corners=False)
#         depths1 = depths1.view(depths1.shape[0], 224, 224, -1)
#         depths1 = depths1.data.cpu().numpy()
#         print(depths1.shape)
#         depths_test_list.append(depths1)
# depths_test_list=np.array(depths_test_list)
# print(depths_test_list.shape)
# depths_total_test=np.zeros((2185,224,224,1))
# for i in range(2185):
#     for j in range(len(depths_test_list)):
#         for k in range(len(depths_test_list[j])):
#             depths_total_test[i]=depths_test_list[j][k]
# print(depths_total_test.shape)
#
#
#create train and val csv
# import csv
#
# list=os.listdir("./IRFF_dataset/test_in_train/depth")
# print(list)
# train_images=np.zeros((1200,224,224,1))
# for i in range(1200):
#     print(list[i])
#     img=cv2.imread("./IRFF_dataset/test_in_train/depth/"+list[i],cv2.IMREAD_GRAYSCALE)
#     img=cv2.resize(img,(224,224))
#     img = (img - np.min(img)) / (np.max(img) - np.min(img))
#     img=img.reshape(224,224,1)
#     train_images[i]=img
# scores=get_score(train_images)
#
# def generate_csv(output_folder,score):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     path_format = os.path.join(output_folder, "{}.csv")  # 生成文件名
#     out_csv1 = path_format.format('val_1')
#     for i in range(len(score)):
#         with open(out_csv1, "a", encoding="utf-8",newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([scores[i]])
# database_path= './score_folder/'
# csv_files=generate_csv(database_path,scores)
# # #
