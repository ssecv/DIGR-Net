import os
import csv
import cv2
# import tensorflow as tf
from PIL import Image
import numpy as np

def decode_csv(csv_file):
    labels=np.zeros(len(open(csv_file).readlines()))
    with open(csv_file) as f:
        reader=csv.reader(f)
        i=0
        for row in reader:
            label=row
            labels[i]=float(label[0][1:-1])
            i=i+1
    return labels

# score=decode_csv('./score_folder/test_ssd.csv')
# print(score)