# DIGR-Net
This is the code repository for DIGR-Net（Depth-induced Gap-reducing Network for RGB-D Salient Object Detection: An Interaction, Guidance and Refinement Approach）

## dataset prapartion for training and test
create training dataset fold: open DIGR-Net Fold and create a new fold named by “dataset_dut”（datestet for training dataset(NUJ2000+NPLR+DUT-RGBD,2985 samples）, the structure of “dataset_dut” is:
- dataset_dut 

-- RGBD_for_test #test dataset
---NJU2K
----RGB
----depth
----GT
---NLPR
...
-- RGBD_for train  #training dataset
---RGB
--- depth
--- GT
-- test_in_train  #validation dataset
---NJU2K
----RGB
----depth
----GT
---NLPR
...


## train
###trianing with 2985 samples(NJU2K+NLPR+DUT-RGBD)
With datstet for trainig preapread, open the fold "DIGR-Net" to find the file 'digr_train.py' , just run it!

###trianing with 2185 samples(NJU2K+NLPR+DUT-RGBD)
open ./DIGR-Net/options.py and change the default fold from "./dataset_dut/RGBD_for_train/RGB/" to "./dataset/RGBD_for_train/RGB/",

and you should open digr_train.py to set 

train_score=decode_csv('./score_folder/train_dut.csv')
val_score=decode_csv('./score_folder/val_dut.csv')

to 

train_score=decode_csv('./score_folder/train.csv')
val_score=decode_csv('./score_folder/val.csv')

respectively.

Finally, you should set batch from 10 to 6,
then you can run digr_train.py to train

datasets can be downloaded from : [link](https://pan.baidu.com/s/1KkL1b5QQn6CmzLpZ9SGjoA?pwd=slpy) (slpy)

## test
open the fold "DIGR-Net" to find the file 'digr_test.py' （if you want to train with 2985 samples） of you can change the default test_path from "./dataset_dut/RGBD_for_test/" to "./dataset/RGBD_for_test/

## trained model and results
trained model：
result maps: 

## Evaluation
find the evaluation method(matlab code) from [evaluation code](http://dpfan.net/d3netbenchmark/)



