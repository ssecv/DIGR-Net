import os
import torch
import torch.nn.functional as F
import sys

import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from digr_res50 import DIGR
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, set_seed
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
# from score_test import get_score
from decode_csv import decode_csv

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# set_seed(0)
# build the model
model = DIGR()
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_score=decode_csv('./score_folder/train_dut.csv')
val_score=decode_csv('./score_folder/val_dut.csv')
train_loader = get_loader(image_root, gt_root, depth_root,train_score, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
total_step = len(train_loader)


logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("DIGRNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths, scores, names) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            scores = scores.cuda().unsqueeze(1).unsqueeze(2).unsqueeze(3).type_as(depths)

            s1,s2,s3,s4,s5,s6,s7 = model(images, depths, scores)
            loss1 = CE(s1, gts)
            loss2 = CE(s2, gts)
            loss3 = CE(s3, gts)
            loss4 = CE(s4, gts)
            loss5 = CE(s5, gts)
            loss6 = CE(s6, gts)
            loss7 = CE(s7, gts)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}, '
                      'Loss5: {:.4f}, Loss6: {:.4f} , Loss7: {:.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data, loss6.data, loss7.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f},Loss5: {:.4f}, Loss6: {:.4f}, Loss7: {:.4f}  '.
                             format(epoch, opt.epoch, i, total_step, loss1.data, loss2.data, loss3.data, loss4.data, loss5.data, loss6.data, loss7.data))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s1', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'DIGRNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'DIGRNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

# test function
def Jtest(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            # print(name)

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()

            score = val_score[i]
            score = torch.from_numpy(np.array([score]))
            score = score.cuda().unsqueeze(1).unsqueeze(2).unsqueeze(3).type_as(depth)

            _,_,_,_,_,_,res = model(image, depth, score)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'DIGRNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))

import csv
def generate_csv(output_folder,scores):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    path_format = os.path.join(output_folder, "{}.csv")  # 生成文件名
    out_csv1 = path_format.format('train-shuffle')
    for i in range(len(scores)):
        with open(out_csv1, "a", encoding="utf-8",newline='') as f:
            writer = csv.writer(f)
            writer.writerow([scores[i]])

if __name__ == '__main__':
    print("Start train...")

    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        Jtest(test_loader, model, epoch, save_path)
