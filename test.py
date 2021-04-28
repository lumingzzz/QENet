# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, csv
from PIL import Image
import numpy as np
from modules import qe_model
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.empty_cache()


def valid():
    test_augment = True
    in_img_path = '/data/NTIRE_2021/validation_images/fixed_qp/'
    out_img_path = '/workspace/test/validation_images/fixed_qp/'

    with open('/data/NTIRE_2021/data_validation_fixed_QP.csv', 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            if row[0] == 'Name':
                continue
            else:
                for i in range(10,int(row[3])+1,10):
                    in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                    ref1_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                    ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                    ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                    if i == int(row[3]):
                        ref4_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                    else:
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'
                
                    in_img = np.array(Image.open(in_img)).astype(np.float32)/255.
                    ref1_img = np.array(Image.open(ref1_img)).astype(np.float32)/255.
                    ref2_img = np.array(Image.open(ref2_img)).astype(np.float32)/255.
                    ref3_img = np.array(Image.open(ref3_img)).astype(np.float32)/255.
                    ref4_img = np.array(Image.open(ref4_img)).astype(np.float32)/255.
                    ref5_img = np.array(Image.open(ref5_img)).astype(np.float32)/255.
                    ref6_img = np.array(Image.open(ref6_img)).astype(np.float32)/255.
                        
                    in_img = torch.FloatTensor(in_img).permute(2,0,1).unsqueeze(0)
                    ref1_img = torch.FloatTensor(ref1_img).permute(2,0,1).unsqueeze(0)
                    ref2_img = torch.FloatTensor(ref2_img).permute(2,0,1).unsqueeze(0)
                    ref3_img = torch.FloatTensor(ref3_img).permute(2,0,1).unsqueeze(0)
                    ref4_img = torch.FloatTensor(ref4_img).permute(2,0,1).unsqueeze(0)
                    ref5_img = torch.FloatTensor(ref5_img).permute(2,0,1).unsqueeze(0)
                    ref6_img = torch.FloatTensor(ref6_img).permute(2,0,1).unsqueeze(0)

                    if test_augment == True:
                        out_sum = torch.zeros(in_img.shape).cuda()

                        for k in range(0, 2):

                            in_imgs = torch.rot90(torch.stack([ref1_img, ref2_img, ref3_img, in_img, ref4_img, ref5_img, ref6_img], dim=1), k, [3,4])
                            in_imgs = in_imgs.cuda()

                            with torch.no_grad():
                                out_1 = model(in_imgs, train=False)
                                out_1 = torch.rot90(out_1, -k, [2,3])
                                # lr
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                out_2 = model(in_imgs, train=False)
                                out_2 = torch.flip(out_2, dims=[2])
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                out_2 = torch.rot90(out_2, -k, [2,3])
                                # ud
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_3 = model(in_imgs, train=False)
                                out_3 = torch.flip(out_3, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_3 = torch.rot90(out_3, -k, [2,3])
                                # lr-ud
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_4 = model(in_imgs, train=False)
                                out_4 = torch.flip(out_4, dims=[2])
                                out_4 = torch.flip(out_4, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_4 = torch.rot90(out_4, -k, [2,3])
                        
                            out = 1/4 * (out_1 + out_2 + out_3 + out_4)
                            out_sum += out

                        out = 1/4 * out_sum

                    else:
                        in_imgs = torch.stack([ref1_img, ref2_img, ref3_img, in_img, ref4_img, ref5_img, ref6_img], dim=1)
                        in_imgs = in_imgs.cuda()

                        with torch.no_grad():
                            out = model(in_imgs, train=False)

                    out = out.clamp(0,1.0)
                    out = out.squeeze(0).permute(1,2,0).cpu().numpy()
                    out = (out*255.0).astype(np.uint8)
                    out = Image.fromarray(out)
                    if not os.path.exists(out_img_path+row[0]):
                        os.makedirs(out_img_path+row[0])
                    out.save(out_img_path+row[0]+'/'+str(i).zfill(3)+'.png')


def test():
    test_augment = True
    in_img_path = '/data/NTIRE_2021/test_images/fixed_qp/'
    out_img_path = '/workspace/test/test_images/fixed_qp/'

    with open('/data/NTIRE_2021/data_test_fixed-QP.csv', 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            # print(row)
            if row[0] == '\ufeffName':
                continue
            else:
                for i in range(10,int(row[3])+1,10):
                    in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                    ref1_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                    ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                    ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                    if i == int(row[3]):
                        ref4_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                    else:
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'
                
                    in_img = np.array(Image.open(in_img)).astype(np.float32)/255.
                    ref1_img = np.array(Image.open(ref1_img)).astype(np.float32)/255.
                    ref2_img = np.array(Image.open(ref2_img)).astype(np.float32)/255.
                    ref3_img = np.array(Image.open(ref3_img)).astype(np.float32)/255.
                    ref4_img = np.array(Image.open(ref4_img)).astype(np.float32)/255.
                    ref5_img = np.array(Image.open(ref5_img)).astype(np.float32)/255.
                    ref6_img = np.array(Image.open(ref6_img)).astype(np.float32)/255.
                        
                    in_img = torch.FloatTensor(in_img).permute(2,0,1).unsqueeze(0)
                    ref1_img = torch.FloatTensor(ref1_img).permute(2,0,1).unsqueeze(0)
                    ref2_img = torch.FloatTensor(ref2_img).permute(2,0,1).unsqueeze(0)
                    ref3_img = torch.FloatTensor(ref3_img).permute(2,0,1).unsqueeze(0)
                    ref4_img = torch.FloatTensor(ref4_img).permute(2,0,1).unsqueeze(0)
                    ref5_img = torch.FloatTensor(ref5_img).permute(2,0,1).unsqueeze(0)
                    ref6_img = torch.FloatTensor(ref6_img).permute(2,0,1).unsqueeze(0)

                    if test_augment == True:
                        out_sum = torch.zeros(in_img.shape).cuda()

                        for k in range(0, 2):

                            in_imgs = torch.rot90(torch.stack([ref1_img, ref2_img, ref3_img, in_img, ref4_img, ref5_img, ref6_img], dim=1), k, [3,4])
                            in_imgs = in_imgs.cuda()

                            with torch.no_grad():
                                out_1 = model(in_imgs, train=False)
                                out_1 = torch.rot90(out_1, -k, [2,3])
                                # lr
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                out_2 = model(in_imgs, train=False)
                                out_2 = torch.flip(out_2, dims=[2])
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                out_2 = torch.rot90(out_2, -k, [2,3])
                                # ud
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_3 = model(in_imgs, train=False)
                                out_3 = torch.flip(out_3, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_3 = torch.rot90(out_3, -k, [2,3])
                                # lr-ud
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_4 = model(in_imgs, train=False)
                                out_4 = torch.flip(out_4, dims=[2])
                                out_4 = torch.flip(out_4, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_4 = torch.rot90(out_4, -k, [2,3])
                        
                            out = 1/4 * (out_1 + out_2 + out_3 + out_4)
                            out_sum += out

                        out = 1/4 * out_sum

                    else:
                        in_imgs = torch.stack([ref1_img, ref2_img, ref3_img, in_img, ref4_img, ref5_img, ref6_img], dim=1)
                        in_imgs = in_imgs.cuda()

                        with torch.no_grad():
                            out = model(in_imgs, train=False)

                    out = out.clamp(0,1.0)
                    out = out.squeeze(0).permute(1,2,0).cpu().numpy()
                    out = (out*255.0).astype(np.uint8)
                    out = Image.fromarray(out)
                    if not os.path.exists(out_img_path+row[0]):
                        os.makedirs(out_img_path+row[0])
                    out.save(out_img_path+row[0]+'/'+str(i).zfill(3)+'.png')


def test_full():
    test_augment = True
    in_img_path = '/data/NTIRE_2021/test_images/fixed_qp/'
    out_img_path = '/workspace/test/test_images_full/fixed_qp/'

    with open('/data/NTIRE_2021/data_test_fixed-QP.csv', 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            # print(row)
            if row[0] == '\ufeffName':
                continue
            else:
                for i in range(1, int(row[3])+1):
                    if i == 1:
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'

                    elif i == 2:
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'

                    elif i == 3:
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'

                    elif i == int(row[3]):
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'

                    elif i == int(row[3])-1:
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'

                    elif i == int(row[3])-2:
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'

                    else:
                        in_img = in_img_path + row[0] + '/' + str(i).zfill(3) + '.png'
                        ref1_img = in_img_path + row[0] + '/' + str(i-3).zfill(3) + '.png'
                        ref2_img = in_img_path + row[0] + '/' + str(i-2).zfill(3) + '.png'
                        ref3_img = in_img_path + row[0] + '/' + str(i-1).zfill(3) + '.png'
                        ref4_img = in_img_path + row[0] + '/' + str(i+1).zfill(3) + '.png'
                        ref5_img = in_img_path + row[0] + '/' + str(i+2).zfill(3) + '.png'
                        ref6_img = in_img_path + row[0] + '/' + str(i+3).zfill(3) + '.png'
                
                    in_img = np.array(Image.open(in_img)).astype(np.float32)/255.
                    ref1_img = np.array(Image.open(ref1_img)).astype(np.float32)/255.
                    ref2_img = np.array(Image.open(ref2_img)).astype(np.float32)/255.
                    ref3_img = np.array(Image.open(ref3_img)).astype(np.float32)/255.
                    ref4_img = np.array(Image.open(ref4_img)).astype(np.float32)/255.
                    ref5_img = np.array(Image.open(ref5_img)).astype(np.float32)/255.
                    ref6_img = np.array(Image.open(ref6_img)).astype(np.float32)/255.
                        
                    in_img = torch.FloatTensor(in_img).permute(2,0,1).unsqueeze(0)
                    ref1_img = torch.FloatTensor(ref1_img).permute(2,0,1).unsqueeze(0)
                    ref2_img = torch.FloatTensor(ref2_img).permute(2,0,1).unsqueeze(0)
                    ref3_img = torch.FloatTensor(ref3_img).permute(2,0,1).unsqueeze(0)
                    ref4_img = torch.FloatTensor(ref4_img).permute(2,0,1).unsqueeze(0)
                    ref5_img = torch.FloatTensor(ref5_img).permute(2,0,1).unsqueeze(0)
                    ref6_img = torch.FloatTensor(ref6_img).permute(2,0,1).unsqueeze(0)

                    if test_augment == True:
                        out_sum = torch.zeros(in_img.shape).cuda()

                        for k in range(0, 2):

                            in_imgs = torch.rot90(torch.stack([ref1_img, ref2_img, ref3_img, in_img, ref4_img, ref5_img, ref6_img], dim=1), k, [3,4])
                            in_imgs = in_imgs.cuda()

                            with torch.no_grad():
                                out_1 = model(in_imgs, train=False)
                                out_1 = torch.rot90(out_1, -k, [2,3])
                                # lr
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                out_2 = model(in_imgs, train=False)
                                out_2 = torch.flip(out_2, dims=[2])
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                out_2 = torch.rot90(out_2, -k, [2,3])
                                # ud
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_3 = model(in_imgs, train=False)
                                out_3 = torch.flip(out_3, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_3 = torch.rot90(out_3, -k, [2,3])
                                # lr-ud
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_4 = model(in_imgs, train=False)
                                out_4 = torch.flip(out_4, dims=[2])
                                out_4 = torch.flip(out_4, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[3])
                                in_imgs = torch.flip(in_imgs, dims=[4])
                                out_4 = torch.rot90(out_4, -k, [2,3])
                        
                            out = 1/4 * (out_1 + out_2 + out_3 + out_4)
                            out_sum += out

                        out = 1/4 * out_sum

                    else:
                        in_imgs = torch.stack([ref1_img, ref2_img, ref3_img, in_img, ref4_img, ref5_img, ref6_img], dim=1)
                        in_imgs = in_imgs.cuda()

                        with torch.no_grad():
                            out = model(in_imgs, train=False)

                    out = out.clamp(0,1.0)
                    out = out.squeeze(0).permute(1,2,0).cpu().numpy()
                    out = (out*255.0).astype(np.uint8)
                    out = Image.fromarray(out)
                    if not os.path.exists(out_img_path+row[0]):
                        os.makedirs(out_img_path+row[0])
                    out.save(out_img_path+row[0]+'/'+str(i).zfill(3)+'.png')


if __name__=='__main__':
    
    model_checkpoint = torch.load('./ckpt/epoch_59_loss_0.0004752.pkl')

    model = qe_model.VQE(num_in_ch=3, 
                 num_out_ch=3, 
                 num_feat=128, 
                 num_frame=7, 
                 num_extract_block=8, 
                 num_deformable_group=8, 
                 num_reconstruct_block=16,
                 center_frame_idx=3).cuda()

    model.load_state_dict(model_checkpoint)
    # valid()
    test()
    # test_full()
    

