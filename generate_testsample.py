from __future__ import division
import numpy as np
import sys
sys.path.append("./mingqingscript")

import scipy.io as sio
import scipy.ndimage.interpolation
#import scipy.signal

import os

import math
import random

import pdb
import random
import numpy as np
import pandas as pd
import pickle
import random
import sys
import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

# torch condiguration
import argparse
from math import log10
#import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.cuda
import torch.backends.cudnn as cudnn
import copy
import torch.backends.cudnn as cudnn
import random
from random import uniform
import h5py
import time
import PIL
from PIL import  Image
import skimage

import h5py
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage, misc

import glob
# plt.axis([0, 10, 0, 1])
# plt.ion()

# for i in range(10):
#     y = np.random.random()
#     plt.scatter(i, y)
#     plt.pause(0.05)
#
# while True:
#     plt.pause(0.05)
def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


input_path = '/ssd2/yeyuan/dehaze/DCPDN/facades/testimgs/'
output_path = '/ssd2/yeyuan/dehaze/DCPDN/facades/testimgs_h5/'
index_txt = os.path.join(output_path, 'index.csv')

index=1
train_list_per=glob.glob(input_path + '*png') + glob.glob(input_path + '*jpg')

print train_list_per
# open(index_txt, 'w').write(str(train_list_per))

from skimage import color
total_num=0

info = []
for item in train_list_per:

    img=misc.imread(item)
    print(img.shape)
    info.append([item, img.shape[0], img.shape[1], img.shape[2]])

    # img=skimage.color.rgb2luv(img)
    # zz=img.shape
    # size1=(zz[0]/2)
    # size1=np.int(size1)
    # img=img[0:size1,:,:]
    # pdb.set_trace()

    directory='./facades/test_cvpr/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    zz=filter(str.isdigit, item)
    # zz=plt.imshow(img)
    # plt.show()

    min_length = max((min(img.shape[0], img.shape[1])//512)*512, 512)
    img=misc.imresize(img,[min_length,min_length]).astype(float)
    print(img.shape)

    haze_image=img
    gt_img=img

    # pdb.set_trace()

    img=img/255

    haze_image=haze_image/255
    gt_img=gt_img/255

    maxhazy = img.max()
    minhazy = img.min()
    # img = (img - minhazy) / (maxhazy - minhazy)
    # img = (img) / (maxhazy )

    # pdb.set_trace()

    # h5f=h5py.File('./facades/newsyn/'+str(total_num)+'.h5','w')
    h5f=h5py.File(output_path+str(total_num)+'.h5','w')

    h5f.create_dataset('haze',data=haze_image)
    h5f.create_dataset('trans',data=gt_img)
    h5f.create_dataset('ato',data=gt_img)
    h5f.create_dataset('gt',data=gt_img)

    # print img.max()
    # print img.min()
    total_num=total_num+1
    print total_num

df = pd.DataFrame(info)
df.to_csv(index_txt, sep=',', index=None, header=None)



