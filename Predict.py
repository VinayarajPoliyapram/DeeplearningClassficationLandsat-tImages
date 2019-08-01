from __future__ import print_function
import numpy as np
import gdal
import argparse
import chainer
import chainer.functions as F
import chainer.links as L

import chainer
import numpy as np
import pickle
import os
import cv2
from chainer import Function, gradient_check, Variable, optimizers, utils
from chainer import Link, Chain, ChainList
from chainer import serializers
from chainer import cuda
from chainer import serializers
from chainer import Variable
import chainer.functions as F
from model import UNetDilate


def TOA_data (MTL):
   data = {}
   fl = open(MTL, 'r')
   file_lines = fl.readlines()
   for lin in file_lines:
      values = lin.split(' = ')
      if len(values) != 2:
         continue
      data[values[0].strip()] = values[1].strip().strip('"')
   return  data

def TOA_correct (img,i,data):
   #print (bands.max())
   bands  = img[i].astype(np.float32)
   bands[bands == 0] =np.nan
   Ml=float(data['REFLECTANCE_MULT_BAND_'+ str(i+2)])
   Ad =float(data['REFLECTANCE_ADD_BAND_'+ str(i+2)])
   print (float(data['REFLECTANCE_ADD_BAND_'+ str(i+2)]))
   bands = (Ml * bands) + Ad
   return bands


def dataimport(img_nm, channel):
    model = UNetDilate()
    chainer.serializers.load_npz("trained_weights/model_epoch-90", model)
    size = 128
    print ("size", size)
    train = open('pred_list.txt', 'r')
    train = train.readlines()

    even = np.arange(img_nm)
    for line in even:
             k = []
             print (line)
             print ("Data:",train[line])
             x = train[line]       
             x, sep, tail = x.partition('\n')
             p = x
             data = TOA_data("data/"+ x +"/"+ x + "_MTL.txt")
             bands_li = []

             for i in range(channel):
                 # Read tif file
                 x_train = gdal.Open( 'data/' + x +'/'+ x + '_B' + str(i+2)+'.TIF')
                 x_train = x_train.ReadAsArray()
                 bands = TOA_correct(x_train,i,data)
                 bands_li.append(bands)

             x_train = np.stack((bands_li),0)
           
             rowsx_train = x_train.shape[1]%size
             rowsx_train = x_train.shape[1]-rowsx_train
             colsx_train = x_train.shape[2]%size
             colsx_train = x_train.shape[2]-colsx_train
             # make the paches of 128 with the shape(paches,bands,rows,cols) for x_train 
             for y in range(0,rowsx_train,size):
                for x in range(0,colsx_train,size):
                   l = []
                   for i in range(x_train.shape[0]):
                      crop = x_train[i][y:y+size, x:x+size]
                      l.append(crop)

                   if (len(l)) == channel:
                      i_stack= np.stack((l),0)
                      k.append(i_stack)
                   else:
                      continue

             x_train = np.stack((k),0)

             predictions = np.float32( np.zeros((x_train.shape[0],x_train.shape[2],x_train.shape[3],2)) )
             #count = 0
             testdatasize =len(x_train)
             batchsize = 64
             for i in range(0, testdatasize, batchsize):

                x_batch = Variable(cuda.to_cpu(np.float32(x_train[i:i+batchsize])))
                pred = model.predict(x_batch)
                pred_label = np.transpose(pred,(0,2,3,1))
                res = np.float32(cuda.to_cpu(pred_label.data))
                predictions[i:i+len(x_batch)] = res

             print ("predictions",predictions.shape)  
             arr = np.empty(rowsx_train*colsx_train*2,dtype=np.float32).reshape(rowsx_train,colsx_train,2)
             
             #samples = predictions.shape[0]
             siz = predictions.shape[1]
             sample_x = int(rowsx_train/siz)
             sample_y = int(colsx_train/siz)
             for g in range (sample_x):
               for h in range(sample_y):
                  for i in range(siz):
                     for j in range(siz):
                              
                        arr[g*siz+i,h*siz+j,:] = predictions[g*sample_y+h,i,j,:]

             np.save( 'Predict_'+ p +'.npy',arr) 

            

if __name__ == "__main__":
   dataimport(1,6)

