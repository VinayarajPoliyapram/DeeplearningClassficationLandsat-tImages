#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


class UNet_dilate(chainer.Chain):

    insize = 128

    def __init__(self, n_class=3, bn=True, wpad=True):
        super(UNet_dilate, self).__init__()

        pad = 1 if wpad else 0
        self.outsize = self.insize if wpad else (self.insize - 92)

        self.add_link('c1_1', L.Convolution2D(6, 64, ksize=3, stride=1, pad=pad))
        self.add_link('c1_2', L.Convolution2D(64, 64, ksize=3, stride=1, pad=pad))
        self.add_link('c2_1', L.Convolution2D(64, 64, ksize=3, stride=1, pad=pad))
        self.add_link('c2_2', L.Convolution2D(64, 64, ksize=3, stride=1, pad=pad))
        self.add_link('di1_1', L.DilatedConvolution2D(64, 64, ksize=3, stride=1, dilate=2,pad=2))
        self.add_link('di2_1', L.DilatedConvolution2D(64, 64, ksize=3, stride=1, dilate=4,pad=4))
        self.add_link('di3_1', L.DilatedConvolution2D(64, 64, ksize=3, stride=1, dilate=6,pad=6))
        
        #self.add_link('up4', L.Deconvolution2D(1024, 512, ksize=4, stride=2, pad=0))
        #self.add_link('up4', L.Deconvolution2D(1024, 512, ksize=2, stride=2, pad=0))
        self.add_link('dc4_1', L.Convolution2D(128, 64, ksize=3, stride=1, pad=pad))
        self.add_link('dc4_2', L.Convolution2D(64, 64, ksize=3, stride=1, pad=pad))
        #self.add_link('up3', L.Deconvolution2D(512, 256, ksize=4, stride=2, pad=0))
        #self.add_link('up3', L.Deconvolution2D(512, 256, ksize=2, stride=2, pad=0))
        self.add_link('dc3_1', L.Convolution2D(128, 64, ksize=3, stride=1, pad=pad))
        self.add_link('score', L.Convolution2D(64, n_class, ksize=1, stride=1, pad=0))

        if bn:
            self.add_link('bnc1_1', L.BatchNormalization(64))
            self.add_link('bnc1_2', L.BatchNormalization(64))
            self.add_link('bnc2_1', L.BatchNormalization(64))
            self.add_link('bnc2_2', L.BatchNormalization(64))
            
            self.add_link('bndi1_1', L.BatchNormalization(64))
            self.add_link('bndi2_1', L.BatchNormalization(64))
            self.add_link('bndi3_1', L.BatchNormalization(64))
            #self.add_link('bnup4', L.BatchNormalization(512))
            self.add_link('bnd4_1', L.BatchNormalization(64))
            self.add_link('bnd4_2', L.BatchNormalization(64))
            #self.add_link('bnup3', L.BatchNormalization(256))
            self.add_link('bnd3_1', L.BatchNormalization(64))
            #self.add_link('bnd3_2', L.BatchNormalization(64))
        self.bn = bn


    def __call__(self, x, t):
        score = self.calc(x)

        loss = F.softmax_cross_entropy(score, t, ignore_label=3)
        accuracy = F.accuracy(score, t, ignore_label=3)

        chainer.report({'loss': loss, 'accuracy': accuracy}, self)

        return loss


    def forward(self, x):
        with chainer.using_config('train', False):
            score = self.calc(x)

        return F.softmax(score)


    def calc(self, x):
        if self.bn:
            h1_1 = F.relu(self.bnc1_1(self.c1_1(x)))
            h1_2 = F.relu(self.bnc1_2(self.c1_2(h1_1)))
            # p1 = F.max_pooling_2d(h1_2, ksize=2, stride=2)
            
            h2_1 = F.relu(self.bnc2_1(self.c2_1(h1_2)))
            h2_2 = F.relu(self.bnc2_2(self.c2_2(h2_1))) 
            #p2 = F.max_pooling_2d(h2_2, ksize=2, stride=2)
            del h2_1 
            h3_1 = F.relu(self.bndi1_1(self.di1_1(h2_2)))
            h3_2 = F.relu(self.bndi2_1(self.di2_1(h3_1))) 
            h3_3 = F.relu(self.bndi3_1(self.di3_1(h3_2)))
            #p3 = F.max_pooling_2d(h3_2, ksize=2, stride=2)
            del h3_1 

            #up4 = F.relu(self.bnup4(self.up4(h5_2)))
            #up4 = self.__crop_to_target_2d(up4, h4_2)
            dh4_1 = F.relu(self.bnd4_1(self.dc4_1(F.concat([h3_3, h2_2]))))
            dh4_2 = F.relu(self.bnd4_2(self.dc4_2(dh4_1)))
            dh3_1 = F.relu(self.bnd3_1(self.dc3_1(F.concat([dh4_2, h1_2]))))
            #dh3_2 = F.relu(self.bnd3_2(self.dc3_2(dh3_1)))
            del dh4_2,dh4_1

            score = self.score(dh3_1)
            #del dh1_2

        else:
            h1_1 = F.relu(self.c1_1(x))
            h1_2 = F.relu(self.c1_2(h1_1))
            p1 = F.max_pooling_2d(h1_2, ksize=2, stride=2)
            del h1_1
            h2_1 = F.relu(self.c2_1(p1))
            h2_2 = F.relu(self.c2_2(h2_1)) 
            p2 = F.max_pooling_2d(h2_2, ksize=2, stride=2)
            del p1,h1_1 
            h3_1 = F.relu(self.c3_1(p2))
            h3_2 = F.relu(self.c3_2(h3_1)) 
            p3 = F.max_pooling_2d(h3_2, ksize=2, stride=2)
            del p2,h3_1
            h4_1 = F.relu(self.c4_1(p3))
            h4_2 = F.relu(self.c4_2(h4_1)) 
            p4 = F.max_pooling_2d(h4_2, ksize=2, stride=2)
            del p3,h4_1
            h5_1 = F.relu(self.c5_1(p4))
            h5_2 = F.relu(self.c5_2(h5_1)) 
            del p4,h5_1

            up4 = F.relu(self.up4(h5_2))
            up4 = self.__crop_to_target_2d(up4, h4_2)
            dh4_1 = F.relu(self.dc4_1(F.concat([h4_2, up4])))
            dh4_2 = F.relu(self.dc4_2(dh4_1))
            del h5_2, up4, h4_2, dh4_1
            up3 = F.relu(self.up3(dh4_2))
            up3 = self.__crop_to_target_2d(up3, h3_2)
            dh3_1 = F.relu(self.dc3_1(F.concat([h3_2, up3])))
            dh3_2 = F.relu(self.dc3_2(dh3_1))
            del dh4_2, up3, h3_2, dh3_1
            up2 = F.relu(self.up2(dh3_2))
            up2 = self.__crop_to_target_2d(up2, h2_2)
            dh2_1 = F.relu(self.dc2_1(F.concat([h2_2, up2])))
            dh2_2 = F.relu(self.dc2_2(dh2_1))
            del dh3_2, up2, h2_2, dh2_1
            up1 = F.relu(self.up1(dh2_2))
            up1 = self.__crop_to_target_2d(up1, h1_2)
            dh1_1 = F.relu(self.dc1_1(F.concat([h1_2, up1])))
            dh1_2 = F.relu(self.dc1_2(dh1_1))
            del dh2_2, up1, h1_2, dh1_1
            score = self.score(dh1_2)
            del dh1_2

        return score


    def __crop_to_target_2d(self, x, target):
        """Crop variable to target shape.
        Args:
            x (~chainer.Variable): Input variable of shape :math:`(n, c_I, h, w)`.
            target (~chainer.Variable): Variable with target output shape
                :math:`(n, h, w)` or `(n, c_I, h, w)`.
        """
        if target.ndim == 3:
            t_h, t_w = target.shape[1], target.shape[2]
        
        elif target.ndim == 4:
            t_h, t_w = target.shape[2], target.shape[3]
        
        cr = int((x.shape[2] - t_h) / 2)
        cc = int((x.shape[3] - t_w) / 2)
        x_cropped = x[:, :, cr:cr + t_h, cc:cc + t_w]

        return x_cropped
