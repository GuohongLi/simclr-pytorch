#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: feature.py
Author: zhenglinhai(zhenglinhai@baidu.com)
Date: 2018/04/23 11:40:14
"""

from scipy.spatial.distance import pdist
import time
import sys
import base64
import json
import numpy as np
import time 
import os
import random
import cv2


if __name__ == '__main__':

    imgfea_list = []
    with open(sys.argv[1]) as fp:
        for line in fp:
            imgfea_list.append(line.strip().split('\t'))
    max_len = len(imgfea_list)
    index = 0
    while True:
        if index+1 >= max_len:
            break 
        name1, fea2048_1, fea128_1 = imgfea_list[index][0], [float(x) for x in imgfea_list[index][1].split(' ')], [float(x) for x in imgfea_list[index][2].split(' ')]
        name2, fea2048_2, fea128_2 = imgfea_list[index+1][0], [float(x) for x in imgfea_list[index+1][1].split(' ')], [float(x) for x in imgfea_list[index+1][2].split(' ')]
        dist2048 = pdist(np.vstack([fea2048_1, fea2048_2]),'cosine')
        dist128 = pdist(np.vstack([fea128_1, fea128_2]),'cosine')
        print(name1, name2, dist2048, dist128)
        index = index + 2
