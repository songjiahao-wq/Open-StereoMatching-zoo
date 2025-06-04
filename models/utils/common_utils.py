# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 21:00
# @Author  : sjh
# @Site    : 
# @File    : common_utils.py
# @Comment :
import os
import yaml
import random
import shutil
import numpy as np
import torch
import logging
import inspect
from easydict import EasyDict
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2
def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs