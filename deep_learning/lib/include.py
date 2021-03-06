import os
import glob as glob
import tqdm
import shutil
import math
import requests
import random
import functools
import logging
import json
import importlib
from multiprocessing import Process
from queue import Queue
from threading import Thread

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import cv2

import torchvision
import torch

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel.data_parallel import data_parallel

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import pretrainedmodels as pmodels
import segmentation_models_pytorch as smp

LOG_DIR = 'outputs/'
DATA_DIR = 'data/'
