# basic utils
import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from typing import Optional, Callable, List
import random
from tqdm.auto import tqdm as tq
import argparse
from rich.traceback import install
install(show_locals=True, suppress=["torch", "timm", "pytorch_lightning"])



# libraries for data processing 
import numpy as np
import pandas as pd
import copy
import json
import datasets
from PIL import Image

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune

# torchvision
import torchvision
from torchvision import transforms as T

# torch utils
from torchinfo import summary
from torchmetrics.classification import MultilabelAccuracy, Accuracy

# timm
from timm import create_model
from timm.optim import create_optimizer_v2

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

# logging
from loguru import logger

# sklearn
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# huggingface transformers
from transformers import ElectraTokenizer, ElectraTokenizerFast, ElectraModel
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, DistilBertModel
from transformers import AutoTokenizer, AutoModel
from transformers import logging
logging.set_verbosity_error()


