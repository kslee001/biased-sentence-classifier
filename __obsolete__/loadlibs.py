import os
import random
from typing import Optional
from tqdm.auto import tqdm as tq
import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
import datasets
import copy
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from transformers import ElectraTokenizer, ElectraTokenizerFast, ElectraModel
from transformers import RobertaTokenizer, RobertaTokenizerFast, RobertaModel
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, DistilBertModel
from transformers import AutoTokenizer, AutoModel

from transformers import logging
logging.set_verbosity_error()


from torchinfo import summary
import argparse