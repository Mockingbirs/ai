import random
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import  metrics
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

np.random.seed(5)

import warnings
warnings.filterwarnings(action='ignore')



df = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

df = pd.DataFrame(df)
df