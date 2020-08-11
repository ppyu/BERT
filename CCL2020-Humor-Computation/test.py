import tensorflow as tf
from tensorflow import keras
import jieba
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense
import sklearn
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

from collections import Counter
print(Counter(np.array([1,0,0,0,1,0,1,1])))
