# tensorflow decision forests

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd


# load dataset

dataset_df = pd.read_csv('data//train.csv')