from __future__ import print_function
import random
import statistics
import re
import functools
import numpy as np
import pickle
import _pickle as cPickle

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

import shap
import lime
import lime.lime_tabular

import fairlearn
from fairlearn.metrics import MetricFrame
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.metrics import * 
import fatf.fairness.data.measures as fatf_dfm
import fatf.utils.data.tools as fatf_data_tools
import fatf.utils.models as fatf_models
import fatf.fairness.models.measures as fatf_mfm
import fatf.utils.metrics.tools as fatf_mt
import sklearn.metrics as skm

global scoring 
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}
