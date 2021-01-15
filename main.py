import timeit
import warnings

warnings.filterwarnings('ignore')
import sklearn
from numpy.random.mtrand import RandomState
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

# 1. Loading dataset
dataset = pd.read_csv("survey.csv")
print(dataset.shape)

# 2. Cleaning our dataset
# Delete some of the features that we don't want to use.
dataset = dataset.drop(["Timestamp"], axis=1)
dataset = dataset.drop(["state"], axis=1)
dataset = dataset.drop(["comments"], axis=1)
dataset = dataset.drop(["Country"], axis=1)
dataset = dataset.drop(["tech_company"], axis=1)
dataset = dataset.drop(["obs_consequence"], axis=1)

# Changing missing (NA) variables to TEMP.
dataset = dataset.fillna("TEMP")
'''print(dataset.iloc[0])'''

# Changing 'self_employed' with 'No'
dataset["self_employed"].replace(["TEMP"], "No", inplace=True)
'''print(dataset["self_employed"])
print(dataset["self_employed"].unique())'''

# Changing 'work_interfere' with 'Don't know'
dataset["work_interfere"].replace(["TEMP"], "Don't know", inplace=True)

# Changing Age values with median if age < 18 or > 120
median = int(dataset["Age"].median())
print(median)

for i in range(dataset.shape[0]):
    if dataset["Age"][i] < 18 or dataset["Age"][i] > 120:
        dataset["Age"][i] = median

# Create 3 arrays for genders (male, female, trans)
male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man", "msle", "mail", "malr",
        "cis man", "cis male", "p"]
female = ["cis female", "f", "female", "woman", "femake", "female ", "cis-female/femme", "female (cis)", "femail",
          "a little about you"]
trans = ["trans-female", "something kinda male?", "queer/she/they", "non-binary", "nah", "all", "enby", "fluid",
         "genderqueer",
         "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter",
         "female (trans)", "queer", "ostensibly male, unsure what that really means"]

dataset["Gender"] = dataset["Gender"].str.lower()

# Decreasing gender values as male, female, trans
for i in range(dataset.shape[0]):
    if dataset["Gender"][i] in male:
        dataset["Gender"][i] = "male"

    elif dataset["Gender"][i] in female:
        dataset["Gender"][i] = "female"

    elif dataset["Gender"][i] in trans:
        dataset["Gender"][i] = "trans"

print(dataset["Gender"].unique())
