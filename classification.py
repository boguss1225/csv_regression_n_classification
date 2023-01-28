import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import lazypredict

from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from sklearn.metrics import make_scorer

print("Import Data")
df=pd.read_csv("heart_attack_dataset/heart.csv")
print(df.head())
print(df.info())
print(df.nunique())

#category variable
category_var=["sex",'cp',"fbs","exng","restecg","thall","caa","slp",'output']
category_df=df[category_var]

#Continuous variable
continuous_var=["age","trtbps","chol","thalachh","oldpeak"]
continuous_df=df[continuous_var]

#encoding the categorical columns
df = pd.get_dummies(df, columns = category_var[:-1], drop_first = True)

#extract the labels from the data
y=df['output']
X=df.drop('output', axis=1)

#normalize the data
X[continuous_var]=normalize(X[continuous_var])

# divide into train and 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
print("done")
