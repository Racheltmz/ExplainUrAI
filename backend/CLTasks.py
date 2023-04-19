# Import Libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from CLELI5 import CLELI5
from CLShapash import CLShapash
from CLDalex import CLDalex

'''
Task: organise class for cl
'''
# Classical Learning Classification and Regression Tasks
class CLClassification():
    def __init__(self, task, model, X_test, y_test, feature_names=None, target_names=None, sample_indexes=None, scoring=None, TOP_FEATURES=5, 
                 feature=None, dist_graph='boself.sh_expot', LOCAL_MAX_FEATURES=3, COMPACITY_NB_FEATURES=5, xai_type='shap'):
        # General
        self.task = task # String (Regression or Classification)
        self.model = pickle.load(open(f'../models/{model}.pkl', 'rb')) # pickle file
        self.X_test = pd.read_csv(X_test) # Dataframe
        self.y_test = pd.Series(pd.read_csv(y_test), dtype=int) # Series
        self.feature_names = feature_names # List
        self.target_names = target_names # List
        self.sample_indexes = sample_indexes # List
        
        # ELI5
        self.scoring = scoring # String
        self.TOP_FEATURES = TOP_FEATURES # Integer

        # Shapash
        self.feature = feature # String
        self.dist_graph = dist_graph # String
        self.LOCAL_MAX_FEATURES = LOCAL_MAX_FEATURES # Integer
        self.COMPACITY_NB_FEATURES = COMPACITY_NB_FEATURES # Integer

        # Dalex
        self.xai_type = xai_type # String

    # Call for plots in the different models and generate a report
    def generate_report(self):
        pass

# Either process the dataset (test set) beforehand or add a pipeline with a preprocesser
model = 'mushroom_dt'

# Get dataset and process data
columns = pd.Series(['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 
                     'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                     'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 
                     'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
df_mushroom = pd.read_csv('../dataset/agaricus-lepiota.data', names = columns)
sample_indexes = [1, 2]
# # get the most frequent value
# mostfreqval = df_mushroom['stalk-root'].mode()
# # replace missing values
# df_mushroom['stalk-root'] = df_mushroom['stalk-root'].replace(['?'], mostfreqval)
# mushroom_features = df_mushroom.drop(['class', 'veil-type'], axis = 1).columns.values

# X = pd.get_dummies(df_mushroom[mushroom_features])
# y = pd.get_dummies(df_mushroom['class'])['e']

# def index_reset(dfs):
#     reseted_dfs = []
#     for df in dfs:
#         reseted_dfs.append(df.reset_index(drop=True))
#     return reseted_dfs

# # Still need the datasets whether you want to process or not, if use pipeline no need to process
# X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y, test_size = 0.3, random_state = 42)
# X_train_m, X_test_m, y_train_m, y_test_m = index_reset([X_train_m, X_test_m, y_train_m, y_test_m])
# X_test_m.to_csv('../dataset/Xtestm.csv', index=False)
# y_test_m.to_csv('../dataset/ytestm.csv', index=False)

X_test = '../dataset/Xtestm.csv'
y_test = '../dataset/ytestm.csv'
CLClassification(model, X_test, y_test, columns, sample_indexes)