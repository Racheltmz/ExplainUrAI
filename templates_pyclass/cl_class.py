# Import class
import sys
sys.path.insert(0, '../scripts/')
from classical_learning import CLTasks

# Import libraries
import pandas as pd

'''
Inputs
'''
# Model
# Either process the dataset (test set) beforehand or add a pipeline with a preprocesser
model = 'mushroom_dt'

# Test set
X_test = '../dataset/Xtestm.csv'
y_test = '../dataset/ytestm.csv'

# Features and Targets
columns = pd.Series(['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 
                     'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                     'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 
                     'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
target_names=['Poisonous', 'Edible']

# Selected graphs and sample indexes
selected_all = True
selected_graphs = [
    'global_importance',
    'local_importance',
    'permutation_importance',
    'local_contribution',
    'feature_contribution',
    'stability',
    'compacity',
]
sample_indexes = [1, 2]

# Generate report
cl_xai = CLTasks('classification', model, X_test, y_test, selected_all, selected_graphs, target_names, sample_indexes, scoring='roc_auc_ovr_weighted', feature='habitat_w')
cl_xai.generate_report()