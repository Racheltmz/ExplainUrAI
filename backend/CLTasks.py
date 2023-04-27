# Import Libraries
import pandas as pd
import pickle
from XAI import XAI
import eli5
from eli5.sklearn import PermutationImportance
import dalex as dx
from shapash.explainer.smart_explainer import SmartExplainer

'''
Task: organise class for cl
'''
# Classical Learning Classification and Regression Tasks
class CLTasks(XAI):
    def __init__(self, task, model, X_test, y_test, selected_all, selected_graphs=None, target_names=None, sample_indexes=None, scoring='balanced_accuracy', 
                 TOP_FEATURES=5, feature=None, dist_graph='boself.sh_expot', LOCAL_MAX_FEATURES=3, COMPACITY_NB_FEATURES=5, xai_type='shap'):
        super().__init__()
        # General
        self.task = task # String (Regression or Classification)
        self.selected_all = selected_all # Boolean
        self.selected_graphs = selected_graphs # List
        self.model = pickle.load(open(f'../models/{model}.pkl', 'rb')) # pickle file
        self.X_test = pd.read_csv(X_test) # Dataframe
        self.y_test = pd.Series(pd.read_csv(y_test), dtype=int) # Series
        self.feature_names = list(X_test.columns.values) # List
        self.target_names = target_names # List
        self.sample_indexes = sample_indexes # List
        
        # ELI5
        self.scoring = scoring # String
        self.TOP_FEATURES = TOP_FEATURES # Integer

        # Dalex
        self.xai_type = xai_type # String
        # Initialise Dalex explainer
        self.dx_exp = dx.Explainer(self.model, self.X_test, self.y_test)

        # Shapash
        self.feature = feature # String
        self.dist_graph = dist_graph # String
        self.LOCAL_MAX_FEATURES = LOCAL_MAX_FEATURES # Integer
        self.COMPACITY_NB_FEATURES = COMPACITY_NB_FEATURES # Integer
        self.label_dict = dict([(idx, lbl) for idx, lbl in enumerate(self.target_names)]) # Dictionary
        # Initialise SmartExplainer
        self.sh_exp = SmartExplainer.sh_expainer(model=self.model, label_dict=self.label_dict).compile(x=self.X_test, y_target=self.y_test)

        # Dictionary of graph names to graph functions
        self.cl_graphs = {
            'global_importance': self.global_importance(),
            'local_importance': self.local_importance(),
            'permutation_importance': self.permutation_importance(),
            'model_eval': self.model_eval(),
            'local_contribution': self.local_contribution(),
            'feature_contribution': self.feature_contribution(),
            'stability': self.stability(),
            'compacity': self.compacity(),
        }

        # List of all HTML explanations
        self.html_explanations = []


    # def export_to_html(self, filename):
    #     with open('test.html', 'w', encoding='utf-8') as f:
    #         f.write(eli5.format_as_html(eli5_exp, show=('method', 'description', 'transition_features', 'targets', 'feature_importances')))

    #     with open('test.html', 'r', encoding='utf-8') as f_in:
    #         cleaned_html = list(filter(lambda x: x.strip().replace('\n', ''), f_in.readlines()))
            
    #     with open('test.html', 'w', encoding='utf-8') as f_out:
    #         f_out.writelines(cleaned_html)

    # Global feature importance
    def global_importance(self):
        # ELI5
        eli5_gi = eli5.explain_weights(self.model, feature_names=self.feature_names, target_names=self.target_names, top=self.TOP_FEATURES)
        # with open('test.html', 'w', encoding='utf-8') as f:
        #     f.write(eli5.format_as_html(eli5_exp, show=('method', 'description', 'transition_features', 'targets', 'feature_importances')))
        self.html_explanations.append(eli5.format_as_html(eli5_gi, show=('method', 'description', 'transition_features', 'targets', 'feature_importances')))

        # Dalex
        dx_gi = self.dx_exp.model_parts().plot()
        self.html_explanations.append(dx_gi.to_html())

        # Shapash
        sh_gi = self.sh_exp.plot.features_importance(max_features=self.TOP_FEATURES)
        self.html_explanations.append(sh_gi.to_html())

    # Local feature importance
    def local_importance(self):
        for index in self.sample_indexes:
            # ELI5
            eli5_li = eli5.explain_prediction(self.model, self.X_test.iloc[index], feature_names=self.feature_names, 
                target_names=self.target_names, show_feature_values=True)
            self.html_explanations.append(eli5.format_as_html(eli5_li))

            # Dalex
            dx_li = self.dx_exp.predict_parts(self.X_test.iloc[index], type=self.xai_type).plot()
            self.html_explanations.append(dx_li.to_html())

            # Shapash
            sh_li = self.sh_exp.plot.local_plot(index=index)
            self.html_explanations.append(sh_li.to_html())

    # Permutation Based Importance
    def permutation_importance(self):
        perm = PermutationImportance(self.model, scoring=self.scoring)
        perm.fit(self.X_test, self.y_test)
        eli5_exp = eli5.explain_weights(perm, feature_names=self.feature_names, top=self.TOP_FEATURES)
        self.html_explanations.append(eli5.format_as_html(eli5_exp))

    # Model Performance
    def model_eval(self):
        me_dx = self.dx_exp.model_performance()
        self.html_explanations.append(me_dx.to_html())

    # Compare contribution values with the neighbours by analysing local neighbourhood of the instance
    def local_contribution(self):
        for index in self.sample_indexes:
            lc_sh = self.sh_exp.plot.local_neighbors_plot(index=index, max_features=self.LOCAL_MAX_FEATURES)
            self.html_explanations.append(lc_sh.to_html())

    # Contribution for each column to the predictions
    # Displays a plotly scatter/violin plot of a selected feature
    def feature_contribution(self):
        fc_sh = self.sh_exp.plot.contribution_plot(list(self.label_dict.values()).index(self.feature))
        self.html_explanations.append(fc_sh.to_html())

    # Stability Plot
    def stability(self):
        self.sh_exp.plot.stability_plot()
        if self.task == 'Classification':
            # Stability plot in more detail with distribution
            st_sh = self.sh_exp.plot.stability_plot(distribution=self.dist_graph)
            self.html_explanations.append(st_sh.to_html())

    # Compacity Plot
    def compacity(self):
        cp_sh = self.sh_exp.plot.compacity_plot(nb_features=self.COMPACITY_NB_FEATURES)
        self.html_explanations.append(cp_sh.to_html())
    
    # Call for plots in the different models and generate a report
    def generate_report(self):
        if self.selected_all:
            for keys_func in self.cl_graphs.keys():
                self.cl_graphs[keys_func]()

        else:
            for choice in self.selected_graphs:
                self.cl_graphs[choice]()

# Either process the dataset (test set) beforehand or add a pipeline with a preprocesser
model = 'mushroom_dt'

# Get dataset and process data
columns = pd.Series(['class','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 
                     'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                     'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 
                     'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
df_mushroom = pd.read_csv('../dataset/agaricus-lepiota.data', names = columns)
target_names=['Poisonous', 'Edible']
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
selected_all = True
selected_graphs = [
    'global_importance',
    'local_importance',
    'permutation_importance',
    'model_eval',
    'local_contribution',
    'feature_contribution',
    'stability',
    'compacity',
]
CLTasks('classification', model, X_test, y_test, selected_all, selected_graphs, target_names, sample_indexes, scoring='roc_auc_ovr_weighted')
