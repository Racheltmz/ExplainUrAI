# Import Libraries
import pandas as pd
import pickle
from XAI import XAI
import eli5
from eli5.sklearn import PermutationImportance
import dalex as dx
from shapash.explainer.smart_explainer import SmartExplainer

'''
Work on Regression model
Work on pipelines
Work on fixing Dalex issue, unable to convert to html, plot is none (but launches as browser)
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
        self.y_test = pd.read_csv(y_test) # Series
        self.feature_names = list(self.X_test.columns.values) # List
        self.target_names = target_names # List
        self.sample_indexes = sample_indexes # List
        self.html_explanations = [] # List of all HTML explanations

        # ELI5
        self.scoring = scoring # String
        self.TOP_FEATURES = TOP_FEATURES # Integer

        # Dalex
        self.xai_type = xai_type # String
        # Initialise Dalex explainer
        self.dx_exp = dx.Explainer(self.model, self.X_test, self.y_test, verbose=False)

        # Shapash
        self.feature = feature # String
        self.dist_graph = dist_graph # String
        self.LOCAL_MAX_FEATURES = LOCAL_MAX_FEATURES # Integer
        self.COMPACITY_NB_FEATURES = COMPACITY_NB_FEATURES # Integer
        self.label_dict = dict([(idx, lbl) for idx, lbl in enumerate(self.target_names)]) # Dictionary
        self.feature_dict = dict([(idx, lbl) for idx, lbl in enumerate(self.feature_names)]) # Dictionary
        # Initialise SmartExplainer
        self.sh_exp = SmartExplainer(model=self.model, label_dict=self.label_dict, features_dict=self.feature_dict)
        self.sh_exp.compile(x=self.X_test, y_target=self.y_test)

        # Dictionary of graph names to graph functions
        self.cl_graphs = {
            'global_importance': self.global_importance(),
            'local_importance': self.local_importance(),
            'permutation_importance': self.permutation_importance(),
            # 'model_eval': self.model_eval(),
            'local_contribution': self.local_contribution(),
            'feature_contribution': self.feature_contribution(),
            'stability': self.stability(),
            'compacity': self.compacity(),
        }

    # Global feature importance
    def global_importance(self):
        # Add header
        self.html_explanations.append('<h1>Global Feature Importance</h1>')

        # ELI5
        eli5_gi = eli5.explain_weights(self.model, feature_names=self.feature_names, target_names=self.target_names, top=self.TOP_FEATURES)
        self.html_explanations.append(eli5.format_as_html(eli5_gi, show=('method', 'description', 'transition_features', 'targets', 'feature_importances')))

        # # Dalex
        # dx_gi = self.dx_exp.model_parts().plot()
        # self.html_explanations.append(dx_gi.to_html())

        # Shapash
        sh_gi = self.sh_exp.plot.features_importance(max_features=self.TOP_FEATURES)
        self.html_explanations.append(sh_gi.to_html())

    # Local feature importance
    def local_importance(self):
        # Add header
        self.html_explanations.append('<h1>Local Feature Importance</h1>')

        for index in self.sample_indexes:
            # ELI5
            eli5_li = eli5.explain_prediction(self.model, self.X_test.iloc[index], feature_names=self.feature_names, target_names=self.target_names)
            self.html_explanations.append(eli5.format_as_html(eli5_li))

            # # Dalex
            # dx_li = self.dx_exp.predict_parts(self.X_test.iloc[index], type=self.xai_type).plot()
            # self.html_explanations.append(dx_li.to_html())

            # Shapash
            sh_li = self.sh_exp.plot.local_plot(index=index)
            self.html_explanations.append(sh_li.to_html())

    # Permutation Based Importance
    def permutation_importance(self):
        # Add header
        self.html_explanations.append('<h1>Permutation Based Importance</h1>')
        perm = PermutationImportance(self.model, scoring=self.scoring)
        perm.fit(self.X_test, self.y_test)
        eli5_exp = eli5.explain_weights(perm, feature_names=self.feature_names, top=self.TOP_FEATURES)
        self.html_explanations.append(eli5.format_as_html(eli5_exp))

    # # Model Performance
    # def model_eval(self):
    #     me_dx = self.dx_exp.model_performance()
    #     self.html_explanations.append(me_dx.to_html())

    # Compare contribution values with the neighbours by analysing local neighbourhood of the instance
    def local_contribution(self):
        for index in self.sample_indexes:
            # Add header
            self.html_explanations.append('<h1>Local Contribution</h1>')
            lc_sh = self.sh_exp.plot.local_neighbors_plot(index=index, max_features=self.LOCAL_MAX_FEATURES)
            self.html_explanations.append(lc_sh.to_html())

    # Contribution for each column to the predictions
    # Displays a plotly scatter/violin plot of a selected feature
    def feature_contribution(self):
        # WARNING: COULD HV SOME ERRORS WITH THE DICTIONARY (FEATURES OR LABEL)
        if self.task == 'Classification':
            # Add header
            self.html_explanations.append('<h1>Feature Contribution</h1>')
            fc_sh = self.sh_exp.plot.contribution_plot(list(self.feature_dict.values()).index(self.feature))
            self.html_explanations.append(fc_sh.to_html())

    # Stability Plot
    def stability(self):
        # Add header
        self.html_explanations.append('<h1>Stability Plot</h1>')
        st_sh = self.sh_exp.plot.stability_plot()
        self.html_explanations.append(st_sh.to_html())
        if self.task == 'Classification':
            # Stability plot in more detail with distribution
            self.html_explanations.append('<h2>Stability Plot with distribution</h2>')
            st_sh_dist = self.sh_exp.plot.stability_plot(distribution=self.dist_graph)
            self.html_explanations.append(st_sh_dist.to_html())

    # Compacity Plot
    def compacity(self):
        # Add header
        self.html_explanations.append('<h1>Compacity Plot</h1>')
        cp_sh = self.sh_exp.plot.compacity_plot(nb_features=self.COMPACITY_NB_FEATURES)
        self.html_explanations.append(cp_sh.to_html())
    
    # Call for plots in the different models
    def plot_visualisations(self):
        if self.selected_all:
            for keys_func in self.cl_graphs.keys():
                self.cl_graphs[keys_func]()

        else:
            for choice in self.selected_graphs:
                self.cl_graphs[choice]()
    
    # Generate report
    def generate_report(self):
        html_filename = f'{super().get_report_convention}cl_{self.task.lower()}_{super().get_current_epoch}.html'
        html_report = ''.join(self.html_explanations)
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_report)

'''
Inputs
'''
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

X_test = '../dataset/Xtestm.csv'
y_test = '../dataset/ytestm.csv'
selected_all = True
selected_graphs = [
    'global_importance',
    'local_importance',
    'permutation_importance',
    # 'model_eval',
    'local_contribution',
    'feature_contribution',
    'stability',
    'compacity',
]
cl_xai = CLTasks('classification', model, X_test, y_test, selected_all, selected_graphs, target_names, sample_indexes, scoring='roc_auc_ovr_weighted', feature='condition')
cl_xai.generate_report()