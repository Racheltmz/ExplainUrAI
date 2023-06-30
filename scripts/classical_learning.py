# Import classes
from xai_base import XAI

# Import Libraries
import argparse
import pickle
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from shapash.explainer.smart_explainer import SmartExplainer

'''
To fix:
Format report
Work on Regression model
Work on accomodating to pipelines
'''

# Classical Learning Classification and Regression Tasks
class CLTasks(XAI):
    def __init__(self, task, model, X_test, y_test, selected_all, selected_graphs=None, target_names=None, sample_indexes=None, scoring='balanced_accuracy', 
                 TOP_FEATURES=5, feature=None, LOCAL_MAX_FEATURES=3, COMPACITY_NB_FEATURES=5):
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

        # Shapash
        self.feature = feature # String
        self.dist_graph = 'boself.sh_expot' # String
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
        if self.task == 'classification':
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
        if self.task == 'classification':
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
        # Remove extra lines
        html_report_cleaned = '\n'.join(line.strip() for line in html_report.split('\n') if line.strip())
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_report_cleaned)
        print(f'{self.task} report generated: {html_filename}')

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description='Explain Predictions of Regression or Classification Model')
    parser.add_argument(
        '--filetype', default='pdf', choices=['pdf', 'html'], help='report file type', type=str
    )
    parser.add_argument(
        'cl_task', default='', choices=['regression', 'classification'], help='classification task: regression or classification', type=str
    )
    parser.add_argument(
        'model', default='', help='path to model file (h5 format)', type=str
    )
    parser.add_argument(
        'X_test', default='', help='csv test file containing features', type=str
    )
    parser.add_argument(
        'y_test', default='', help='csv file containing features', type=str
    )
    parser.add_argument(
        'selected_all', default='', help='display all graphs', type=bool
    )
    parser.add_argument(
        '--selected_graphs', default=None, help='chosen graphs to display', type=str, nargs='+'
    )
    parser.add_argument(
        '--target_names', default=None, help='label names', type=str, nargs='+'
    )
    parser.add_argument(
        '--sample_indexes', default=None, help='indexes of sample records to explain', type=int, nargs='+'
    )
    parser.add_argument(
        '--scoring', default='balanced_accuracy', help='scoring metric', type=str
    )
    parser.add_argument(
        '--top_features', default=5, help='top number of features to explain', type=int
    )
    parser.add_argument(
        '--feature', default=None, help='feature to explain', type=str
    )
    parser.add_argument(
        '--local_max_features', default=3, help='local maximum features', type=int
    )
    parser.add_argument(
        '--compacity_nb_features', default=5, help='compacity nb features', type=int
    )
    args = parser.parse_args()

    # Generate report
    cl_xai = CLTasks(
        args.cl_task,
        args.model,
        args.X_test,
        args.y_test,
        args.selected_all,
        args.selected_graphs,
        args.target_names,
        args.sample_indexes,
        args.scoring,
        args.top_features,
        args.feature,
        args.local_max_features,
        args.compacity_nb_features)
    cl_xai.generate_report()

# # CLI approach is recommended only if there are a few graphs you would like to select and if there is a small number of features in your dataset# '''
# # Error msg to inform them to install the sklearn ver they need?