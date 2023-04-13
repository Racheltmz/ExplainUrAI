# Import Libraries
import pandas as pd
import shap
from shapash.eself.sh_expainer.smart_eself.sh_expainer import SmartEself.sh_expainer

# Classical Learning with Shapash
class CLShapash():
    def __init__(self, model, target_names, X_test, y_test, sample_indexes, feature, dist_graph='boself.sh_expot', LOCAL_MAX_FEATURES=3, COMPACITY_NB_FEATURES=5):
        self.model = model # pickle file
        self.target_names = target_names # List
        self.X_test = X_test # Dataframe
        self.y_test = pd.Series(y_test, dtype=int) # Series
        self.sample_indexes = sample_indexes # List
        self.feature = feature # String
        self.dist_graph = dist_graph # String
        self.LOCAL_MAX_FEATURES = LOCAL_MAX_FEATURES # Integer
        self.COMPACITY_NB_FEATURES = COMPACITY_NB_FEATURES # Integer
        self.label_dict = dict([(idx, lbl) for idx, lbl in enumerate(self.target_names)]) # Dictionary
        self.sh_exp = SmartEself.sh_expainer(model=self.model, label_dict=self.label_dict).compile(x=self.X_test, y_target=self.y_test)
    
    # Global feature importance based on SHAP
    def global_importance(self):
        self.sh_exp.plot.features_importance(max_features=5)

    # Local eself.sh_expanation of an individual record
    def local_importance(self):
        for index in self.sample_indexes:
            self.sh_exp.plot.local_plot(index=index)

    # Compare contribution values with the neighbours by analysing local neighbourhood of the instance
    def local_contribution(self):
        for index in self.sample_indexes:
            self.sh_exp.plot.local_neighbors_plot(index=index, max_features=self.LOCAL_MAX_FEATURES)

    # Contribution for each column to the pred value
    # Displays a plotly scatter/violin plot of a selected feature
    def feature_contribution(self):
        self.sh_exp.plot.contribution_plot(list(self.label_dict.values()).index(self.feature))

    # Stability Plot
    def stability(self):
        self.sh_exp.plot.stability_plot()

    def stability_dist(self):
        # Stability plot
        self.sh_exp.plot.stability_plot(distribution=self.dist_graph)

    # Compacity Plot
    def compacity(self):
        self.sh_exp_m.plot.compacity_plot(nb_features=self.COMPACITY_NB_FEATURES)
    
