# Import Libraries
import eli5
from eli5.sklearn import PermutationImportance

# Classical Learning with ELI5
class CLELI5():
    def __init__(self, model, feature_names, target_names, X_test, y_test, sample_indexes, scoring=None, TOP_FEATURES=5 ):
        self.model = model # pickle file
        self.feature_names = feature_names # List
        self.target_names = target_names # List
        self.X_test = X_test # Dataframe
        self.y_test = y_test # Series
        self.sample_indexes = sample_indexes # List
        self.scoring = scoring # String
        self.TOP_FEATURES = TOP_FEATURES # Integer

    def global_importance(self):
        # Global feature importance
        eli5.show_weights(self.model, feature_names=self.feature_names, target_names=self.target_names, top=self.TOP_FEATURES)

    def local_importance(self):
        for index in self.sample_indexes:
            eli5.show_prediction(self.model, self.X_test.iloc[index], feature_names=self.feature_names, 
                target_names=self.target_names, show_feature_values=True)
            
    def permutation_importance(self):
        # Permutation Based Importance
        perm = PermutationImportance(self.model, scoring=self.scoring)
        perm.fit(self.X_test, self.y_test)
        eli5.show_weights(perm, feature_names=self.feature_names, top=self.TOP_FEATURES)