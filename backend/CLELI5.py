# Import Libraries
import eli5
from eli5.sklearn import PermutationImportance

# Classical Learning with ELI5
class CLELI5(CLTasks):
    def __init__(self, model, feature_names, target_names, X_test, y_test, sample_indexes, scoring=None, TOP_FEATURES=5):
        super()

    # Global feature importance
    def global_importance(self):
        eli5.show_weights(self.model, feature_names=self.feature_names, target_names=self.target_names, top=self.TOP_FEATURES)

    # Local feature importance
    def local_importance(self):
        for index in self.sample_indexes:
            eli5.show_prediction(self.model, self.X_test.iloc[index], feature_names=self.feature_names, 
                target_names=self.target_names, show_feature_values=True)
            
    # Permutation Based Importance
    def permutation_importance(self):
        perm = PermutationImportance(self.model, scoring=self.scoring)
        perm.fit(self.X_test, self.y_test)
        eli5.show_weights(perm, feature_names=self.feature_names, top=self.TOP_FEATURES)