# Import Libraries
import dalex as dx

# Classical Learning with Dalex
class CLDalex():
    def __init__(self, model, X_test, y_test, sample_indexes, xai_type='shap'):
        self.model = model # pickle file
        self.X_test = X_test # Dataframe
        self.y_test = y_test # Series
        self.sample_indexes = sample_indexes # List
        self.xai_type = xai_type # String
        # Initialise Dalex explainer
        self.dx_exp = dx.Explainer(self.model, self.X_test, self.y_test)

    # Model Performance
    def model_eval(self):
        self.dx_exp.model_performance()
    
    # Global feature importance
    def global_importance(self):
        self.dx_exp.model_parts().plot()
    
    # Local feature importance
    def local_importance(self):
        for index in self.sample_indexes:
            self.dx_exp.predict_parts(self.X_test.iloc[index], type=self.xai_type).plot()