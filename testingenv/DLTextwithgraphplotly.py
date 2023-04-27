# Import Libraries
import numpy as np
import pandas as pd
import plotly.express as px
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from XAI import XAI

# Deep Learning Text Tasks
class DLText(XAI):
    def __init__(self, model, text, classes, NUM_FEATURES=5, NUM_SAMPLES=1000, need_process=False, ds=None, ds_text_field=None, MAX_NUM=None, MAX_SEQ=None):
        super().__init__()
        self.model = load_model(model) # Keras model instance
        self.text = text # String
        self.classes = classes # List
        self.NUM_FEATURES = NUM_FEATURES # Integer
        self.NUM_SAMPLES = NUM_SAMPLES # Integer
        # If the model does not have a layer to tokenize the text
        self.need_process = need_process # Boolean
        self.ds = ds # Dataframe
        self.ds_text_field = ds_text_field # String
        self.MAX_NUM = MAX_NUM # Integer
        self.MAX_SEQ = MAX_SEQ # Integer

    # Change the number of samples parameter if the sentence is shorter than the num of features
    def fix_num_samples(self):
        sentence_len = len(self.text)
        if sentence_len < self.NUM_FEATURES:
            self.NUM_FEATURES = sentence_len

    # Get explanation from LimeTextexplainer
    def get_explanation(self, class_names, pred_fn):
        self.fix_num_samples()
        # Initialise LIME explainer
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(self.text, pred_fn, num_features=self.NUM_FEATURES, num_samples=self.NUM_SAMPLES)
        # # Get list of tuples of the feature mapped to the weight
        # df_weights = pd.DataFrame(exp.as_list(), columns=['Words', 'Weights'])
        # # Convert weights to absolute value
        # df_weights['Weights_abs'] = df_weights['Weights'].abs()
        # # Round off weights to 2dp
        # df_weights['Weights'] = df_weights['Weights'].round(2)
        # # Differentiate positives and negatives 
        # df_weights['Color'] = np.where(df_weights['Weights'] < 0, 'red', 'green')
        # # Model's prediction probabilities
        # fig = px.bar(df_weights, 
        #             x='Weights', 
        #             y='Words', 
        #             text_auto=True,
        #             category_orders = {
        #                 'Words': df_weights.sort_values('Weights_abs', ascending=False)['Words'].values
        #             })
        # fig.update_traces(marker_color=df_weights['Color'])
        # fig.update_layout(
        #     title = {
        #         'text': f"<span style='color:red'>{class_names[0]}</span> vs <span style='color:green'>{class_names[1]}</span>",
        #         'y': 0.95,
        #         'x': 0.5,
        #         'xanchor': 'center',
        #         'yanchor': 'top'
        #     },
        #     font = {
        #         'size': 20
        #     }
        # )
        return exp
