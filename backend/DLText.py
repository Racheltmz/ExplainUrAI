# Import Libraries
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model

# Import classes
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

    # Get explanation from LimeTextExplainer
    def get_explanation(self, class_names, pred_fn):
        self.fix_num_samples()
        # Initialise LIME explainer
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(self.text, pred_fn, num_features=self.NUM_FEATURES, num_samples=self.NUM_SAMPLES)
        return exp
