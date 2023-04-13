# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from skimage.segmentation import mark_boundaries

# Deep Learning Text Tasks
class DLText():
    def __init__(self, model, text, classes, ds=None, ds_text_field=None, NUM_FEATURES=5, NUM_SAMPLES=1000, MAX_NUM=None, MAX_SEQ=None):
        self.model = model
        self.text = text
        self.classes = classes
        self.ds = ds
        self.ds_text_field = ds_text_field
        self.NUM_FEATURES = NUM_FEATURES
        self.NUM_SAMPLES = NUM_SAMPLES
        self.MAX_NUM = MAX_NUM
        self.MAX_SEQ = MAX_SEQ
    
    def generate_report(self, class_names, pred_fn):
        # Initialise LIME explainer
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(self.text, pred_fn, num_features=self.NUM_FEATURES, num_samples=self.NUM_SAMPLES)
        exp.as_list()
        # Model's prediction probabilities
        exp.show_in_notebook(text=True)