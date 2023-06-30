'''
Change font of lime auto (HTML)
'''

# Import classes
from xai_base import XAI

# Import Libraries
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        if ds is not None:
            self.ds = pd.read_csv(ds) # Dataframe
        else:
            self.ds = ds
        self.ds_text_field = ds_text_field # String
        self.MAX_NUM = MAX_NUM # Integer
        self.MAX_SEQ = MAX_SEQ # Integer

    # Change the number of samples parameter if the sentence is shorter than the num of features
    def fix_num_samples(self):
        sentence_len = len(self.text)
        if sentence_len < self.NUM_FEATURES:
            self.NUM_FEATURES = sentence_len

    # For models that do not tokenize the text within its architecture
    def model_pred(self, text):
        tokenizer = Tokenizer(num_words=self.MAX_NUM, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.ds[self.ds_text_field].values)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQ)
        return self.model.predict(padded)

    # Get explanation from LimeTextExplainer
    def get_explanation(self, class_names, tokenize, task_type='', label=None):
        self.fix_num_samples()
        # Tokenize and padding the sequence if the model architecture does not consist a tokenizer layer
        if tokenize:
            # Get the prediction
            def lime_explainer_pipeline(input_text):
                if task_type == 'binary':
                    return self.model_pred(input_text)
                elif task_type == 'multi':
                    # Get label
                    label_index = class_names.index(label)
                    predict_probs = self.model_pred(input_text)
                    prob_true = predict_probs[:, label_index]
                    result = np.transpose(np.vstack(([1-prob_true, prob_true])))  
                    result = result.reshape(-1, 2)
                    return result
            pred_fn = lime_explainer_pipeline
        else:
            pred_fn = self.model.predict

        # Initialise LIME explainer
        if task_type == 'multi':
            class_names = ['Not ' + label, label]
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(self.text, pred_fn, num_features=self.NUM_FEATURES, num_samples=self.NUM_SAMPLES)
        return exp
