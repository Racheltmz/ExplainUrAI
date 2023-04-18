# Import Libraries
import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from DLText import DLText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Deep Learning Text Tasks (Multi-class or Multi-label)
class DLTextMulti(DLText):
    def __init__(self, model, text, classes, NUM_FEATURES=5, NUM_SAMPLES=1000, need_process=False, ds=None, ds_text_field=None, MAX_NUM=None, MAX_SEQ=None):
        super().__init__(model, text, classes, NUM_FEATURES, NUM_SAMPLES, need_process, ds, ds_text_field, MAX_NUM, MAX_SEQ)

    # Function for model prediction
    def model_pred(self, text):
        tokenizer = Tokenizer(num_words=self.MAX_NUM, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.ds[self.ds_text_field].values)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQ)
        return self.model.predict(padded)

    # Add multiple plotly graphs to a html file
    def figures_to_html(self, figs, filename='report.html'):
        with open(filename, 'w', encoding="utf-8") as dashboard:
            dashboard.write('<html><head></head><body>' + '\n')
            for fig in figs:
                inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
            dashboard.write('</body></html>' + '\n')

    # Add multiple explanation visualisations to a html file
    def explanations_to_html(self, exps, filename='report.html'):
        # Iterate through each explanation
        for (idx, exp) in enumerate(exps):
            # If it is the first record, parse the html page with BeautifulSoup to get the body tag
            if idx == 0:
                # Get explanation as a html file
                html = exp.as_html()
                soup = BeautifulSoup(html, 'html.parser')
                old_body = soup.find('body')
            else:
                html_to_extract = exp.as_html()
                soup_to_extract = BeautifulSoup(html_to_extract, 'html.parser')
                visualisation_tags = soup_to_extract.find('body').find_all(recursive=False)
                for tag in visualisation_tags:
                    old_body.append(tag)

        with open(filename, 'w', encoding='utf-8') as report:
            report.write(soup.prettify())
        report.close()

    def generate_report(self):
        html_filename = f'./reports/explainurai_dl_textmulti_{super().get_current_epoch}.html'
        # Instantiate list to store all explanations
        # figs = []
        exps = []
        # Get the prediction for each label
        labels = self.classes
        for label in labels:
            class_names = ['Not ' + label, label]

            def classifier_fn(input_text):
                label_index = labels.index(label)
                # pick the corresponding output node 
                def lime_explainer_pipeline(input_text):
                    predict_probs = self.model_pred(input_text)
                    prob_true = predict_probs[:, label_index]
                    result = np.transpose(np.vstack(([1-prob_true, prob_true])))  
                    result = result.reshape(-1, 2)
                    return result
                return lime_explainer_pipeline(input_text)

            # # Get figure and add it to the list of figures
            # fig = super().get_plots(class_names, classifier_fn)
            # figs.append(fig)
            exp = super().get_explanation(class_names, classifier_fn)
            exps.append(exp)
        self.explanations_to_html(exps, html_filename)
        # self.figures_to_html(figs, html_filename)

# The maximum number of words to consider as features for the tokenizer
MAX_NB_WORDS = 25000
# Max number of words in each comment
MAX_SEQUENCE_LENGTH = 250

# Multi-Label Text Classification
MODEL_NAME = 'topics'
text_model = f'../models/{MODEL_NAME}.h5'

# Get dataset
df = pd.read_csv('../dataset/topics.csv')
df_text_field = 'text'
# Sample text
text = "We present novel understandings of the Gamma-Poisson (GaP) model, a probabilistic matrix factorization"

# Define labels
classes = list(df.columns[1:])

# Generate report
report = DLTextMulti(text_model, text, classes, NUM_FEATURES=6, need_process=True, 
                     ds=df, ds_text_field=df_text_field, MAX_NUM=MAX_NB_WORDS, MAX_SEQ=MAX_SEQUENCE_LENGTH).generate_report()