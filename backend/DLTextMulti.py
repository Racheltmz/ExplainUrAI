# Import Libraries
import numpy as np
from DLText import DLText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Deep Learning Text Tasks (Multi-class or Multi-label)
class DLTextMulti(DLText):
    def __init__(self, model, text, classes, NUM_FEATURES, MAX_NUM, MAX_SEQ):
        super().__init__(model, text, classes, NUM_FEATURES, MAX_NUM, MAX_SEQ)

    # Function for model prediction
    def model_pred(self, text):
        tokenizer = Tokenizer(num_words=self.MAX_NUM, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.ds[self.ds_text_field].values)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQ)
        return self.model.predict(padded)

    def multi_label_explainer(self, labels):
        # Get the prediction for each label
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
        
            super().generate_report(class_names, classifier_fn)