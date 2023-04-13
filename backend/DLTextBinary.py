# Import Libraries
from DLText import DLText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Deep Learning Text Tasks (Binary)
class DLTextBinary(DLText):
    def __init__(self, model, text, classes, NUM_FEATURES, MAX_NUM, MAX_SEQ):
        super().__init__(model, text, classes, NUM_FEATURES, MAX_NUM, MAX_SEQ)
    
    # Function for model prediction
    def model_pred(self, text):
        tokenizer = Tokenizer(num_words=self.MAX_NUM, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.ds[self.ds_text_field].values)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQ)
        return self.model.predict(padded)

    def binary_explanation(self):
        super().generate_report(self.classes, self.model.predict)