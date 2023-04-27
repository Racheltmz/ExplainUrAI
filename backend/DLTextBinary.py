# Import Libraries
from DLText import DLText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Deep Learning Text Tasks (Binary)
class DLTextBinary(DLText):
    def __init__(self, model, text, classes, NUM_FEATURES=5, NUM_SAMPLES=1000, need_process=False, ds=None, ds_text_field=None, MAX_NUM=None, MAX_SEQ=None):
        super().__init__(model, text, classes, NUM_FEATURES, NUM_SAMPLES, need_process, ds, ds_text_field, MAX_NUM, MAX_SEQ)
    
    # Function for model prediction
    def model_pred(self, text):
        tokenizer = Tokenizer(num_words=self.MAX_NUM, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(self.ds[self.ds_text_field].values)
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(seq, maxlen=self.MAX_SEQ)
        return self.model.predict(padded)

    def generate_report(self):
        html_filename = f'{super().get_report_convention}dl_textbinary_{super().get_current_epoch}.html'
        # # Get explanation
        # fig = super().get_plots(self.classes, self.model.predict)
        # # Write to a html file
        # fig.write_html(html_filename)
        # Get explanation
        exp = super().get_explanation(self.classes, self.model.predict)
        # Write to a html file
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(exp.as_html())
        f.close()

'''
Inputs
'''
model = '../models/content/Sentiment'
text = "Beware of counterfeits trying to sell fake masks at cheap prices. Lets defeat coronavirus threat, #Covid_19 collectively. #BeSafe #BeACascader #CoronavirusReachesDelhi #coronavirusindia"
classes = ['Negative', 'Positive']
dl_textbin_xai = DLTextBinary(model, text, classes, NUM_FEATURES=6)
dl_textbin_xai.generate_report()