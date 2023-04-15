# Import Libraries
from DLText import DLText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate
from utils.PlotlyGraph import PlotlyGraph

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
        pdf_filename = f'./reports/explainurai_dl_textbinary_{super().get_current_epoch}.pdf'
        print(pdf_filename)
        # Get figure
        fig = super().get_plots(self.classes, self.model.predict)
        # Create report
        doc = SimpleDocTemplate(pdf_filename, pagesize=landscape(letter))

        # Add the Plotly graph to the report
        plotly_graph = PlotlyGraph(fig, width=500, height=400)
        doc.build([plotly_graph])

model = '../models/content/Sentiment'
text = "Beware of counterfeits trying to sell fake masks at cheap prices. Let's defeat coronavirus threat, #Covid_19 collectively. #BeSafe #BeACascader #CoronavirusReachesDelhi #coronavirusindia"
classes = ['Negative', 'Positive']
report = DLTextBinary(model, text, classes, NUM_FEATURES=6).generate_report()