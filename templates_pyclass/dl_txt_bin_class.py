# Import class
import sys
sys.path.insert(0, '../scripts/')
from deep_learning_text_bin import DLTextBinary

'''
Inputs
'''
# Model
MODEL_FILE = 'content/Sentiment'
model = f'../models/{MODEL_FILE}'

# Sample text
text = "Beware of counterfeits trying to sell fake masks at cheap prices. Lets defeat coronavirus threat, #Covid_19 collectively. #BeSafe #BeACascader #CoronavirusReachesDelhi #coronavirusindia"

# Dataset
# df = '../dataset'
# df_text_field = ''
classes = ['Negative', 'Positive']

# Explainability
NUM_FEATURES = 6

# Tokenizer
# MAX_NB_WORDS = 25000
# MAX_SEQUENCE_LENGTH = 250

# Generate report (does not require tokenization)
dl_textbin_xai = DLTextBinary(
    model, 
    text, 
    classes, 
    NUM_FEATURES=NUM_FEATURES)
dl_textbin_xai.generate_report()

# Generate report (require tokenization)
# dl_textbin_xai = DLTextBinary(
#     model, 
#     text, 
#     classes, 
#     NUM_FEATURES=NUM_FEATURES, 
#     need_process=True, 
#     ds=df, 
#     ds_text_field=df_text_field, 
#     MAX_NUM=MAX_NB_WORDS, 
#     MAX_SEQ=MAX_SEQUENCE_LENGTH)
# dl_textbin_xai.generate_report()