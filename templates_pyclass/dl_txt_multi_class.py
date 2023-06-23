# Import class
import sys
sys.path.insert(0, '../scripts/')
from deep_learning_text_multi import DLTextMulti

'''
Inputs
'''
# Model
MODEL_FILE = 'topics.h5'
model = f'../models/{MODEL_FILE}'

# Sample text
text = "We present novel understandings of the Gamma-Poisson (GaP) model, a probabilistic matrix factorization"

# Dataset
df = '../dataset/topics.csv'
df_text_field = 'text'
classes = ['Computer Science', 'Physics', 'Mathematics', 'Statistics']

# Explainability
NUM_FEATURES = 6

# Tokenizer
MAX_NB_WORDS = 25000
MAX_SEQUENCE_LENGTH = 250

# # Generate report (does not require tokenization)
# dl_textmulti_xai = DLTextMulti(
#     model, 
#     text, 
#     classes)
# dl_textmulti_xai.generate_report()

# Generate report (require tokenization)
dl_textmulti_xai = DLTextMulti(
    model, 
    text, 
    classes, 
    NUM_FEATURES=NUM_FEATURES, 
    need_process=True, 
    ds=df, 
    ds_text_field=df_text_field, 
    MAX_NUM=MAX_NB_WORDS, 
    MAX_SEQ=MAX_SEQUENCE_LENGTH)
dl_textmulti_xai.generate_report()