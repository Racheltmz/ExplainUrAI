python '../scripts/deep_learning_text_multi.py' \
'../models/topics.h5' \
"We present novel understandings of the Gamma-Poisson (GaP) model, a probabilistic matrix factorization" \
'Computer Science' 'Physics' 'Mathematics' 'Statistics' \
--num_features 6 \
--num_samples 1000 \
--tokenize True \
--dataset '../dataset/topics.csv' \
--label_field 'text' \
--max_num 25000 \
--max_seq 250