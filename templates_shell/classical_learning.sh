python '../scripts/classical_learning.py' \
'classification' \
'mushroom_dt' \
'../dataset/Xtestm.csv' \
'../dataset/ytestm.csv' \
True \
--selected_graphs 'global_importance' 'local_importance' 'permutation_importance' 'model_eval' 'local_contribution' 'feature_contribution' 'stability' 'compacity' \
--target_names 'Poisonous' 'Edible' \
--sample_indexes 1 2 \
--scoring 'roc_auc_ovr_weighted' \
--feature 'habitat_w' \