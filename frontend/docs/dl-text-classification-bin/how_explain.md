---
sidebar_position: 2
---

# Arguments and Methods

The `DLText` class can be found under `scripts/deep_learning_text_bin.py` in the GitHub repository.

## Arguments (add data type)

| Name         | Description                          |
| ------------ | ------------------------------------ |
| model        | Image classification model file path |
| text         | Text to explain                      |
| classes      | Classes in this classification task  |
| num_features | Maximum number of features to present in the explanation |
| num_samples  | Size of the neighborhood to learn the linear model |
| tokenize     | set to True if model does not have a tokenizer layer |
| dataset      | Dataset filepath                     |
| label_field  | Label field name in dataset file     |
| max_num      | Maximum number for tokenization      |
| max_seq      | Maximum sequence for padding         |

## Methods

| Name              | Description                                   |
| ----------------- | --------------------------------------------- |
| fix_num_samples    | Change the number of samples parameter if the sentence is shorter than the num of features |
| model_pred | Prediction method for models that do not have a tokenizer layer within its architecture |
| generate_report   | Generate a report to show the visualisations. |
