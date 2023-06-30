---
sidebar_position: 2
---

# Arguments and Methods

The `DLImage` class can be found under `scripts/deep_learning_image.py` in the GitHub repository.

## Arguments

| Name   | Description                          |
| ------ | ------------------------------------ |
| model  | Image classification model file path |
| imgs   | Filepath of all images to explain    |

## Methods

| Name              | Description                                   |
| ----------------- | --------------------------------------------- |
| lime_explainer    | Explain the model using LIME and output the weights of how important a feature is, the image, and the mask.             |
| heatmap_explainer | Display contributions in a heatmap.           |
| generate_report   | Generate a report to show the visualisations. |
