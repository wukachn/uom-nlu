---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/username/project_name

---

# Model Card for GROUP_19_NLI_A

This is a classification model that was trained to determine if a given hypothesis is true based on its premise.


## Model Details

### Model Description

This model utilizes a traditional machine learning approach for a natural language inference (NLI) task, determining if a given hypothesis is true based on its premise.
The final model uses an ensemble architecture to combine 3 sub-models which all use TF-IDF embeddings. These models each cast a vote to determine the final classification. The 3 sub-models which make up the final ensemble model are listed below:
- A Logistic Regression Model
- A Random Forest Model
- A Gradient Boosting Model


- **Developed by:** Peter Hamer and Vinayak Singh Bhadoriya
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Ensemble

<!-- (Model Resources) Provide links where applicable. Dont think i need this section-->

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K premise-hypothesis pairs.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
#### The Logistic Regression Model
Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - C: 0.1
      - max_iter: 100000
      - solver: lbfgs

Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: TODO
      - model size: 543 KB

#### The Random Forest Model
Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - max_depth: None
      - max_features: sqrt
      - min_samples_leaf: 2
      - min_samples_split: 10
      - n_estimators: 200

Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: TODO
      - model size: 42.4 MB

#### The Gradient Boosting Model
Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 0.1
      - max_depth: 7
      - n_estimators: 300
      - subsample: 1.0

Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: TODO
      - model size: 749 KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to TODO K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of TODO % and an accuracy of TODO %.

## Technical Specifications

### Hardware (Trained On)

<!-- Do i need minimum RAM and Storage?? -->

      - CPU: 11th Gen Intel(R) Core(TM) i7-11700K
      - GPU: NVIDIA GeForce RTX 3070 Ti

### Software


      - xgboost 2.0.3
      - sklearn 1.4.2


<!-- (Bias, Risks, and Limitations) This section is meant to convey both technical and sociotechnical limitations. Dont think i have anything for this-->


## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Each model was optimized individually, finding the best set of hyperparameters using a grid search. Then, each model was combined into the final ensemble model.

