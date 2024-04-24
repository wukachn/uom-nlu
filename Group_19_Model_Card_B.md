---
{}
---
language: en
license: cc-by-4.0
tags:
- natural-language-inference
repo: https://github.com/wukachn/uom-nlu

---

# Model Card for r87977ph-s63644vb-NLI

<!-- Provide a quick summary of what the model is/does. -->

This model is a neural network based model that can determine whether, when provided
      with a pair of texts, the second text is an entailment, or a contradiction/neutral with respect to the first text.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The model is a neural network based model trained on 26K premise-hypothesis pairs.

- **Developed by:** Peter Hamer and Vinayak Singh Bhadoriya
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Neural Network

### Model Resources

<!-- Provide links where applicable. -->

- **Paper or documentation:** https://nlp.stanford.edu/pubs/snli_paper.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

26K premise-hypothesis pairs

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: adadelta optimizer does not require a learning rate
      - train_batch_size: 32
      - seed: 42
      - num_epochs: 100

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 6m 11.1s
      - duration per training epoch: 4s 6ms
      - model size: 3.29MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A development set consisting of 6K pairs was used to evaluate the model.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an accuracy of 64%, and an F1-score of 0.65 for entailments and 0.63 for contradictions/neutral pairs on the development set.

## Technical Specifications

### Hardware


      - RAM: at least 2 GB,
      - Storage: at least 4 GB to account for the word embeddings

### Software


      - Numpy 1.26.4
      - Pandas 2.2.1
      - Tensorflow 2.16.1
      - Matplotlib 3.8.3

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The original model that Bowman et al. (2015) proposed used a sum of word embeddings approach to represent the input. In our case, we
    have opted to go for an approach that uses the average of the word embeddings. Additionally, we have added more layers to the model to improve its performance.
