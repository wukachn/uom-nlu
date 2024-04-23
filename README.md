# Natural Language Inference Shared Task (Group 19)

## Traditional ML Approach (A)

Please ensure that the `(train|dev|test).csv` files are stored at `data/<file>`.

If you do not plan on training the models yourself, please load the trained models and place them in the correct path: https://drive.google.com/drive/folders/1lvrWGcUjme_sG4yvtNi0cQ4TAOil10ce?usp=sharing

### Training Notebook
First, this model loads the training (path: `data/train.csv`) and dev (path: `data/dev.csv`) datasets, obtaining training/test features using TF-IDF embeddings.
Then, the three individual models are trained/optimized using the same training data. Each model's optimimal hyperparameters are found using a grid search. The three models:
- A Logistic Regression Model
- A Random Forest Model
- A Gradiant Boosting Model (using the xgboost library for added performance)

Finally, the final ensemble model is built using the three models with `hard` voting, before being quickly tested for an accuracy score.

### Evaluation Notebook
This notebook simply loads the trained ensemble model (path: `models/ensemble_model.joblib`) and the TF-IDF vectorizer (path: `models/tfidf/tfidf_vectorizer.joblib`).
Then, loads test data from `data/<file-name>.csv`, generates predictions using the trained model and produces a series of evaluation metrics:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score...)
- Cross-Validation Accuracy

### Demo Notebook
This notebook simply loads the trained ensemble model (path: `models/ensemble_model.joblib`) and the TF-IDF vectorizer (path: `models/tfidf/tfidf_vectorizer.joblib`).
Then, loads test data from `data/test>.csv`, generates predictions using the trained model and saves them to a CSV file.

## Deep Learning (w/o Transformers) Approach (B)

### Training Notebook

### Evaluation Notebook

### Demo Notebook
