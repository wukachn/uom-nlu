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
As mentioned in the previous section, please ensure that the `(train|dev|test).csv` files are stored at `data/<file>`.

Please also make sure that the GloVe embeddings are downloaded and stored at a location of your choice but the path is specified in the notebook. The path currently used is
`./input/embeddings/glove.6B/glove.6B.300d.txt`.

### Training Notebook
The training notebook imports the necessary libraries and loads the glove embeddings.
It then loads the training set and uses the glove embeddings to get a vector representation of the sentences.
The model is then defined and compiled using the TensorFlow library, and finally trained on the given data set.
The model is trained for 100 epochs and the final model is saved as `models/deep_learning/model.keras`.
It can be found in this [link](https://drive.google.com/drive/folders/1_nZ7zuid0HlLF7CLwIF7lZl0WevTdWGA?usp=sharing).

#### Architecture
The model is based on the sentence embedding sum-of-words approach that Bowman et al. (2015) proposed in their paper, "A large annotated corpus for learning natural language inference". In this approach, the model takes in two sentences, and for each sentence, the words are embedded using GloVe embeddings. The embeddings are then summed to get a sentence embedding. The two sentence embeddings are concatenated and passed through a feedforward neural network to get the final output.

Our model has a few modifications: instead of the sum-of-words approach, we use the average of the word embeddings which resulted in slightly better accuracy. We further experimented with different layers and different numbers of dense, dropout, and batch normalization layers to improve the model's performance, but the final model has the following architecture:

Input(300+300) -> Dense(500) -> Dense(400) -> Dense(300) -> Dense(300) -> Dense(300) -> Dense(200) -> Dense(2) -> Output

### Evaluation Notebook
The evaluation notebook imports the necessary libraries and loads the trained model. Then loads the dev set and uses the glove embeddings to get a vector representation of the sentences. The model is then evaluated on the dev set and the results are printed, which are then evaluated using the metrics mentioned below:
- Precision
- Recall
- F1-score
- Accuracy
- Area under the ROC curve

### Demo Notebook
The demo notebook imports the necessary libraries and loads the trained model. Then loads the test set for which we require the predictions to be reported and then saves the same in a csv file. The notebook also includes a function to predict the output for a given pair of sentences, and also if the input is in the form of a csv file.
