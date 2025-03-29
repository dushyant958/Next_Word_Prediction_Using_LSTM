LSTM Model for Next-Word Prediction

Overview

This project builds an LSTM (Long Short-Term Memory) model to predict the next word in a given sequence using Shakespeare's Hamlet dataset. The model is trained and evaluated for word prediction accuracy and deployed on a Streamlit web application for real-time predictions.

Features

Uses an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function.

Trained on Shakespeare's Hamlet dataset from the NLTK Gutenberg corpus.

Evaluates predictions based on test sentences.

Deployed via a Streamlit web application for real-time user interaction.

Dataset

The dataset is collected using the NLTK Gutenberg corpus.

Raw text from Hamlet is preprocessed and tokenized for model training.

Installation

Prerequisites

Ensure you have the following dependencies installed:

pip install nltk tensorflow numpy pandas scikit-learn streamlit

Data Preparation

Download the dataset using NLTK:

import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg
data = gutenberg.raw('shakespeare-hamlet.txt')
with open('data.txt', 'w') as file:
    file.write(data)

Preprocess and tokenize the data using Keras Tokenizer.

Model Training

Run the notebook to:

Load and preprocess the dataset.

Tokenize text sequences and prepare training data.

Build and train the LSTM model.

Evaluate performance using a test set.

Deployment

The trained model is deployed using Streamlit:

streamlit run app.py

This launches a web application where users can input a word sequence to receive predictions in real time.

License

This project is licensed under the MIT License.
