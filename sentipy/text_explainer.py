import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from spacy.util import load_model_from_path
from os.path import join, dirname, abspath
from utils import process_patterns
import matplotlib.pyplot as plt
import click
import streamlit.cli

# Set directory paths
BASE_DIR = dirname(abspath(__file__))
DATA_DIR = join(BASE_DIR, 'data')
MODEL_DIR = join(BASE_DIR, 'model')

# Load model
sentiment_clf = load_model_from_path(MODEL_DIR)

# Load airline CSV
training_csv = join(DATA_DIR, 'airline_sentiment.csv')

train_df = pd.read_csv(training_csv, encoding='latin')
train_df = train_df[['text', 'airline_sentiment']]
cats = list(train_df['airline_sentiment'].unique())

arr_len = len(cats)

# Initialize LIME text explainer
explainer = LimeTextExplainer(class_names=cats)


def predict_prob(context):
    """
    Provideds sklearn style predict_proba output

    Params:
    -------
    context: str - text content to evaluate

    Returns:
    --------
    numpy array of probabilities
    """

    fin_arr = np.zeros(shape=(1, arr_len))

    for txt in context:
        doc = sentiment_clf(txt)
        preds = np.array(list(doc.cats.values())).reshape(1, arr_len)
        fin_arr = np.vstack((fin_arr, preds))

    return fin_arr[1:]


def explain_text_features(context, viz_features, delete_patterns=False):
    """
    Explains the mot important features of text

    Params:
    -------
    context: str - text content to evaluate
    viz_features: int - Number of features to visualize. Defaults to 5

    Returns:
    --------
    Explanation of features for class
    """

    # Print predicted class
    doc = sentiment_clf(process_patterns(context, delete=delete_patterns))
    outcome = max(doc.cats, key=doc.cats.get)
    # print(f'Predicted class - {outcome}\n')

    # Initialize explainer
    exp = explainer.explain_instance(context, predict_prob, labels=[0, 1, 2],
                                     num_features=viz_features)

    # Initialize a matpotlib figure
    fig = plt.figure(figsize=(15, 20))

    # Create subplot for Neutral
    plt.subplot(3, 1, 1)
    plt.title('Features contribution for class Neutral', fontsize=21,
              color='blue')
    neu_values = [i[1] for i in exp.as_list(label=0)]
    neu_words = [i[0] for i in exp.as_list(label=0)]
    neu_colors = ['green' if x > 0 else 'red' for x in neu_values]
    plt.barh(y=neu_words, width=neu_values, align='center', color=neu_colors)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Create subplot for Positive
    plt.subplot(3, 1, 2)
    plt.title('Features contribution for class Positive', fontsize=21,
              color='blue')
    pos_values = [i[1] for i in exp.as_list(label=1)]
    pos_words = [i[0] for i in exp.as_list(label=1)]
    pos_colors = ['green' if x > 0 else 'red' for x in pos_values]
    plt.barh(y=pos_words, width=pos_values, align='center', color=pos_colors)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Create subplot for Negative
    plt.subplot(3, 1, 3)
    plt.title('Features contribution for class Negative', fontsize=21,
              color='blue')
    neg_values = [i[1] for i in exp.as_list(label=2)]
    neg_words = [i[0] for i in exp.as_list(label=2)]
    neg_colors = ['green' if x > 0 else 'red' for x in neg_values]
    plt.barh(y=neg_words, width=neg_values, align='center', color=neg_colors)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Create a Super Title
    plt.suptitle('Classwise most important features', fontsize=34,
                 color='navy')

    return outcome, fig


@click.group()
def main():
    pass


@main.command("streamlit")
def main_streamlit():
    fname = join(BASE_DIR, 'streamlit_app.py')
    args = []
    streamlit.cli._main_run(fname, args)


if __name__ == "__main__":
    main()
