# Sentipy
This package is a basic sentiment analyzer

## About `sentipy`
`Sentipy` attempts to classify a text into `positive`, `negative` or `neutral`
sentiment. It uses Spacy's `textcat` with ensemble architecture in the back-end.
The ultimate objective of this package is to classify the sentiments as
accurately as possible

While the sentiment analysis at the core is absolutely basic, the current focus
is to understand the features the model is learning. `sentipy` leverages on
`lime` to get the features learnt and uses `streamlit` to crate a simple webapp
that helps with the visualization

## How to install `sentipy`
1. git clone the repo in to your local system
2. run setup.py install

## How to run the feature visualizer app
On your terminal run `sentipy streamlit`

## What next?
1. Use transformers and take the self supervised learning approach for classification
2. Include visualizations for pre-processed text
3. Make pre-processing options available on web app
4. Better visualization