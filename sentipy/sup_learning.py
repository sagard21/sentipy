from __future__ import unicode_literals
from pathlib import Path
import pandas as pd
from utils import spacy, process_patterns
from spacy.util import minibatch, compounding
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


def prepare_training_data(train_df, text_col, label_col, test_ratio=0.2):
    """
    Prepares the data in training format required by Spacy

    Params:
    -------
    train_df: pandas dataframe of training data
    text_col: str - col name of text data
    label_col: str - col name of labels (y)
    test_ratio: float - Amount of data to be used for testing. Defaults to 0.2

    Returns:
    --------
    tuples: (train_text, train_categories), (test_text, text_categories)
    """

    assert isinstance(test_ratio, float), "test_ratio mist be a float"
    assert 0.01 < test_ratio < 0.99, "test_ratio must be between 0.01 and 0.99"
    assert isinstance(train_df, pd.DataFrame), "data_df must be a \
                                                Pandas dataframe"
    assert all(i in train_df.columns for i in [text_col, label_col]), "both \
                                        text and label columns must be present"

    train_df = train_df[[text_col, label_col]]
    train_df.columns = ['text', 'label']

    # Dummify the data with bool values for label
    train_df = pd.get_dummies(train_df, columns=['label'], prefix='',
                              prefix_sep='', dtype=bool)

    # Shuffle training data
    train_df = shuffle(train_df)

    # Create a tuple of text data and list of categories dictionaries
    text_data = tuple(train_df['text'].values)
    cat_data = train_df.drop('text', axis=1).to_dict(orient='records')

    # Split the data into train and test set and return the data
    split_idx = int(len(text_data) * test_ratio)

    return ((text_data[split_idx:], cat_data[split_idx:]),
            (text_data[:split_idx], cat_data[:split_idx]))


def train_seniment_classifier(trainin_data_path, text_col, label_col,
                              model_output_dir, test_ratio=0.2, epochs=10):
    """
    Trains sentiment classifier and saves model

    Params:
    -------
    trainin_data_path: str - path of training csv file (inclusive of filename)
    text_col: str - col name of text data
    label_col: str - col name of labels (y)
    test_ratio: float - Amount of data to be used for testing. Defaults to 0.2
    model_output_dir: str - path to save the model
    """

    assert isinstance(trainin_data_path, str), "trainin_data_path must a str"
    assert isinstance(model_output_dir, str), "trainin_data_path must a str"

    # Load the csv data into pandas dataframe and create traning data
    try:
        train_df = pd.read_csv(trainin_data_path)
    except Exception:
        train_df = pd.read_csv(trainin_data_path, encoding='latin')

    train_df[text_col] = train_df[text_col].map(process_patterns)

    (train_text, train_cats), (test_text, test_cats) = \
        prepare_training_data(train_df, text_col, label_col)

    train_data = list(zip(train_text, [{"cats": cats} for cats in train_cats]))

    test_cats_actuals = [k for m in test_cats for k, v in m.items() if v]

    # Check if the output dir exists. If not, create it
    model_output_dir = Path(model_output_dir)

    if not model_output_dir.exists():
        model_output_dir.mkdir()

    # Load spacy blank english model
    nlp = spacy.blank('en')

    # Add text classifier to the spacy pipeline
    textcat = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "ensemble"
        }
    )
    nlp.add_pipe(textcat, last=True)

    # Add labels to text classifier
    for label in train_df[label_col].unique():
        textcat.add_label(label)

    # Disable all other components in pipeline except textcat for training
    other_components = [cmp for cmp in nlp.pipe_names if cmp != "textcat"]

    with nlp.disable_pipes(*other_components):
        optimizer = nlp.begin_training()

        train_data_len = len(train_text)
        print(f"Starting training on {train_data_len} examples\n")

        # Configure batch sizes
        batch_sizes = compounding(4.0, 32.0, 1.001)

        for i in range(epochs):
            iter_num = i+1
            print(f"Iteration number - {iter_num}")

            losses = {}
            batches = minibatch(train_data, size=batch_sizes)

            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)

            # Test performance on each iteration
            test_cats_pred = []

            with textcat.model.use_params(optimizer.averages):
                docs = [nlp.tokenizer(txt) for txt in test_text]
                for doc in textcat.pipe(docs):
                    test_cats_pred.append(max(doc.cats, key=doc.cats.get))

            print("Classification Report:\n")
            print(classification_report(test_cats_actuals, test_cats_pred))

    # Save the model
    print(f"Saving model to {model_output_dir}")
    nlp.to_disk(model_output_dir)

    with nlp.use_params(optimizer.averages):
        nlp.to_disk(model_output_dir)

    print('Model saved!!!')
