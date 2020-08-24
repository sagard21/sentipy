import re
from sentipy.patterns import (like_currency, like_date, like_link,
                              like_mentions, like_number)
import spacy


# Initialize Spacy's English class
nlp = spacy.load('en_core_web_sm')

# Create a combined pattern
combined_pattern = f'{like_currency}|{like_date}|{like_link}|{like_mentions}|{like_number}'

# Pattern replace list
pattern_replace_ls = [
    (like_currency, '_currency_'),
    (like_date, '_date_'),
    (like_link, '_link_'),
    (like_mentions, '_mention_'),
    (like_number, '_number_')
]


def process_patterns(text, lower_case=True, delete=True):
    """
    Identifies the RE pattern and replaces it with pattern name

    Params:
    -------
    text: str - text to be analyzed
    delete: bool - True by defaullt
        True deletes the patterns identified
        False replaces the pattern identified by pattern name

    Returns:
    --------
    str - text with patterns deleted or replaced by pattern names
    """

    if delete:
        text = re.sub(combined_pattern, '', text, flags=re.I | re.M)

    else:
        for pat, norm_text in pattern_replace_ls:
            text = re.sub(pat, norm_text, text, flags=re.I | re.M)

    if lower_case:
        return text.lower()

    return text


def eliminate_stopwords(text):
    """
    Eliminates stop words

    Params:
    -------
    text: str - text to be analyzed

    Returns:
    --------
    str - text without stopwords
    """

    # Save tokens
    doc = nlp(text)

    return ' '.join([i.text for i in doc if not i.is_stop])


if __name__ == "__main__":
    print(process_patterns('This is a $100 question. You have 2 attempts as \
                            on 21/10/2019', False))
    print(eliminate_stopwords('This is a $100 question. You have 2 attempts \
                               as on 21/10/2019'))
