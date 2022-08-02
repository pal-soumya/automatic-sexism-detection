from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd
import re

from stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
                                 

def remove_stop_words_utility(sentence,lang):
    stop_words = list(get_stop_words(lang))         #About 900 stopwords
    nltk_words = list(stopwords.words(lang))   #About 150 stopwords
    stop_words.extend(nltk_words)
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def remove_stop_words(subset,lang):
    filtered = subset['text'].apply(lambda x: remove_stop_words_utility(x,lang))
    return filtered

    
def pre_process(df,lang):
    filtered = remove_stop_words(df,lang)
    df['text'] = filtered
    if (lang == 'english'):
        Word = WordNetLemmatizer()    
        clean_data = df['text'].apply(lambda x : \
            ' '.join([Word.lemmatize(word) for word in clean_str(x).split()]) )
    else:
        stemmer = SnowballStemmer('spanish')
        clean_data = df['text'].apply(lambda x : \
        ' '.join([stemmer.stem(word) for word in clean_str(x).split()]))
    return clean_data
