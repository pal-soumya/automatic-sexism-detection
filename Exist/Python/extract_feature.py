from clean_data import pre_process
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy import sparse
import pandas as pd
import numpy as np
import re


swear_words_list1 = "../EXIST2021_dataset/swear_words/bad_words.csv"


def extract_feature1(train_data, test_data):
    """
    TfIfd feature
    """
    vect = TfidfVectorizer(min_df=20,max_features=(2000),max_df=0.6)
    X_train = vect.fit_transform(train_data)
    X_test = vect.transform(test_data)
    return X_train, X_test


def swear_word_count(string):
    swear_word = pd.read_csv(swear_words_list1)
    count = 0
    for word in swear_word['jigaboo']:
        if(word in string):
            count += 1
    return count


def extract_feature2(subset):
    """
    Number of Swear Words
    """
    swear_count = subset['text'].apply(lambda x : swear_word_count(x))
    rows = subset.shape[0]
    return swear_count.to_numpy().reshape((rows,1))


def extract_feature3(subset):
    """
    Upper Case Count
    """
    uppercase_count = subset['text'].apply(lambda x : sum(1 for c in x if c.isupper()))
    rows = subset.shape[0]
    return uppercase_count.to_numpy().reshape((rows,1))   

def extract_feature4(subset):
    """
    Hashtag Count
    """
    hashtag_count = subset['text'].apply(lambda x : sum(1 for c in x if c=='#'))
    rows = subset.shape[0]
    return hashtag_count.to_numpy().reshape((rows,1)) 
    
def extract_feature5(subset):
    """
    Links Count
    """
    link_count = subset['text'].apply(lambda x : len(re.findall(r'(https?://[^\s]+)', x)))
    rows = subset.shape[0]
    return link_count.to_numpy().reshape((rows,1)) 

def extract_feature6(subset):
    """
    Tweet length
    """
    tweet_length = subset['text'].apply(lambda x : len(x))
    rows = subset.shape[0]
    return tweet_length.to_numpy().reshape((rows,1))   


def extract_all(subset):
    f2 = extract_feature2(subset)
    f3 = extract_feature3(subset)
    f4 = extract_feature4(subset)
    f5 = extract_feature5(subset)
    f6 = extract_feature6(subset)
    result = np.hstack((f2,f3,f4,f5,f6))
    return result

def extract_sp(subset):
    f3 = extract_feature3(subset)
    f4 = extract_feature4(subset)
    f5 = extract_feature5(subset)
    f6 = extract_feature6(subset)
    result = np.hstack((f3,f4,f5,f6))
    return result

def extract(train_df, test_df, lang):
    if(lang=='english'):
        X1_train = extract_all(train_df)
        X1_test = extract_all(test_df)
    else:
        X1_train = extract_sp(train_df)
        X1_test = extract_sp(test_df)
    clean_train = pre_process(train_df, lang)
    clean_test = pre_process(test_df, lang)
    X2_train,X2_test = extract_feature1(clean_train,clean_test)
    #print(X1_train)
    X_train = hstack([sparse.coo_matrix(X2_train), sparse.coo_matrix(X1_train)])
    X_test = hstack([sparse.coo_matrix(X2_test), sparse.coo_matrix(X1_test)])
    y_train = train_df['task1']
    return X_train, y_train ,X_test
    

def extract_feature7(subset):
    """
    Bag of words
    """
    


def extract_feature8(subset):
    """
    Hate Lexicon
    """
    

