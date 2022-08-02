from extract_feature import extract
import models

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def svm(X_train, y_train, X_test,y_test):
    for kernel in ('linear', 'rbf'):
        y_pred_svm = models.svm(X_train, y_train, X_test, kernel)
        acc=accuracy_score(y_test,y_pred_svm)
        print(precision_recall_fscore_support(y_test, y_pred_svm, average='macro'))
        print(precision_recall_fscore_support(y_test, y_pred_svm, average='micro'))
        print("SVM Accuracy for", kernel, " :",acc)      
    
    
def rand_forest(X_train, y_train, X_test,y_test):
    y_pred_rand = models.random_forest(X_train, y_train, X_test)
    acc = accuracy_score(y_test,y_pred_rand)
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
    print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
    print ("RF Accuracy: ",acc)
    
def knn(X_train, y_train, X_test,y_test):
    y_pred_rand = models.KNN(X_train, y_train, X_test)
    acc = accuracy_score(y_test,y_pred_rand)
    print ("KNN Accuracy: ",acc)    
    
def en1(X_train, y_train, X_test,test_en):
    y_pred = models.svm(X_train, y_train, X_test, 'linear')
    a = y_pred
    b =  test_en['id'].to_numpy()
    final = np.hstack( (np.resize(b,(len(b),1)),np.resize(a,(len(a),1))) )
    return final
     

def sp1(X_train, y_train, X_test,test_sp):
    y_pred = models.svm(X_train, y_train, X_test, 'linear')
    a = y_pred
    b =  test_sp['id'].to_numpy()
    final = np.hstack( (np.resize(b,(len(b),1)),np.resize(a,(len(a),1))))
    return final
    
def en2(X_train, y_train, X_test,test_en):
    y_pred = models.random_forest(X_train, y_train, X_test)
    a = y_pred
    b =  test_en['id'].to_numpy()
    final = np.hstack( (np.resize(b,(len(b),1)),np.resize(a,(len(a),1))) )
    return final
    
def sp2(X_train, y_train, X_test, test_sp):
    y_pred = models.random_forest(X_train, y_train, X_test)
    a = y_pred
    b =  test_sp['id'].to_numpy()
    final = np.hstack( (np.resize(b,(len(b),1)),np.resize(a,(len(a),1))))
    return final
    
    

training_data_path = "../EXIST2021_dataset/training/EXIST2021_training.tsv"
test_data_path = "../EXIST2021_dataset/test/EXIST2021_test.tsv"

training = pd.read_csv(training_data_path, sep="\t")
test = pd.read_csv(test_data_path, sep="\t")

training_en = training.loc[training['language']== 'en']
training_sp = training.loc[training['language']== 'es']

test_en = test.loc[test['language']== 'en']
test_sp = test.loc[test['language']== 'es']

train_subset_en = training_en.sample(frac = 1,random_state = 42) 
train_subset_sp = training_sp.sample(frac = 1,random_state = 42)   

train_en = train_subset_en.sample(frac = 0.9)
testt_en = train_subset_en.drop(train_en.index)
train_sp = train_subset_sp.sample(frac = 0.9)
testt_sp = train_subset_sp.drop(train_sp.index)

X_train,y_train,X_test = extract(train_en, testt_en, 'english')
X_train_, y_train_, X_test_ = extract(train_sp, testt_sp, 'spanish')

print("extraction done")
y_test = testt_en['task1']
y_test_ = testt_sp['task1']

# final_en = en2(X_train,y_train,X_test, test_en)
# final_sp = sp2(X_train_, y_train_, X_test_, test_sp)

# final = np.vstack((final_en, final_sp))


# pd.DataFrame(final).to_csv("run2_2.tsv", sep = "\t", index=False,header=False)
# file1 = pd.read_csv('run2_2.tsv', sep="\t")

svm(X_train, y_train, X_test, y_test)
svm(X_train_, y_train_, X_test_, y_test_)


#final_pred(train_subset_en, train_subset_sp, test_english, test_spanish)


