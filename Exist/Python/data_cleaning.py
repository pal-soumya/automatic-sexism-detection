from clean_data import clean_str 


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

training_data_path = "../EXIST2021_dataset/training/EXIST2021_training.tsv"
training_tsv = open(training_data_path);
training = pd.read_csv(training_data_path, sep="\t")

training_english = training.loc[training['language']== 'en']
training_spanish = training.loc[training['language']== 'es']

subset = training_english.sample(1000,random_state=42)    


Word = WordNetLemmatizer()
subset['clean text'] = subset['text'].apply(lambda x : \
    ' '.join([Word.lemmatize(word) for word in clean_str(x).split()]) )
    

vect = TfidfVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(subset['clean text'])
Y = np.array(subset['task1'])  


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print ("train size:", X_train.shape)
print ("test size:", X_test.shape)


model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print ("\nAccuracy: ",acc)
    
training_tsv.close()
