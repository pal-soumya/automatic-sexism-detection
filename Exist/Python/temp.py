from clean_data import pre_process
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from extract_feature import extract_all
from extract_feature import extract

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 5000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100


training_data_path = "../EXIST2021_dataset/training/EXIST2021_training.tsv"
test_data_path = "../EXIST2021_dataset/test/EXIST2021_test.tsv"

training = pd.read_csv(training_data_path, sep="\t")
test = pd.read_csv(test_data_path, sep="\t")


train_en = training.loc[training['language']== 'en']
train_sp = training.loc[training['language']== 'es']

test_en = test.loc[test['language']== 'en']
test_sp = test.loc[test['language']== 'es']

# train_subset_en = training_en.sample(frac = 1,random_state = 42) 
# train_subset_sp = train_sp.sample(100,random_state = 42)   

# # train_en = train_subset_en.sample(frac = 0.8)
# # test_en = train_subset_en.drop(train_en.index)
# train_sp = train_subset_sp.sample(frac = 0.8)
# test_sp = train_subset_sp.drop(train_sp.index)

lang = 'spanish'


subset__ = pre_process(train_sp, lang)


tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                      lower=True)
tokenizer.fit_on_texts(subset__.values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X1 = tokenizer.texts_to_sequences(subset__.values)
X1 = pad_sequences(X1, maxlen=MAX_SEQUENCE_LENGTH)
X1_test = tokenizer.texts_to_sequences(pre_process(test_sp,lang).values)
X1_test = pad_sequences(X1_test, maxlen=MAX_SEQUENCE_LENGTH)


X2 = extract_all(train_sp)
X2_test = extract_all(test_sp)
X_train = np.hstack((X1,X2))
X_test = np.hstack((X1_test,X2_test))

print('Shape of data tensor:', X_train.shape)
print('Shape of test tensor:', X_test.shape)

Y_train = pd.get_dummies(train_sp['task1']).values
print('Shape of label tensor:', Y_train.shape)

# Y_test = pd.get_dummies(test_en['task1']).values
# print('Shape of label test tensor:', Y_test.shape)

#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape)
      

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())


epochs = 5
batch_size = 64
history = model.fit(X_train, Y_train, epochs=epochs,
                    batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss',
                                             patience=3,
                                             min_delta=0.0001)])

y_pred = model.predict_classes(X_test)
#accr = model.evaluate(X_test,Y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


a = y_pred
b =  test_sp['id'].to_numpy()
final = np.hstack( (np.resize(b,(len(b),1)),np.resize(a,(len(a),1))))
    
    
pd.DataFrame(final).to_csv("run4.tsv", sep = "\t", index=False,header=False)
file1 = pd.read_csv('run4.tsv', sep="\t")
