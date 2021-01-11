# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:43:09 2021

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#Amazon data
input_file = 'C:\\Users\\User\\Downloads\\sentiment labelled sentences\\sentiment labelled sentences\\amazon_cells_labelled.txt'
amazon = pd.read_csv(input_file, delimiter='\t', header=None)
amazon.columns = ['Sentence', 'Class']

#Yelp data
input_file = 'C:\\Users\\User\\Downloads\\sentiment labelled sentences\\sentiment labelled sentences\\yelp_labelled.txt'
yelp = pd.read_csv(input_file, delimiter='\t', header=None)
yelp.columns = ['Sentence', 'Class']

#IMDB data
input_file = 'C:\\Users\\User\\Downloads\\sentiment labelled sentences\\sentiment labelled sentences\\imdb_labelled.txt'
imdb = pd.read_csv(input_file, delimiter='\t', header=None)
imdb.columns = ['Sentence', 'Class']

#combine all datasets

data = pd.DataFrame()
data = pd.concat([amazon, yelp, imdb])
data['index'] = data.index
print(data)
#*************************************************************************
#Total count of each category
pd.set_option('display.width', 4000)
pd.set_option('display.max_rows', 1000)

distOfDetails = data.groupby(by='Class', 
                             as_index=False).agg({'index':pd.Series.nunique}).sort_values(by='index', ascending=False)
distOfDetails.columns = ['Class', 'COUNT']
print(distOfDetails)

#Distribution of all categories
plt.pie(distOfDetails['COUNT'], 
        autopct='%1.0f%%', 
        shadow=True, 
        startangle=360)
plt.show() #UseQt5

#************************************************************************
#Text preprocessing
columns = ['index', 'Class', 'Sentence']
_df = pd.DataFrame(columns=columns)

#lower string
data['Sentence'] = data['Sentence'].str.lower()

#remove email address
data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', 
                                            '', 
                                            regex=True)

#remove IP address
data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', 
                                            '', 
                                            regex=True)

#remove punctuations and special characters
data['Sentence'] = data['Sentence'].str.replace('[^\w\s]', '')

#remove numbers
data['Sentence'] = data['Sentence'].str.replace('\d', '', regex=True)

#remove stop words
for index, row in data.iterrows():
    word_tokens = word_tokenize(row['Sentence'])
    filtered_sentence = [w for w in word_tokens if not w in stopwords.words('english')]
    _df = _df.append({'index':row['index'], 'Class': row['Class'], 'Sentence': " ".join(filtered_sentence[0:])}, ignore_index=True)

data = _df
print('data', data)
#*************************************************************************

X_train, X_test, y_train, y_test = train_test_split(data['Sentence'].values.astype('U'), 
                                                    data['Class'].values.astype('int32'), 
                                                    test_size=0.3, 
                                                    random_state=10)
classes = data['Class'].unique()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

#Grid search result
vectorizer = TfidfVectorizer(analyzer='word', 
                             ngram_range=(1, 2), 
                             max_features=50000, 
                             max_df=0.5, 
                             use_idf=True, 
                             norm='l2')

counts = vectorizer.fit_transform(X_train)
vocab = vectorizer.vocabulary_
classifier = SGDClassifier(alpha=1e-05, 
                           max_iter=50, 
                           penalty='elasticnet')
targets = y_train
classifier = classifier.fit(counts, targets)
example_counts = vectorizer.transform(X_test)
predictions = classifier.predict(example_counts)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

#Model evaluation
acc = accuracy_score(y_test, predictions, normalize=True)
hit = precision_score(y_test, predictions, average=None, labels=classes)
capture = recall_score(y_test, predictions, average=None, labels=classes)
print('Model Accuracy: %.2f'%acc)
print(classification_report(y_test, predictions))
#********************************************************************
import itertools

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis] #Normalized CM
    
    else:
        print()

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = 'center',
                 color='white' if cm[i, j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(150, 100))


cnf_matrix = confusion_matrix(y_test, predictions, classes)
np.set_printoptions(precision=2)
class_names = range(1, classes.size+1)

#Plot for non-normlaized CM
plt.figure()

plot_confusion_matrix(cnf_matrix, 
                      classes=class_names, 
                      title='Confusion matrix, without normalization')
classInfo = pd.DataFrame(data=[])
for i in range(0, classes.size):
    classInfo = classInfo.append([[classes[i], i+1]], ignore_index=True)

classInfo.columns = ['Category', 'Index']
print(classInfo)