# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:44:01 2021

@author: Ranji Raj 
"""

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import warnings
warnings.filterwarnings('ignore')

f = open('C:\\Users\\User\\Downloads\\5G.txt', 'r', errors='ignore')
text = f.read()

stopwords = list(STOP_WORDS)
#print(stopwords)

#spacy.cli.download("en")
nlp = spacy.load('en_core_web_sm')

doc = nlp(text)

tokens = [token.text for token in doc]
#print(tokens)

#print(punctuation)
####################################
word_frequencies = {}
for word in doc:
    if word.text.lower() not in stopwords:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

#print(word_frequencies)
###################################
max_frequency = max(word_frequencies.values())
#print(max_frequency)
###########################
for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word]/max_frequency
    
#print(word_frequencies)

##################################################
sentence_tokens = [sent for sent in doc.sents]
#print(sentence_tokens)

##################################################
sentence_scores = {}
for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_scores.keys():
                sentence_scores[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_scores[sent] += word_frequencies[word.text.lower()]

#print(sentence_scores)

#Now obtain 30% of sentence with maximum score and is done by heapq

from heapq import nlargest

select_length = int(len(sentence_tokens)*0.3)
#print(select_length)

summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
#print(summary)

final_summary = [word.text for word in summary]
summary = ''.join(final_summary)

print("******************************SUMMARY**************************************",summary)

print(len(text))
print(len(summary))
