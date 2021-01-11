# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:28:01 2021

@author: Ranji Raj 
"""
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer

#PorterStemmer implementation
print('*************PorterStemmer****************')

ps = PorterStemmer()
tokens = ['consult', 'consulting', 'consultant', 'consultants', 'consultative']

for w in tokens:
    print(w +'-->'+ps.stem(w))


ttwister = 'She sells seashells by the seashore.'
tokenized = word_tokenize(ttwister)

for words in tokenized:
    print(words+'-->'+ps.stem(words))

#SnowballStemmer implementation

print('*************SnowballStemmer****************')

ps2 = SnowballStemmer('german')
das_zeichen = ['Atomkraftwerk','Atomkraftwerksdirektorenzimmer']

for _ in das_zeichen:
    print(_+'-->'+ps2.stem(_))

#LancasterStemmer implementation


print('*************LancasterStemmer****************')
ph = LancasterStemmer()

print(ph.stem('pets'))
print(ph.stem('trouble'))
print(ph.stem('troubling'))
print(ph.stem('troubled'))
print(ph.stem('demilitarized')) #Overstemming
print(ph.stem('degenerated')) #Overstemming 