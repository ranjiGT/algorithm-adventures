# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:51:38 2021

@author: Ranji Raj 
"""
import nltk
from nltk import pos_tag
from nltk import RegexpParser

quote = "If you tell the truth, you don't have to remember anything."

tokens = nltk.word_tokenize(quote)
print(tokens)

tags =  nltk.pos_tag(tokens)
print(tags)

grammar = "NP: {<DT>?<JJ>*<NN>}"

cp = nltk.RegexpParser(grammar)

res = cp.parse(tags)
print(res)

res.draw()