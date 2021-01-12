# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:23:06 2021

@author: Ranji Raj 
https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
"""

import nltk
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

sw = set(stopwords.words('english'))

quote = "Two things are infinite: the universe and human stupidity; and I'm not sure about the universe."
tokenized = sent_tokenize(quote)

for i in tokenized:
    word_list = nltk.word_tokenize(i)    
    word_list = [w for w in word_list if not w in sw]    
    tagged = nltk.pos_tag(word_list)    
    print(tagged)