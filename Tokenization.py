# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 00:40:11 2021

@author: User
"""

import nltk

from nltk.tokenize import word_tokenize

quote = "Big dreams have small beginnings"

quote_tokens = word_tokenize(quote)
print(quote_tokens)

print(len(quote_tokens))

from nltk.util import bigrams, trigrams, ngrams
string = "Sit alone, you will find all your answers."

tokens = word_tokenize(string)
string_bigrams = list(nltk.bigrams(tokens))
print("******************Bigrams******************")
print(string_bigrams)

string_trigrams = list(nltk.trigrams(tokens))
print("******************Trigrams******************")
print(string_trigrams)

string_ngrams = list(nltk.ngrams(tokens, 5))
print("******************Ngrams******************")
print(string_ngrams)