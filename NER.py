# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:27:07 2021

@author: Ranji Raj 
https://spacy.io/usage/linguistic-features#named-entities
"""
import spacy

nlp = spacy.load('en_core_web_sm')

fact = "When Facebook (United States) announced its plans to acquire WhatsApp in February 2014, WhatsApp's founders attached a purchase price of $16 billion: $4 billion in cash and $12 billion remaining in Facebook shares. This price tag is dwarfed by the actual price Facebook paid: $21.8 billion, or $55 per user."

doc = nlp(fact)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)