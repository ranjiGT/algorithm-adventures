# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 15:06:45 2021

@author: Ranji Raj
"""

import speech_recognition as sr

AUDIO_FILE = ('C:\\Users\\User\\Downloads\\preamble.wav')

r = sr.Recognizer()

with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)
    
try:
    print("The audio file reads: " +r.recognize_google(audio))
    
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
    
except sr.RequestError as e:
    print("Could not request results from Google Speech recognition service; {0}".format(e))
