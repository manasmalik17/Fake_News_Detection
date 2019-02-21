# Fake_News_detection
#In this project we have made prediction of fake news from the dataset. We make use of python to simplify.
#We have downloaded the dataset from "https://www.kaggle.com/".

#Make sure you have anaconda navigator in your system. We have coded on Jupiter Notebook for the simplification, though it is just a preference.

#import all the libraries that are required for your fitting, classification and plotting.
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
#We make use of Naive Bayes Classifier for our model. The two important things that are important here are the count vectorizer and tfidf vectorizer.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Using Tfidf- Instead of just counting frequency we could do something more advanced like penalizing words that appear frequently in most of the sample.
#tfidf not only counts the occurrence of a word in the given text(or document), but also reflect how important the word is to the document. 
#Naive Bayes classifiers are a good example of being both simple (naive) and powerful for NLP tasks such as text classification.
#For a read on the classifier follow the link-
# "https://medium.com/syncedreview/applying-multinomial-naive-bayes-to-nlp-problems-a-practical-explanation-4f5271768ebf"
#We take "label" from the dataset that we have for the collection of news, as it contains the precise mark of "Fake" and "Real". 

#We have defined a function to plot the confusion matrix. Confusion matrix shows exactly the number of "Fake" and "Real" news in the dataset.
import matplotlib.pyplot as plt

#Multinomial Naive Bayes gives us the accuracy on tfidf:85.7%a
#Multinomial Naive Bayes gives us the accuracy on count:89.3%

#To increase the accuracy we have applied Passive Aggressive Classifier to our model.
#For a read on Passive Aggressive Classifier follow the link -
# "https://www.bonaccorso.eu/2017/10/06/ml-algorithms-addendum-passive-aggressive-algorithms/amp/"

#Our final accuracy: 93.639%
