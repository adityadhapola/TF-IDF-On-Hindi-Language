import pandas as pd
import numpy as np
import os
import glob
import matplotlib as plt
from textblob import TextBlob as tb
from cltk.tokenize.sentence import TokenizeSentence
import re
from cltk.stop.classical_hindi.stops import STOPS_LIST
from wordcloud import WordCloud


punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''


hindi_text = open('speech.txt','r', encoding = 'utf-8').read()

hindi_text = re.sub("[a-zA-Z0-9]+", "", hindi_text)

hindi_text_1 = ""
for char in hindi_text:
   if char not in punctuations:
       hindi_text_1 = hindi_text_1 + char
     

tokenizer = TokenizeSentence('hindi')

hindi_text_tokenize = tokenizer.tokenize(hindi_text_1)

hindi_text_tokenize = pd.DataFrame(hindi_text_tokenize)

hindi_text_tokenize = hindi_text_tokenize.rename(columns={ hindi_text_tokenize.columns[0]: "words" })

counts = pd.DataFrame()

counts['word_frequency'] = hindi_text_tokenize['words'].value_counts()

counts['author'] = 'modi'

counts.reset_index(inplace = True) 

counts = counts[['author', 'index', 'word_frequency']]

stop_words = pd.DataFrame(STOPS_LIST)


final_words = pd.DataFrame()

final_words = counts[~counts['index'].isin(stop_words[0]).dropna()]



################################################################################################
# removing all the stop words

#idf

all_unique = counts.author.nunique()

idf = pd.DataFrame()

idf = counts.groupby('index').author.nunique()

idf = idf.reset_index()

idf['idf'] = np.log(2 / idf['author'])

#tf

word_sum = counts.groupby('author').sum()

tf = pd.DataFrame()

tf = counts

tf['word_sum'] = int(1375)


tf['term_frequency'] = tf['word_frequency']/tf['word_sum']

#tf*idf

tf_idf = tf

tf_idf['tf_idf'] = tf_idf.term_frequency.values * idf['idf']



