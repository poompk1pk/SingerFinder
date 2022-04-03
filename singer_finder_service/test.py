import logging
import pythainlp.util
import requests
from bs4 import BeautifulSoup
import re
import threading
import pandas as pd
import warnings
from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import nltk
from nltk import stem

import nltk
nltk.download('omw-1.4')

from pythainlp.corpus.wordnet import wordnet as wn

from pythainlp.corpus.wordnet import synsets
from pythainlp.corpus.wordnet import all_lemma_names

print(wn.synsets('ฟ้า', lang='tha')[0].lemma_names('tha'))