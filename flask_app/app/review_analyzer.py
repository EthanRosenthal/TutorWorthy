from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from HTMLParser import HTMLParser
import pandas as pd
from pandas import Series, DataFrame
import numpy as np


class feature_extractor():

    def __init__(self, reviews):
        self.reviews = reviews
        self.tokenizer = RegexpTokenizer('\w+')
        self.sentiment_dict = self.get_sentiment_dict()
    ###################################
    """
    Stolen from stack overflow:
    http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
    """
    class MLStripper(HTMLParser):
        def __init__(self):
            self.reset()
            self.fed = []
        def handle_data(self, d):
            self.fed.append(d)
        def get_data(self):
            return ''.join(self.fed)

    def strip_tags(self, html):
        s = self.MLStripper()
        s.feed(html)
        return s.get_data()
###################################

    def get_sentiment_dict(self):
        """ Load in sentiment file AFINN-111 as dictionary """

        sent_file = open('./data/AFINN-111.txt')
        sentiment_dict = {}
        for line in sent_file:
          term, score  = line.split("\t")
          sentiment_dict[term] = int(score)

        return sentiment_dict

    def extract(self):

        def row_func(row):
            if row:
                review_sent = []
                review_length = []
                for review in row:
                    words = (review['text'] + ' ' + review['title']).lower()
                    tokens = self.tokenizer.tokenize(self.strip_tags(words))
                    tokens = [w for w in tokens if w not in stopwords.words('english')]
                    sent_score, word_count = self.sentiment_count(tokens)
                    review_sent.append(sent_score)
                    review_length.append(word_count)

                avg_review_sent = np.mean(review_sent)
                avg_review_length = np.mean(review_length)
            else:
                avg_review_sent = 0.
                avg_review_length = 0.
            return Series({'avg_review_sentiment':avg_review_sent, \
                        'avg_review_length':avg_review_length})

        review_df = self.reviews.apply(lambda x: row_func(x))

        return review_df

    def sentiment_count(self, tokens):
        """
        Calculate sentiment score for list of "tokens".
        """
        # Initialize
        sent_score = 0
        word_count = 0

        for word in tokens:
            if self.sentiment_dict.has_key(word):
                sent_score += int(self.sentiment_dict[word])
            word_count += 1

        if word_count == 0:
            sent_score = 0.
        else:
            # Note: normalizing by sqrt of total number of words.
            sent_score = float(sent_score)/np.sqrt(word_count)

        return sent_score, word_count

