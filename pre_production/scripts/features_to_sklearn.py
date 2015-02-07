import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import itertools
import json
import re
import datetime

from sklearn import preprocessing



class to_sklearn():

    def __init__(self, df, possible_subjects):

        self.df = df
        self.possible_subjects = possible_subjects


    def to_X_matrix(self):
        usable_features = [ 'number_of_ratings', 'profile_picture', \
                            'rating', 'lat', 'lon', 'zip_radius', \
                            'badge_hours', 'Bachelors', 'J.D.', \
                            'MBA', 'MD', 'MEd', 'Masters', 'PhD', \
                            'rating_x_number_of_ratings', \
                            'number_of_reviews', 'avg_review_length', \
                            'days_since_last_review' ]

        usable_features.extend(self.possible_subjects)

        self.df['zip_radius'] = self.df['zip_radius'].apply(lambda x: \
                                                            np.float(x))

        X = self.df[usable_features].as_matrix().astype(np.float)
        self.scaler_X = preprocessing.StandardScaler().fit(X) # Normalize
        self.X = self.scaler_X.transform(X)

    def to_y_array(self):
        y = self.df['hourly_rate'].as_matrix().astype(np.float)
        self.scaler_y = preprocessing.StandardScaler().fit(y) # Normalize
        self.y = self.scaler_y.transform(y)
