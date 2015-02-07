import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
import pickle
# import itertools
# import json
# import re
# import sklearn
# from sklearn import cross_validation
# from sklearn import tree
# from sklearn import svm
# from sklearn import ensemble
# from sklearn import neighbors
# from sklearn import linear_model
# from sklearn import metrics
from sklearn import preprocessing
# from sklearn import grid_search
# from sklearn.externals import joblib
# from sklearn.decomposition import PCA
# from sklearn import cluster
sns.set()


def jaccard_similarity(v1, v2):
    v1 = set(v1)
    v2 = set(v2)
    intersection = float(len(v1.intersection(v2)))
    jac = intersection/(len(v1) + len(v2) - intersection)
    return jac

def subject_similarity(example, df, possible_subjects):
    """
    Finds other tutors who teach similar subjects to "example" as measured by their jaccard similarity.
    """
    ex_subj = example[possible_subjects][example[possible_subjects]==1].index.values

    sim_tuts = df[possible_subjects].apply(\
                                 lambda x: jaccard_similarity( \
                                    x[x[possible_subjects]==1].index.values, \
                                    ex_subj), axis=1)
    sim_tuts.sort(ascending=False) # Example tutor should be at the index 0.
    sim_tuts = df[sim_tuts>.3] # Jaccard similarity threshold = 0.3

    return sim_tuts



def location_overlap(example, df):
    """
    Find tutors that overlap with the example tutor. Overlap is True if the tutoring radius (in miles) encompasses the center of the zip code of the example tutor.
    """

    def haversin(lat1, lon1, lat2, lon2):
        """
        Finds haversin distance (distance along great circle) in miles between two points. Points are defined by latitude and longitude. Radius of the Earth is assumed to be the midpoint between radius at the equator and radius at the pole.
        """
        r = 3956.545
        # Conver to radians
        lat1 = np.pi/180*lat1
        lon1 = np.pi/180*lon1
        lat2 = np.pi/180*lat2
        lon2 = np.pi/180*lon2


        d = 2*r*np.arcsin(np.sqrt(\
                                  np.sin((lat2-lat1)/2)**2 + \
                                  np.cos(lat1)*np.cos(lat2)*\
                                  np.sin((lon2-lon1)/2)**2))
        return d

    def is_overlap(example, row):
        d = haversin(row['lat'],row['lon'], example['lat'], example['lon'])
        if d<(row['zip_radius']):
            return True
        else:
            return False

    close_tuts = df[df.apply(lambda x: is_overlap(x, example), axis=1)]

    return close_tuts

def graduate_degrees(example, df):
    """
    Checks to see if example tutor has a graduate degree. If so, everybody else is fair game to compare. If not, then only look at other tutors with Bachelors.
    """
    degrees = ['EdD',
               'Enrolled',
               'Graduate_Coursework',
               'J.D.',
               'MBA',
               'MD',
               'MEd',
               'Masters',
               'Other',
               'PhD']
    if all(x==0 for x in example[degrees].tolist()):
        # example has no graduate degree
        return df[df.apply(lambda x: all(y==0 for y in x[degrees].tolist()))]
    else:
        return df

def main(example=None):

    print "loading dataframe"
    df = pickle.load(open('../data/tutor_df_20150201.pkl', 'rb'))
    print "finished loading dataframe"
    possible_subjects = pickle.load(open('../data/possible_subjects.pkl', 'rb'))
    example = df.iloc[example]
    sim_tuts = subject_similarity(example, df, possible_subjects)
    # example tutor is now at index 0
    sim_tuts = location_overlap(example, sim_tuts)
    # Relevant features for computing similarity
    rel_feats = ['avg_review_length',\
                 'badge_hours',\
                 'days_since_last_review',\
                 'has_rating',\
                 'number_of_ratings',\
                 'number_of_reviews',\
                 'profile_picture',\
                 'rating',\
                 'has_ivy_degree',\
                 'has_background_check',\
                 'response_time',\
                 'avg_review_sentiment']
    print "Your info: \n" + str(example[rel_feats])
    print str(example['url'])

    X = sim_tuts[rel_feats].as_matrix().astype(np.float)
    y = sim_tuts['hourly_rate'].as_matrix().astype(np.float)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    # Recall from subject_simiarity() that example tutor is at index 0
    X_corr = np.corrcoef(X)[0,:] # example tutor's correlation with all others
    X_weights = X_corr[X_corr[1:]>0.25][1:]
    y_match = y[X_corr[1:]>0.25][1:]
    plt.figure()
    sns.kdeplot(y_match, shade=True, bw=5)
    plt.xlabel('Hourly Rates ($)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.ylabel('Relative frequency', fontsize=20)
    plt.yticks(fontsize=12)
    plt.title("Pricing distribution for similar tutors", fontsize=28)
    plt.tight_layout()
    plt.show()
    # plt.hist(y_match, bins=20, normed=True)
    # print "Your hourly rate: $" + str(example['hourly_rate']) + "\n"
    # print "Number of other similar tutors = " + str(len(y_match))
    # sorted_tuts = sim_tuts.reset_index()
    # sorted_tuts = sim_tuts.iloc[np.argsort(X_corr[X_corr>0.25])]
    # print str(sorted_tuts[sorted_tuts['hourly_rate']==sorted_tuts['hourly_rate'].max()]['url'].values)








