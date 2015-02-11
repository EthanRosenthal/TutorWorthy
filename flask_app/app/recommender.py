"""
Finds similar tutors to input tutor using Jaccard index, cosine similarity, and location-based filtering. Input tutor identified as "example" variable throughout.

Author: Ethan Rosenthal
Last Modified: 2/7/2015
"""

import matplotlib
matplotlib.use('Agg') # Avoid DISPLAY on AWS EC2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import seaborn as sns
import pickle
from cStringIO import StringIO
from sklearn import preprocessing
from weighted_kde import gaussian_kde
sns.set()
# Attempt to fix font issues in AWS. Doesn't work right now.
matplotlib.rcParams['font.sans-serif'].insert(0, 'Liberation Sans')
matplotlib.rcParams['font.sans-serif'].insert(0, 'Arial')
matplotlib.rcParams['font.family'] = 'sans-serif'


def jaccard_similarity(v1, v2):
    """
    Caluclate Jaccard similarity between two lists.
    J(A,B) = Intersection(A, B) / Union(A, B).

    Uses set theorem Union(A, B) = A + B - Intersection(A, B)
    """

    v1 = set(v1)
    v2 = set(v2)
    intersection = float(len(v1.intersection(v2)))
    jac = intersection/(len(v1) + len(v2) - intersection)
    return jac

def subject_similarity(example, df, possible_subjects):
    """
    Filter out tutors with Jaccard similarity <= 0.3

    Occasionally tutor teaches rare subject such that there is nobody with Jaccard similarity >0.3. The encompassing "try/except" in the parent views.py file picks this up as a failure to imput a correct url. Need to fix this error handling.
    """
    # Get subjects that example tutor tutors.
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
    Find tutors that overlap with the example tutor. Overlap is True if their tutoring radius (in miles) encompasses the center of the zip code of the example tutor.
    """

    def haversin(lat1, lon1, lat2, lon2):
        """
        Finds haversin distance (distance along great circle) in miles between two points. Points are defined by latitude and longitude. Radius of the Earth is assumed to be halfway between radius at the equator and radius at the pole.
        """
        r = 3956.545 # Radius of the Earth in miles

        # Conver to radians
        lat1 = np.pi/180*lat1
        lon1 = np.pi/180*lon1
        lat2 = np.pi/180*lat2
        lon2 = np.pi/180*lon2

        # Haversin formula
        d = 2*r*np.arcsin(np.sqrt(\
                                  np.sin((lat2 - lat1)/2)**2 + \
                                  np.cos(lat1) * np.cos(lat2)*\
                                  np.sin((lon2 - lon1)/2)**2))
        return d

    def is_overlap(row, example):
        """
        Find if two tutors overlap in location:
        Is the distance between the tutors less than the radius of the non-example tutor?
        """
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
        return df[df.apply(lambda x: all(y==0 for y in x[degrees].tolist()), axis=1)]
    else:
        return df

def cosine_similarity(v1, v2):
    """ Find cosine similarity between two arrays """
    sim = np.sum(v1*v2)/np.sqrt(np.sum(v1**2))/np.sqrt(np.sum(v2**2))
    return sim

def scale_kde(y, cos_tuts):
    """
    Find scaling to convert KDE from relative frequency to counts.

    Technically, this is likely incorrect. I should modify the weighted_kde routine, but this is suitable for now. Assume that the difference in scaling between the normalized and unnormalized NumPy histogram functions is a proxy for the unnormalized KDE scaling.
    """
    # Get unnormalized histogram
    hist, bins = np.histogram(y[cos_tuts>0], \
         weights=cos_tuts[cos_tuts>0], \
         bins=np.linspace(0, y[cos_tuts>0].max(), y[cos_tuts>0].max()/5+1), normed=False)

    # Get normalized histogram
    hist_norm, bins = np.histogram(y[cos_tuts>0], \
             weights=cos_tuts[cos_tuts>0], \
             bins=np.linspace(0, y[cos_tuts>0].max(), y[cos_tuts>0].max()/5+1), normed=True)

    hist_norm[hist_norm==0.] = 1. # Avoid divide by zero.

    scaling = hist/hist_norm
    scaling = scaling[np.isnan(scaling)!=True].max()

    return scaling

def make_kde_plot(x, pdf):
    """
    Make plot of KDE. Write image to StringIO object.
    """

    fig = plt.figure(figsize=(768/96, 400/96), dpi=9)
    ax = plt.gca()
    ax.plot(x, pdf)
    ax.fill_between(x, pdf, alpha=.5)

    # Formatting
    plt.xlabel('Hourly rate ($)', fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylabel('Number of tutors', fontsize=18)
    plt.yticks(fontsize=12)
    plt.title("Pricing distribution for similar tutors", fontsize=24)
    plt.tight_layout()
    plt.show()

    # Save file to variable instead of writing to disk.
    img_io = StringIO()
    plt.savefig(img_io, dpi=96, format='png')
    img_io.seek(0)

    return img_io



def main(example, df, possible_subjects):
    """
    Find similar tutors to example tutor. Calculate weighted kernal density estimate (KDE) of pricing distribution of similar tutors.

    Similarity conditions:
        (1) Jaccard index between subjects tutored >= 0.3
        (2) Radius that a tutor is willing to travel encompasses the center of the zip code of the example tutor.
        (3) Cosine similarity between profile features is >=0.5 for nearest neighbor, max-priced tutor, and min-priced tutor. Otherwise use cosine similarity to weight KDE.

    INPUTS
    example = (Series) with same format as df, but with only the input tutor from the website.
    df = (DataFrame) with all NYC tutors.
    possible_subjects = (List) of subjects that tutors tutor. This is restricted to the top 100 most popular subjects as previously calculated in cleanup_features.py

    OUTPUTS
    nearest_neighbor = (Series) with same format as df with most similar tutor to example.
    max_tut = (Series) of tutor that charges the highest hourly rate of tutors with cosine similarity > 0.5.
    min_tut = (Series) of tutor that charges the lowest hourly rate of tutors with cosine similarity < 0.5.
    img_io = KDE plot image. Actually in memory but behaves like a file (written to disk with StringIO)

    """

    # Drop example tutor if in df
    try:
        df.drop(df[example['url_id']==df['url_id']].index.values, inplace=True)
        df.reset_index(drop=True, inplace=True)
    except:
        pass # Tutor is not in database

    # Check for graduate degree
    df = graduate_degrees(example, df)

    # Filter by Jaccard index and location.
    sim_tuts = subject_similarity(example, df, possible_subjects)
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

    # Convert similar tutors to matrix. Normalize features.
    # In parlance of machine learning, X are features, y is hourly rate.
    X = sim_tuts[rel_feats].as_matrix().astype(np.float)
    y = sim_tuts['hourly_rate'].as_matrix().astype(np.float)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    X_example = example[rel_feats].as_matrix().astype(np.float)
    y_example = np.float(example['hourly_rate'])
    X_example = scaler.transform(X_example)

    # Get cosine similarity between example tutor and tutor db.
    cos_tuts = np.empty(X.shape[0])
    for i in xrange(X.shape[0]):
        cos_tuts[i] = cosine_similarity(X[i,:], X_example)

    # Sort by similarity
    sorted_idx = np.argsort(cos_tuts)[::-1]
    cos_tuts = cos_tuts[sorted_idx]
    y = y[sorted_idx]
    sim_tuts.reset_index(drop=True, inplace=True)

    # Only keep tutors with similarity > 0.5
    sim_tuts = sim_tuts.iloc[sorted_idx][cos_tuts>.5]

    # Calculate three outputted tutors.
    nearest_neighbor = sim_tuts.iloc[0] # Highest similarity
    max_tut = sim_tuts[sim_tuts['hourly_rate']==sim_tuts['hourly_rate'].max()].iloc[0]
    min_tut = sim_tuts[sim_tuts['hourly_rate']==sim_tuts['hourly_rate'].min()].iloc[0]

    scaling = scale_kde(y, cos_tuts)

    kde = gaussian_kde(y[cos_tuts>0], weights=cos_tuts[cos_tuts>0])
    x = np.linspace(0, y.max()+50, y.max()+50+1)

    pdf = kde(x)*scaling # Probability density function (estimated)

    img_io = make_kde_plot(x, pdf)

    return nearest_neighbor, max_tut, min_tut, img_io








