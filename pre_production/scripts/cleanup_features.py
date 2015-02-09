import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import itertools
import json
import re
import datetime
import csv
import string
import review_analyzer

from sklearn import preprocessing


class clean():
    """
    This class takes the raw JSON file outputted from scraper.py and cleans and generates features.

    Args:
        path (str): path to the .txt JSON file outputted from scraper.py

    Attributes:
        df (DataFrame): Built from the JSON output of scraper.py

    Methods:
        remove_duplicate_tutors: Find duplicate tutors by url and remove one instance.

        just_subjects: Creates a column in self.df that contains a list of all subjects that the tutor tutors.

        subjects_to_boolean_features: For all possible subjects that one can tutor, create a boolean column returning True or False depending on whether or not the tutor tutors that subject.

        badge_to_hours_and_boolean: Convert tutor badge to column with number of hours and boolean column identifying existence of badge.

        ratings_to_boolean: Create column identifying existence of rating.

        get_education: Create column with boolean dictionary identifying existence of various educational degrees.

        student_reviews_to_features: Collect all student reviews. Also calculate number of reviews, avg word count of review, days since last review.

        zipcode_to_lat_lon: Convert zip codes to latitude and longitude feature vectors.

        fillna: Handle some missing values (days since last review, no rating).

        remove_unpopular_subjects: Only keep top 50 subject feature vectors. Remove tutors who only tutored something below top 50.

        get_subject_categories: Find out which "categories" tutor tutors. ELementary education, Science, Math, etc... Return 1 or 0 whether or not tutor tutors *anything* in this category.

        get_qualified_subjects: Create new boolean columns for subjects for which tutors are qualified to teach.

        create_interaction_features: Cross-terms.

        sort_columns: Sort non-subject feature columns alphabetically. Append sorted subjects.

        all_features: Run all above methods.

    STILL TO DO
        - Parse educational degrees -> currently converting non-specific degrees to "Bachelors"
    """

    def __init__(self, \
                path, \
                df=DataFrame(), \
                possible_subjects=[], \
                ts_table=DataFrame()):

        self.path = path
        self.df = df
        self.possible_subjects = possible_subjects
        self.ts_table = ts_table

    def read_json(self):
        """ Read in json dump from scraper.py """
        self.df = DataFrame([json.loads(line) for line in open(self.path)])

    def read_pickle(self):
        """ Read in pickled file """
        self.df = pd.read_pickle(self.path)

    def remove_duplicate_tutors(self):
        """ Remove any tutors that appear multiple times in dataframe """

        # Find unique identifier in profile url
        url_id = self.df['url'].str.split('/\?z').apply(lambda x: x[0].split('/')[-1])
        url_id.name = 'url_id'
        dup_idx = url_id[url_id.duplicated('url')].index.values
        self.df.drop(dup_idx, inplace=True)
        url_id.drop(dup_idx, inplace=True)
        # add unique url_id column
        self.df = pd.concat([self.df, url_id], axis=1)
        self.df.reset_index(drop=True, inplace=True)



    def just_subjects(self):
        """
        Parse raw_subjects dictionaries and make set variable of all subjects tutored by tutor
        """
        # Just get the subjects (ignore topics)
        just_subjects = self.df['raw_subjects']

        # Get rid of topics
        just_subjects = just_subjects.apply(lambda x: x.values())

        # Flatten lists
        just_subjects = just_subjects.apply(lambda x: \
                        list(set(y.strip(' ') for z in x for y in z)))
        just_subjects.name = 'just_subjects'
        self.df = pd.concat([self.df, just_subjects], axis=1)

    def subjects_to_boolean_features(self):
        """
        Transform from considering subjects as a categorical variable into binary feature vector. Create new column for every subject. Value is 1 or 0 depending on whether or not tutor tutors this subject.
        """
        possible_subjects = list(set(itertools.chain.from_iterable(self.df['just_subjects'].values)))

        dummies_df = DataFrame(columns=possible_subjects, dtype='Bool')

        def subj_to_bool(row, categories):
            """ Return True/False if subject in categories is in row """

            out_row = []
            for subj in categories:
                if subj in row:
                    out_row.append(True)
                else:
                    out_row.append(False)
            return np.array(out_row)

        # Make new boolean column for each subject. Place in dummies_df DataFrame
        for i in xrange(0, self.df.shape[0]):
            dummies_df.loc[i] = subj_to_bool( \
                self.df['just_subjects'].loc[i], possible_subjects)

        self.possible_subjects = possible_subjects
        self.df = pd.concat([self.df, dummies_df], axis=1)

    def badge_to_hours_and_boolean(self):
        """
        Parse badge text to get number of hours tutored ('badge_hours') and boolean value for having a badge ('has_badge')
        """

        numbers_regex = re.compile('\d+')
        def get_badge_info(row, numbers_regex):
            row_regex = re.findall(numbers_regex, row)
            if len(row_regex)==1:
                return Series({'has_badge':True, 'badge_hours':int(row_regex[0])})
            else:
                return Series({'has_badge':False, 'badge_hours':0})

        badge_df = self.df['badge'].apply(lambda x: get_badge_info(x, numbers_regex))
        self.df = pd.concat([self.df, badge_df], axis=1)
        self.df = self.df.drop('badge', axis=1)

    def ratings_to_boolean(self):
        """ Return boolean value for whether or not tutor has ratings """
        has_rating = self.df['rating'].apply(lambda x: True - np.isnan(x))
        has_rating.name = 'has_rating'
        self.df = pd.concat([self.df, has_rating], axis=1)

    def get_education(self):
        """
        Parse educational information. Create integer columns for all possible graduate degrees. Value in column is number of that degree that tutor has.
        """

        def get_degree(ed_list):
            grad_degrees = {'MEd':0, 'MBA':0, 'Masters':0, \
                            'PhD':0, 'J.D.':0, 'MD':0, 'EdD':0, \
                            'Graduate_Coursework':0, 'Other':0, \
                            'Enrolled':0}

            # Flatten education dictionary values
            ed_list = list(itertools.chain.from_iterable(ed_list.values()))

            for tutor_degree in ed_list:
                if grad_degrees.has_key(tutor_degree):
                    grad_degrees[tutor_degree] += 1
            return Series(grad_degrees)

        ed = self.df['education'].apply(lambda x: get_degree(x))
        self.df = pd.concat([self.df, ed], axis=1)

        ########### Not currently using method below

        deg_subjects = {}

        def get_deg_subjects(ed_list, deg_subjects):
            for v in ed_list.itervalues():
                for x in v:
                    if deg_subjects.has_key(x):
                        deg_subjects[x] += 1
                    else:
                        deg_subjects[x] = 1


    def student_reviews_to_features(self):
        """
        Parse student reviews. Return 'review_date' for each review and 'days_since_last_review' for tutor.
        """


        number_of_reviews = self.df['student_reviews'].apply(lambda x: len(x))
        number_of_reviews.name = 'number_of_reviews'
        numbers_regex = re.compile('\d+')

        def parse_date(review_list, numbers_regex):
            review_date = []

            for review in review_list:
                # Pick out date
                date_info = re.findall(numbers_regex, review['author_info'])
                date_info = date_info[-3:]
                # ex: [12', '31', '14'] for New Year's Eve 2014
                date_info[-1] = '20' + date_info[-1]
                review_date.append(datetime.date(int(date_info[2]), \
                                                int(date_info[0]), \
                                                int(date_info[1])))

            if len(review_list)>0:
                days_since_last_review = (datetime.date.today() - \
                                        max(review_date)).days
            else:
                review_date = np.nan
                days_since_last_review = np.nan

            return Series({'review_date':review_date, \
                           'days_since_last_review':days_since_last_review})

        student_reviews = self.df['student_reviews'].apply( \
                                        lambda x: parse_date(x, numbers_regex))

        self.df = pd.concat([self.df, \
                            number_of_reviews, \
                            student_reviews], axis=1)

    def background_check_info(self):
        """
        Determine if tutor has a background check ('has_background_check'). Get 'days_since_background_check'
        """

        numbers_regex = re.compile('\d+')

        def parse_date(bg_row, numbers_regex):

            date_info = re.findall(numbers_regex, bg_row)

            if len(date_info)>0:
                date_info = date_info[-3:]
                    # ex: [12', '31', '14'] for New Year's Eve 2014
                date_info[-1] = '20' + date_info[-1]
                background_date = datetime.date(int(date_info[2]), \
                                                int(date_info[0]), \
                                                int(date_info[1]))
                has_background_check = 1
                days_since_background_check = (datetime.date.today() - \
                                                    background_date).days
            else:
                has_background_check = 0
                days_since_background_check = np.nan

            return Series({'has_background_check':has_background_check, \
                    'days_since_background_check':days_since_background_check})

        bg_check = self.df['background_check'].apply(lambda x: \
                                                parse_date(x, numbers_regex))
        self.df = pd.concat([self.df, bg_check], axis=1)

    def get_response_time(self):
        """ Parse average response time in minutes"""
        time_regex = re.compile('\d+\s\w+')

        def parse_response(response_row, time_regex):

            if response_row:
                time_str = re.findall(time_regex, response_row)[0]
                # ex. u'12 hours'
                time_str = time_str.split(' ')
                if time_str[1][:6] == 'minute':
                    # ex. u'35 minutes'
                    response_time = float(time_str[0])/60.
                elif time_str[1][:4] == 'hour':
                    response_time = float(time_str[0])
                elif time_str[1][:3] == 'day':
                    response_time = float(time_str[0])*24.
                else:
                    response_time = 'ERROR: Could not parse response time'
            else:
                response_time = np.nan

            return response_time

        self.df['response_time'] = self.df['response_time'].apply(lambda x: \
                                                parse_response(x, time_regex))

    def get_sentiment(self):
        """
        Get 'avg_review_sentiment', the average sentiment of all reviews as calculated by sentiment analysis using AFINN-111.txt
        """
        fe = review_analyzer.feature_extractor(self.df['student_reviews'])
        fe = fe.extract()
        # fe = DataFrame('avg_review_sentiment', 'avg_review_length')
        self.df = pd.concat([self.df, fe], axis=1)



    def zipcode_to_lat_lon(self):
        """
        Get latitude and longitude of center of zip code. This could stand to be sped up by using SQL instead of reading in and searching a csv.
        """
        zips = pd.read_csv('../data/zipcode/zipcode.csv', dtype={'zip':'str'})

        # Remove missing zip codes (should only be one row)
        missing = self.df[self.df['zip_code'].apply(lambda x: \
                                                type(x))!=unicode].index.values
        for idx in missing:
            self.df.drop(idx, inplace=True)

        def get_lat_lon(row, zips):
            try:
                loc = zips[zips['zip']==row]
                lat = loc['latitude'].values[0]
                lon = loc['longitude'].values[0]
                return Series({'lat':lat, 'lon':lon})
            except:
                # Found these two missing zip codes on
                # http://www.maptechnica.com
                if row == '10065': # One of the richest zip codes!
                    lat = 40.76490050000000
                    lon = -73.96243050000000
                elif row == '10075':
                    lat = 40.77355900000000
                    lon = -73.95606900000000

                return Series({'lat':lat, 'lon':lon})

        latlon = self.df['zip_code']
        latlon = latlon.apply(lambda x: get_lat_lon(x, zips))
        self.df = pd.concat([self.df, latlon], axis=1)

    def fillna(self):
        """
        Deal with missing values. Ad hoc.
        """
        # If no reviews, days since last review = max (2395)
        self.df['days_since_last_review'].fillna( \
                    value=self.df['days_since_last_review'].max(), \
                    inplace=True)

        self.df['response_time'].fillna( \
                    value=self.df['response_time'].max(), \
                    inplace=True)
        self.df['days_since_background_check'].fillna( \
                    value=self.df['days_since_background_check'].max(), \
                    inplace=True)

        # If no rating, just replace with average rating (4.77)
        self.df['rating'].fillna(value=self.df['rating'].mean(), \
                                inplace='true')

        self.df['avg_review_length'].fillna(value=0., inplace=True)
        self.df['avg_review_sentiment'].fillna(value=0., inplace=True)

    def remove_unpopular_subjects(self):
        """
        Only keep top 100 most popular subject columns. Remove tutors who only tutor unpopular subjects (sorry!).
        """

        subject_pop = self.df[self.df[self.possible_subjects]==True] \
                    [self.possible_subjects].sum()

        subect_pop = subject_pop.sort(ascending=False)

        # Remove unpopular subject columns
        unpop = subject_pop[100:].index.tolist() # Top 100!
        self.df.drop(labels=unpop, axis=1, inplace=True)
        self.possible_subjects = [x for x in self.possible_subjects \
                                    if x not in unpop]

        # Remove tutors who only tutor unpopular subjects
        unpop_tutors = \
                        self.df[self.df[self.possible_subjects].apply( \
                            lambda x: x.tolist().count(False) == \
                            len(x.tolist()), axis=1) == True].index.tolist()
        self.df.drop(labels=unpop_tutors, axis=0, inplace=True)

    def get_subject_categories(self):
        """
        Parse subject "categories" (ex. "Elementary Education"). Return binary value for each category corresponding to whether or not the tutor tutors in this category.
        """
        subject_cats = self.df['raw_subjects'].apply(lambda x: x.keys())
        subject_cats = set(list( \
                        itertools.chain.from_iterable(subject_cats.tolist()) \
                        ))

        def bool_subject_cats(row, subject_cats):
            row_cats = row.keys()
            return_dict = {}
            for cat in subject_cats:
                if cat in row_cats:
                    return_dict[cat] = 1
                else:
                    return_dict[cat] = 0
            return Series(return_dict)

        subj_cat_df = self.df['raw_subjects'].apply(lambda x: \
                                            bool_subject_cats(x, subject_cats))


        def rename_columns(column_names):
            """
            Lower case, underscore delimitting, append 'cat' to front for category
            """
            new_names = []
            for c in column_names:
                c = re.split('[/\s]', c)
                c = '_'.join([x.lower() for x in c])
                c = 'cat_' + c
                new_names.append(c)
            return new_names

        subj_cat_df.columns = rename_columns(subj_cat_df.columns.tolist())
        self.df = pd.concat([self.df, subj_cat_df], axis=1)
        self.cat_features = subj_cat_df.columns.tolist()

    def get_qualified_subjects(self):
        """
        Analogous to subjects_to_boolean_features(), but for "qualified subjects".
        """
        empty_dict = {'qual_'+x:0 for x in self.possible_subjects}
        empty_series = Series(empty_dict)

        def get_qual(row, empty_series):
            return_series = empty_series.copy()
            subjects = set(itertools.chain.from_iterable(row.values()))
            subjects = set(x.rstrip(', ') for x in subjects)
            for subj in subjects:
                if subj in return_series:
                    return_series[subj] = 1
            return return_series

        qual_subj_df = self.df['qual_subjects'].apply( \
                                        lambda x: get_qual(x, empty_series))
        self.df = pd.concat([self.df, qual_subj_df], axis=1)
        self.qualified_subjects = qual_subj_df.columns.tolist()

    def has_ivy_degree(self):
        """
        Binary valued corresponding to whether or not the tutor has an ivy league degree. Best I could come up with. Maybe there's other versions of the lookup table, but who knows.
        """
        ivy_leagues = ['brown', 'brown university', 'columbia', \
                   'columbia university', 'cornell', \
                   'cornell university', 'dartmouth', \
                   'dartmouth college', 'harvard', \
                   'harvard university', 'princeton', \
                   'princeton university', \
                   'university of pennsylvania', 'upenn', 'u penn',\
                   'penn', 'yale', 'yale university']
        exclude = set(string.punctuation)

        def find_ivy(row, ivy_leagues, exclude):
            ivy = 0
            for school in row.keys():
                school = ''.join(ch for ch in school if ch not in exclude)
                school = school.split(' ')
                school = [word.lower() for word in school]
                school = ' '.join(school)
                if school in ivy_leagues:
                    ivy = 1
            return ivy

        has_ivy = self.df['education'].apply(lambda x: \
                                            find_ivy(x, ivy_leagues, exclude))
        has_ivy.name = 'has_ivy_degree'

        self.df = pd.concat([self.df, has_ivy], axis=1)




    def create_interaction_features(self):
        """
        Not currently using this. Not sure why I put converting the zip_radius value here. Need to move
        """
    #     self.df['rating_x_number_of_ratings'] = self.df['rating'] * \
    #                                             self.df['number_of_ratings']
        # self.df['lat_x_lon'] = self.df['lat'] * self.df['lon']

        self.df['zip_radius'] = self.df['zip_radius'].apply(lambda x: \
                                                            np.float(x))
        # self.df['zip_radius_x_lat_x_lon'] = self.df['zip_radius'] * \
        #                                                 self.df['lat_x_lon']
        # self.df['avg_review_length_x_sent_score'] = \
        #                                     self.df['avg_review_length'] * \
        #                                     self.df['avg_review_sentiment']
        # self.df['number_of_reviews_x_avg_review_length'] = \
        #                                     self.df['number_of_reviews'] * \
        #                                     self.df['avg_review_length']
        # self.df['number_of_reviews_x_avg_review_sentiment'] = \
        #                                     self.df['number_of_reviews'] *\
        #                                     self.df['avg_review_sentiment']
        # self.df['number_of_ratings_x_number_of_reviews'] = \
        #                                     self.df['number_of_ratings'] *\
        #                                     self.df['number_of_reviews']


    def sort_columns(self):
        """
        Sort feature columns (not including possible_subjects, qualified_subjects, or cat_features) by alphabetical order. Concatenate alphabetically sorted columns that were excluded above as groups.
        """
        tmp_columns = self.df.columns.tolist()

        for subj in self.possible_subjects:
            tmp_columns.remove(subj)

        for subj in self.qualified_subjects:
            tmp_columns.remove(subj)

        for subj in self.cat_features:
            tmp_columns.remove(subj)

        tmp_columns.sort()
        self.possible_subjects.sort()
        self.qualified_subjects.sort()
        self.cat_features.sort()

        tmp_columns.extend(self.cat_features)
        tmp_columns.extend(self.possible_subjects)
        tmp_columns.extend(self.qualified_subjects)


        self.df = self.df.reindex_axis(tmp_columns, axis=1)

    def build_tutor_subjects_table(self):
        """
        Convert tutor's tutoring subjects into values in new field

        FROM:
        | tutor_id | Math | English |
        -----------------------------
        |   101    |   1  |    0    |
        |   102    |   0  |    1    |
        |   103    |   1  |    1    |

        TO:
        |   tutor_id    |   tutoring_subject  |
        ---------------------------------------
        |      101      |         Math        |
        |      102      |        English      |
        |      103      |         Math        |
        |      103      |        English      |


        """
        if 'tutor_id' not in self.df.columns:
            self.df.reset_index(level=0, inplace=True)
            self.df.rename(columns={'index':'tutor_id'}, inplace=True)

        ts_table = self.df[['tutor_id'] + self.possible_subjects]
        ts_table = pd.melt(ts_table, \
                            id_vars='tutor_id', \
                            var_name='tutoring_subject')
        ts_table = ts_table[ts_table['value']==True]
        ts_table.drop('value', axis=1, inplace=True)
        ts_table = ts_table.sort(columns='tutor_id')
        self.ts_table = ts_table

    def join_ts_table_to_subject_classes(self):
        """
        Add on columns to indicate if subject is "qualified" or "linked"
        """

        def get_subject_classes(ts_row):
            tutor_row = self.df[self.df['tutor_id']==ts_row[0]]
            qual_dict = tutor_row['qual_subjects'].tolist()[0]
            linked_dict = tutor_row['linked_subjects'].tolist()[0]

            qual_list = list(set(itertools.chain.from_iterable(qual_dict.values())))
            linked_list = list(set(itertools.chain.from_iterable(linked_dict.values())))

            if ts_row[1] in qual_list:
                qual = 1
            else:
                qual = 0

            if ts_row[1] in linked_list:
                linked = 1
            else:
                linked = 0

            return Series({'is_qualified':qual, 'is_linked':linked})

        joined = self.ts_table.apply(lambda x: get_subject_classes(x), axis=1)
        self.ts_table = pd.concat([self.ts_table, joined], axis=1)




    def write_sql_tutor_subjects_table(self):
        """ Write out ts_table to csv for later loading to SQL. """

        write_filename = raw_input('Input tutor_subjects_table path-filename: ')
        self.ts_table.to_csv(write_filename, \
                        index=True, \
                        index_label= 'ts_id', \
                        quoting=csv.QUOTE_NONNUMERIC)




    def all_features(self):
        """
        Run all feature parsing methods from above.
        """
        print "running remove_duplicate_tutors()"
        self.remove_duplicate_tutors()

        print 'running just_subjects()'
        self.just_subjects()

        print 'running subjects_to_boolean_features()'
        self.subjects_to_boolean_features()

        print 'running badge_to_hours_and_boolean()'
        self.badge_to_hours_and_boolean()

        print 'running ratings_to_boolean()'
        self.ratings_to_boolean()

        print 'running get_education()'
        self.get_education()

        print 'running student_reviews_to_features()'
        self.student_reviews_to_features()

        print 'running get_response_time()'
        self.get_response_time()

        print 'running background_check_info()'
        self.background_check_info()

        print 'running get_sentiment()'
        self.get_sentiment()

        print 'running zipcode_to_lat_lon()'
        self.zipcode_to_lat_lon()

        print 'running fillna()'
        self.fillna()

        print 'running remove_unpopular_subjects()'
        self.remove_unpopular_subjects()

        print 'running create_interaction_features()'
        self.create_interaction_features()

        print 'running get_subject_categories'
        self.get_subject_categories()

        print 'running get_qualified_subjects'
        self.get_qualified_subjects()

        print 'running has_ivy_degree'
        self.has_ivy_degree()

        print 'running sort_columns()'
        self.sort_columns()

        print 'running build_tutor_subjects_table()'
        self.build_tutor_subjects_table()

class clean_to_csv():
    def __init__ (self, clean):
        self.clean = clean

    def write_tutor_attributes_table(self):
        self.tutor_columns = ['tutor_id', 'EdD', 'Enrolled', \
                            'Graduate_Coursework', 'J.D.', 'MBA', 'MD', \
                            'MEd', 'Masters', 'Other', 'PhD', \
                            'avg_review_length', 'badge_hours', \
                            'days_since_last_review', 'has_badge', \
                            'has_rating', 'hourly_rate', 'lat', 'lon', \
                            'name', 'number_of_ratings', 'number_of_reviews', \
                            'profile_picture', 'rating', 'zip_code', \
                            'zip_radius']
        write_filename = raw_input('Input tutor_attributes_table path-filename:\n')
        self.clean.df[self.tutor_columns].to_csv(write_filename, \
                        index=False, \
                        quoting=csv.QUOTE_NONNUMERIC)

    def write_tutor_subjects_table(self):

        write_filename = raw_input('Input tutor_subjects_table path-filename: ')
        self.clean.ts_table.to_csv(write_filename, \
                        index=True, \
                        index_label= 'ts_id', \
                        quoting=csv.QUOTE_NONNUMERIC)

class clean_to_sklearn():

    def __init__ (self, clean):
        self.clean = clean

    def to_X_matrix(self):
        # usable_features = [ 'EdD', 'Enrolled', \
        #                     'Graduate_Coursework', 'J.D.', 'MBA', 'MD', \
        #                     'MEd', 'Masters', 'Other', 'PhD', \
        #                     'avg_review_length', 'badge_hours', \
        #                     'days_since_last_review', 'has_badge', \
        #                     'has_rating', 'lat', 'lon', \
        #                     'number_of_ratings', 'number_of_reviews', \
        #                     'profile_picture', 'rating', 'zip_code', \
        #                     'zip_radius', 'rating_x_number_of_ratings', \
        #                     'has_ivy_degree', 'has_background_check', \
        #                     'days_since_background_check', 'response_time', \
        #                     'avg_review_sentiment', 'lat_x_lon', \
        #                     'zip_radius_x_lat_x_lon', \
        #                     'avg_review_length_x_sent_score', \
        #                     'number_of_reviews_x_avg_review_length', \
        #                     'number_of_reviews_x_avg_review_sentiment', \
        #                     'number_of_ratings_x_number_of_reviews']


        usable_features = [ 'EdD', 'Enrolled', \
                            'Graduate_Coursework', 'J.D.', 'MBA', 'MD', \
                            'MEd', 'Masters', 'Other', 'PhD', \
                            'avg_review_length', 'badge_hours', \
                            'days_since_last_review', \
                            'has_rating', 'lat', 'lon', \
                            'number_of_ratings', 'number_of_reviews', \
                            'profile_picture', 'rating', 'zip_code', \
                            'zip_radius', \
                            'has_ivy_degree', 'has_background_check', \
                            'days_since_background_check', 'response_time', \
                            'avg_review_sentiment']

        # self.interaction_features = ['rating_x_number_of_ratings', \
        #                     'lat_x_lon', \
        #                     'zip_radius_x_lat_x_lon', \
        #                     'avg_review_length_x_sent_score', \
        #                     'number_of_reviews_x_avg_review_length', \
        #                     'number_of_reviews_x_avg_review_sentiment', \
        #                     'number_of_ratings_x_number_of_reviews']



        usable_features.extend(self.clean.cat_features)
        usable_features.extend(self.clean.possible_subjects)
        usable_features.extend(self.clean.qualified_subjects)
        self.clean.usable_features = usable_features

        # self.clean.df['zip_radius'] = self.clean.df['zip_radius'].apply(lambda x: \
        #                                                     np.float(x))
        self.clean.df['zip_code'] = self.clean.df['zip_code'].apply(lambda x: np.int(x))

        X = self.clean.df[usable_features].as_matrix().astype(np.float)
        scaler_X = preprocessing.StandardScaler().fit(X) # Normalize
        X = scaler_X.transform(X)

        return X, scaler_X

    def to_y_array(self):
        y = self.clean.df['hourly_rate'].as_matrix().astype(np.float)
        scaler_y = preprocessing.StandardScaler().fit(y) # Normalize
        y = scaler_y.transform(y)

        return y, scaler_y

class only_badges_to_sklearn():

    def __init__ (self, clean):
        self.clean = clean

    def to_X_matrix(self):
        # usable_features = [ 'EdD', 'Enrolled', \
        #                     'Graduate_Coursework', 'J.D.', 'MBA', 'MD', \
        #                     'MEd', 'Masters', 'Other', 'PhD', \
        #                     'avg_review_length', 'badge_hours', \
        #                     'days_since_last_review', \
        #                     'has_rating', 'lat', 'lon', \
        #                     'number_of_ratings', 'number_of_reviews', \
        #                     'profile_picture', 'rating', 'zip_code', \
        #                     'zip_radius', 'rating_x_number_of_ratings', \
        #                     'has_ivy_degree', 'has_background_check', \
        #                     'days_since_background_check', 'response_time', \
        #                     'avg_review_sentiment', 'lat_x_lon', \
        #                     'zip_radius_x_lat_x_lon', \
        #                     'avg_review_length_x_sent_score', \
        #                     'number_of_reviews_x_avg_review_length', \
        #                     'number_of_reviews_x_avg_review_sentiment', \
        #                     'number_of_ratings_x_number_of_reviews']

        usable_features = [ 'EdD', 'Enrolled', \
                            'Graduate_Coursework', 'J.D.', 'MBA', 'MD', \
                            'MEd', 'Masters', 'Other', 'PhD', \
                            'avg_review_length', 'badge_hours', \
                            'days_since_last_review', \
                            'has_rating', 'lat', 'lon', \
                            'number_of_ratings', 'number_of_reviews', \
                            'profile_picture', 'rating', 'zip_code', \
                            'zip_radius', \
                            'has_ivy_degree', 'has_background_check', \
                            'days_since_background_check', 'response_time', \
                            'avg_review_sentiment']

        usable_features.extend(self.clean.cat_features)
        usable_features.extend(self.clean.possible_subjects)
        usable_features.extend(self.clean.qualified_subjects)
        self.clean.usable_features = usable_features

        # self.clean.df['zip_radius'] = self.clean.df['zip_radius'].apply(lambda x: \
        #                                                     np.float(x))
        self.clean.df['zip_code'] = self.clean.df['zip_code'].apply(lambda x: np.int(x))

        X = self.clean.df[self.clean.df['has_badge']==1][usable_features].as_matrix().astype(np.float)
        scaler_X = preprocessing.StandardScaler().fit(X) # Normalize
        X = scaler_X.transform(X)

        return X, scaler_X

    def to_y_array(self):
        y = self.clean.df[self.clean.df['has_badge']==1]['hourly_rate'].as_matrix().astype(np.float)
        scaler_y = preprocessing.StandardScaler().fit(y) # Normalize
        y = scaler_y.transform(y)

        return y, scaler_y

def json_to_sklearn(path):
    cleaned = clean(path)
    cleaned.read_json()
    cleaned.all_features()
    clean_sk = clean_to_sklearn(cleaned)

    X, scaler_X = clean_sk.to_X_matrix()
    y, scaler_y = clean_sk.to_y_array()

    return (X, scaler_X, y, scaler_y, cleaned)

def json_to_sklearn_only_badges(path):
    cleaned = clean(path)
    cleaned.read_json()
    cleaned.all_features()
    clean_sk = only_badges_to_sklearn(cleaned)

    X, scaler_X = clean_sk.to_X_matrix()
    y, scaler_y = clean_sk.to_y_array()

    return (X, scaler_X, y, scaler_y, cleaned)


