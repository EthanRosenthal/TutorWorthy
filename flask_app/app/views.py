from flask import render_template, request
from app import app
from zip_to_latlon import zip_to_latlon
import numpy as np
import pickle
from sklearn.externals import joblib
import scrape_tutor
import cleanup_tutor_features
import pandas as pd
from pandas import DataFrame, Series
import recommender

"""
Would be cool to do gender study.
http://www.wyzant.com/TutorSearch?kw=&z=10025&d=20&im=0 for all female tutors.
http://www.wyzant.com/TutorSearch?kw=&z=10025&d=20&im=0 for all male tutors.
"""


# global clf
# clf = joblib.load('./data/gb.joblib')
global df
df = pickle.load(open('./data/tutor_df_20150201_dropRaquel.pkl', 'rb'))
global possible_subjects
possible_subjects = pickle.load(open('./data/possible_subjects.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
        title = 'TutorWorthy'
        )

@app.route('/')
@app.route('/new_index')
def new_index():
    return render_template("new_index.html",
        title = 'Home', user = { 'nickname': 'Miguel' },
        )

@app.route('/other_input')
def other_input():
    return render_template("other_input.html")

@app.route('/example_output')
def example_output():

    example_tutor_url = str(request.args.get('example_tutor_url'))

    examples = {}
    examples['Elliott L.'] = "http://www.wyzant.com/Tutors/NY/New_York/7875805/?z=10116&d=20&sl=80075877&sort=27" # $50/hr -> $79
    examples['Erica S.'] = "http://www.wyzant.com/Tutors/NY/New_York/8671128/?z=10116&d=20&sl=80075877&sort=27" # $40/hr -> $49
    examples['Jared A.'] = "http://www.wyzant.com/Tutors/NY/New_York/8565916/?z=10116&d=20&sl=80075877&sort=27" # $55/hr -> $50
    examples['Jennifer T.'] = "http://www.wyzant.com/Tutors/NY/New_York/8462587/?z=10116&d=20&sl=80075877&sort=27" # $100/hr -> $82
    examples['Linda W.'] = "http://www.wyzant.com/Tutors/NY/New_York/8587257/?z=10116&d=20&sl=80075877&sort=27" # $60

    tutor_url = examples[example_tutor_url]

    tutor_name = example_tutor_url


    tutor_data = scrape_tutor.main(tutor_url)
    example, hourly_rate = cleanup_tutor_features.clean_tutor(tutor_data)
    nn, max_tut, min_tut, img_io = recommender.main(example, df, possible_subjects)

    tut_list = []
    tut_list.append(dict(rel='Most similar tutor', \
                    name=nn['name'], \
                    rate=nn['hourly_rate'], \
                    url=nn['url']))
    tut_list.append(dict(rel='Maximum priced tutor', \
                    name=max_tut['name'], \
                    rate=max_tut['hourly_rate'], \
                    url=max_tut['url']))
    tut_list.append(dict(rel='Minimum priced tutor', \
                    name=min_tut['name'], \
                    rate=min_tut['hourly_rate'], \
                    url=min_tut['url']))
    example_tutor = dict(name=tutor_name, rate=hourly_rate, url=tutor_data['url'])

    html = """
            <img src="data:image/png;base64,{}">
        """
    img_io = img_io.getvalue().encode('base64')
    img_io = html.format(img_io)


    return render_template("example_output.html", \
                            tutors=tut_list, \
                            example_tutor=example_tutor, \
                            sim_plot=img_io)

@app.route('/output')
def output():


    tutor_url = str(request.args.get('tutor_url'))
    print tutor_url

    # BEGIN Normalize url
    if tutor_url.startswith("www"):
        tutor_url = "http://" + tutor_url
    elif tutor_url.startswith("wyzant"):
        tutor_url = "http://www." + tutor_url

    if "#" in tutor_url:
        tutor_url = tutor_url.split("#")[0]
    # END Normalize url

    try:
        tutor_data = scrape_tutor.main(tutor_url)
        example, hourly_rate = cleanup_tutor_features.clean_tutor(tutor_data)
        nn, max_tut, min_tut, img_io = recommender.main(example, df, possible_subjects)

        tutor_name = example['name']

        tut_list = []
        tut_list.append(dict(rel='Most similar tutor', \
                        name=nn['name'], \
                        rate=nn['hourly_rate'], \
                        url=nn['url']))
        tut_list.append(dict(rel='Maximum priced tutor', \
                        name=max_tut['name'], \
                        rate=max_tut['hourly_rate'], \
                        url=max_tut['url']))
        tut_list.append(dict(rel='Minimum priced tutor', \
                        name=min_tut['name'], \
                        rate=min_tut['hourly_rate'], \
                        url=min_tut['url']))
        example_tutor = dict(name=tutor_name, rate=hourly_rate, url=tutor_data['url'])

        html = """
            <img src="data:image/png;base64,{}">
        """
        img_io = img_io.getvalue().encode('base64')
        img_io = html.format(img_io)


        return render_template("output.html", \
                        tutors=tut_list, \
                        example_tutor=example_tutor, \
                        sim_plot=img_io)
    except:
        # result = "Please enter correct WyzAnt tutor profile url"
        # print result
        # NEED TO MAKE ERROR PAGE!
        return render_template("error.html")


@app.route('/slides')
def slides():
    return render_template("slides.html")


@app.route('/input')
def cities_input():
    return render_template("input.html")

@app.route('/error')
def error():
    return render_template("error.html")

