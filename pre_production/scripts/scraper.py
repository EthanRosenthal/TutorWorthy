"""
Scrapes profile info from all NYC tutors from www.wyzant.com

Specify output file name at bottom in main() function

Author: Ethan Rosenthal
Last Modified: 2/7/2015
"""

from BeautifulSoup import BeautifulSoup
import urllib2
import pandas as pd
from pandas import DataFrame, Series
import json
import re
import time


class region_crawler():

    def __init__(self, base_url='http://www.wyzant.com/', \
                region_url='http://www.wyzant.com/New_York_City_tutors.aspx?sl=80075877&sort=27&pagesize=5&pagenum=', \
                page=1, outfile='tmp.txt'):
        self.page = page
        self.idx = 0
        self.base_url = base_url
        self.region_url = region_url
        self.current_url = ''
        self.outfile = outfile

    def scrape_pages(self, page):
        """
        Main scraping function. Moves through each page of the list of NYC tutors. Statement to check for the end of the list is not currently working. Need to ctrl-c out of the program to stop scraping.
        """

        while True:
            print 'Scraping page ' + str(self.page)

            # Format of url is http://www.wyzant.com/New_York_City_tutors.aspx?sl=80075877&sort=27&pagesize=5&pagenum=1 where the 1 at the end is the page number. Increase page number by 1 each iteration of while loop in order to crawl through pages.
            self.current_url = self.region_url + str(self.page)
            ufile = urllib2.urlopen(self.current_url)

            if ufile.geturl()[-1:] == self.base_url:
                break  # Check for end of list

            people = BeautifulSoup(ufile).findAll('div', {'class':'tutorFR hide medium-show'}) # picks out each person on the page

            self.scrape_people(people)

            self.page += 1 # Next page

    def scrape_people(self, people):
        """
        Scrapes profile info from all people on a given page of WyzAnt. Runs all class methods for picking out different profile features.
        """

        for person in people:
            name = ''
            hourly_rate = int(-1)
            self.current_url = self.base_url + person.find('a')['href']

            # Should do a better version of try/excepting of connection issues.
            try:
                person_ufile = urllib2.urlopen(self.current_url)
            except:
                time.sleep(1)
                person_ufile = urllib2.urlopen(self.current_url)


            soup = BeautifulSoup(person_ufile)

            # Get various features by running all class methods.
            name = self.get_name(soup)
            hourly_rate = self.get_hourly_rate(soup)
            raw_subjects, qual_subjects, linked_subjects = self.get_subjects(soup)
            education = self.get_education(soup)
            badge = self.get_tutor_badge(soup)
            profile_picture = self.has_profile_picture(soup)
            rating, number_of_ratings = self.get_rating(soup)
            zip_radius, zip_code = self.get_zip_code(soup)
            student_reviews = self.get_student_reviews(soup)
            background_check = self.get_background_check(soup)
            response_time = self.get_response_time(soup)
            bio = self.get_bio(soup)

            print name

            # Write features to dictionary.
            person_row = {}
            person_row['name'] = name
            person_row['hourly_rate'] = hourly_rate
            person_row['raw_subjects'] = raw_subjects
            person_row['qual_subjects'] = qual_subjects
            person_row['linked_subjects'] = linked_subjects
            person_row['education'] = education
            person_row['badge'] = badge
            person_row['profile_picture'] = profile_picture
            person_row['rating'] = rating
            person_row['number_of_ratings'] = number_of_ratings
            person_row['zip_radius'] = zip_radius
            person_row['zip_code'] = zip_code
            person_row['student_reviews'] = student_reviews
            person_row['background_check'] = background_check
            person_row['response_time'] = response_time
            person_row['bio'] = bio
            person_row['url'] = self.current_url

            # Write dictionary to JSON ouput file.
            json.dump(person_row, self.outfile)
            self.outfile.write('\n')

            self.idx += 1

    def get_name(self, soup):
        """
        Get tutors name. Just returns first name and last initial. ex: Ethan R.
        """
        name = soup.find('h1', {'itemprop':'name'}).renderContents()
        return name

    def get_hourly_rate(self, soup):
        """
        Get tutor's hourly rate.
        """
        hourly_rate = int(soup.find('span', {'class':'price-amt'}).renderContents())
        return hourly_rate

    def get_subjects(self, soup):
        """
        Get subjects that tutor tutors. Three classes of subjects:
        - qualified (tutor took qualification test)
        - linked (tutor has a hover link with more info about themselves in relation to the subject)
        - raw (Neither qualified nor linked)

        If the topic is "qualified" then search span class: 'qual'
        If the topic is linked to, search 'a' class: 'profile-subjectDescLink'
        If there's no link and topic is not, then just need the text... getText()
        """

        # Pick out subjects
        topics = soup.findAll('div',{'class':'spc-sm-s profile-txt-body profile-subjectGroup'})

        raw_subjects = {}
        qual_subjects = {}
        linked_subjects = {}

        for topic in topics:
            qual_subjects_list = []
            linked_subjects_list = []

            topic_text = topic.find('div', {'class':'txt-semibold'}).getText().rstrip(':')
            raw_subjects[topic_text] = topic.getText().split(':')[1].split(',')

            # Pick out qualified subjects
            qual_subject_spans = topic.findAll('span', {'class':'qual'})
            for subject in qual_subject_spans:
                qual_subjects_list.append(subject.renderContents().rstrip(','))
            qual_subjects[topic_text] = qual_subjects_list

            # Pick out linked subjects
            linked_subject_links = topic.findAll('a', {'class':'profile-subjectDescLink'})
            for subject in linked_subjects_list:
                linked_subjects_list.append(subject.renderContents())
            linked_subjects[topic_text] = linked_subjects_list

        return raw_subjects, qual_subjects, linked_subjects

    def get_education(self, soup):
        """
        Get tutor's educational degrees. Return dictionary with keys =  University and values = Degree (MBA, MD, etc...)
        """
        try:
            ed_sec = soup.find('div', {'id':'EducationSection'})
            degrees = ed_sec.findAll('div', {'class':'profile-subsection'})
            education = {}

            for degree in degrees:
                university = degree.find('h5').renderContents()
                if education.has_key(university) == False:
                    education[university] = []
                try:
                    degree = degree.getText().split(university)
                    # Handle HTML encoding
                    if degree[1] == 'Master&#39;s':
                        degree[1] = 'Masters'
                    education[university].append(degree[1])
                except:
                    education[university].append('')
        except:
            education = {}
            education['no_university'] = ''

        return education

    def get_tutor_badge(self, soup):
        """
        If tutor has badge (star image next to profile pic with number of hours tutored), then get HTML information about the badge. Includes nuber of hours tutored in the text.
        """
        try:
            badge = soup.find('img', {'class':'TutorBadge'})['title']
        except:
            badge = ''
        return badge

    def has_profile_picture(self, soup):
        """
        Find if tutor has a profile picture. Return boolean value.
        """

        img_src = soup.find('img', {'id':'ImageTutor'})['src']

        if img_src.endswith('default.gif'):
            profile_picture = False
        else:
            profile_picture = True

        return profile_picture

    def get_rating(self, soup):
        """
        Get rating and number of ratings.
        """
        try:
            rating_div = soup.find('div', {'class':'tutor-stat tutor-stat-rating'})
            rating = float(rating_div.find('span').renderContents())
            number_of_ratings = rating_div.find('a').renderContents() # ex. (164 ratings)
            number_of_ratings = int(re.findall('\d+', number_of_ratings)[0])
        except:
            rating = None
            number_of_ratings = 0


        return rating, number_of_ratings

    def get_zip_code(self, soup):
        """
        Get zip code and "zip radius" (distance tutor is willing to travle for tutoring)
        """

        try:
            zip_tag = soup.find('h3', {'class':'icon-hdr-travel'})
            zip_tag = zip_tag.findNextSibling().findChild().renderContents()
            zip_radius, zip_code = re.findall('\d+', zip_tag)
        except:
            zip_radius, zip_code = None, None

        return zip_radius, zip_code

    def get_student_reviews(self, soup):

        review_list = []
        reviews = soup.findAll('div', {'class':'review'})
        try:
            for review in reviews:
                review_dict = {}
                review_dict['title'] = review.find('h4').renderContents()
                review_dict['text'] = review.find('p').renderContents()
                review_dict['author_info'] = review.find('h5').renderContents()
                review_list.append(review_dict)
        except:
            pass # No student reviews

        return review_list

    def get_response_time(self, soup):
        """
        Get average time it takes tutor to respond to emails. Returns string. Parse string for actual time later in cleanup_tutor_features.py
        """

        response_time = soup.find('div', \
            {'class':'tutor-stat tutor-stat-response'})
        try:
            response_time = response_time.find('span').renderContents()
        except:
            response_time = ''

        return response_time

    def get_background_check(self, soup):
        """
        Get text associated with background check. If tutor has a background check, text mentions date of background check. Parse text later in cleanup_tutor_features.py.
        """

        try:
            bg = soup.find('span', \
                            {'class':'tutor-stat-bc-date'}).renderContents()
        except:
            bg = ''

        return bg

    def get_bio(self, soup):
        """
        Get text associated with tutor's personal biography.
        """
        try:
            bio = soup.find('p', {'class':'freeResponse'}).renderContents()
        except:
            bio = ''

        return bio





def main():

    with open('../data/nyc_20150130.txt', 'a') as outfile:
        base_url = 'http://www.wyzant.com'
        region_url = 'http://www.wyzant.com/New_York_City_tutors.aspx?sl=80075877&sort=27&pagesize=5&pagenum='
        page = 1 # max pages = 1165

        crawler = region_crawler(base_url, region_url, page, outfile)
        crawler.scrape_pages(page)
    outfile.close()

if __name__ == '__main__':
    main()