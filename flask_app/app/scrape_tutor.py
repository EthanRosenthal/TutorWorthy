from BeautifulSoup import BeautifulSoup
import urllib2
from pandas import DataFrame, Series
import re
import time


class crawler():

    def __init__(self, tutor_url):
        self.url = tutor_url

    def scrape_tutor(self):

        name = ''
        hourly_rate = int(-1)

        # Should do a better version of try/excepting

        try:
            person_ufile = urllib2.urlopen(self.url)
        except:
            pass
            try:
                time.sleep(1)
                person_ufile = urllib2.urlopen(self.url)
            except:
                return False


        soup = BeautifulSoup(person_ufile)

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

        # print name
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
        person_row['url'] = self.url

        return Series(person_row)

    def get_name(self, soup):

        name = soup.find('h1', {'itemprop':'name'}).renderContents()
        return name

    def get_hourly_rate(self, soup):

        hourly_rate = int(soup.find('span', {'class':'price-amt'}).renderContents())
        return hourly_rate

    def get_subjects(self, soup):
        """
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
        try:
            badge = soup.find('img', {'class':'TutorBadge'})['title']
        except:
            badge = ''
        return badge

    def has_profile_picture(self, soup):

        img_src = soup.find('img', {'id':'ImageTutor'})['src']

        if img_src.endswith('default.gif'):
            profile_picture = False
        else:
            profile_picture = True

        return profile_picture

    def get_rating(self, soup):
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

        response_time = soup.find('div', \
            {'class':'tutor-stat tutor-stat-response'})
        try:
            response_time = response_time.find('span').renderContents()
        except:
            response_time = ''

        return response_time

    def get_background_check(self, soup):

        try:
            bg = soup.find('span', \
                            {'class':'tutor-stat-bc-date'}).renderContents()
        except:
            bg = ''

        return bg


def main(tutor_url):

    crawl = crawler(tutor_url)
    tutor_df = crawl.scrape_tutor()
    return tutor_df
