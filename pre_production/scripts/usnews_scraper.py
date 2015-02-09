import urllib2
from BeautifulSoup import BeautifulSoup
import re
from HTMLParser import HTMLParser
import json
import sys

class usnews_crawler():

    def __init__(self):
        self.base_url = 'http://colleges.usnews.rankingsandreviews.com/best-colleges/rankings/national-universities/data/page+'
        self.page = 1
        self.current_url = self.base_url + str(self.page)
        self.numbers_regex = re.compile('\d+')


    def get_school_and_rank(self, outfile):

        rows = self.soup.findAll('tr', {'valign':'top'})
        for row in rows:
            rank = re.findall(self.numbers_regex, \
                                   row.find('span').renderContents())
            current_school = row.find('a', {'class':'school-name'}).contents[0]
            current_school = BeautifulSoup(current_school, convertEntities=BeautifulSoup.HTML_ENTITIES)
            current_school = ''.join([i if ord(i)<128 else ' ' for i in current_school.contents[0]])
            current_school = ' '.join([word.lower() for word in current_school.split(' ')])
            print current_school + ' ' + str(rank)

            json.dump(dict(zip([current_school], rank)), outfile)
            outfile.write('\n')

    def get_current_url(self):
        self.current_url = self.base_url + str(self.page)

    def get_soup(self):
        self.soup = BeautifulSoup(urllib2.urlopen(self.current_url))

    def crawl(self, outfile_name):

        with open(outfile_name, 'w') as outfile:
            page_rankings = {}
            while self.page<12:
                print 'Scraping page ' + str(self.page)
                self.get_current_url()
                self.get_soup()
                self.get_school_and_rank(outfile)
                self.page += 1

            outfile.close()

if __name__ == '__main__':
    outfile = str(sys.argv[1])
    usnews_crawler = usnews_crawler()
    usnews_crawler.crawl(outfile)


