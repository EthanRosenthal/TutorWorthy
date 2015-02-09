import urllib2
from BeautifulSoup import BeautifulSoup
import re
from HTMLParser import HTMLParser
import json
import sys


def scrape_page():
    url = 'http://en.wikipedia.org/wiki/List_of_colloquial_names_for_universities_and_colleges_in_the_United_States'
    soup = BeautifulSoup(urllib2.urlopen(url))

    names = soup.findAll('li')

    school_aliases = {}
    for name in names[28:405]: # Relevant rows of list
        alias = name.renderContents().split('-')[0]
        schools = name.findAll('a')
        for school in schools:
            school = school.renderContents()
            if 'or' in alias: # 'UR or U of R'
                alias = alias.split(' or ')
            if school_aliases.has_key(school):
                school_aliases[school].append(alias)
            else:
                school_aliases[school] = [alias]

    # Get rid of nested lists from 'or' split
    for k, v in school_aliases.iteritems():
        if any(isinstance(x, list) for x in v):
            school_aliases[k] = v[0]

    return school_aliases



if __name__=='__main__':
    school_aliases = scrape_page()

    with open(sys.argv[1], 'w') as outfile:
        json.dump(school_aliases, outfile)
        outfile.close()
