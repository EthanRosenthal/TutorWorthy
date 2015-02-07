import pandas as pd

def zip_to_latlon(user_zip):
    zips = pd.read_csv('./data/zipcode.csv', dtype={'zip':'str'})

    try:
        loc = zips[zips['zip']==user_zip]
        lat = loc['latitude'].values[0]
        lon = loc['longitude'].values[0]
        return (lat, lon)
    except:
        # Found these two missing zip codes on
        # http://www.maptechnica.com
        if user_zip == '10065':
            lat = 40.76490050000000
            lon = -73.96243050000000
            return (lat, lon)
        elif user_zip == '10075':
            lat = 40.77355900000000
            lon = -73.95606900000000
            return (lat, lon)
        else:
            return (False, False)
