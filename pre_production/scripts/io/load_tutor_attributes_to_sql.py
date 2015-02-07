import pymysql as mdb
# from progressbar import ProgressBar

con = mdb.connect('localhost', 'insight', 'dataimpact', 'wyzantv1') #host, user, password, #database

with con:
    with open('../../data/tutor_attributes_table.csv') as fin:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS tutor_attributes_nyc")
        cur.execute("CREATE TABLE tutor_attributes_nyc( \
            tutor_id INT PRIMARY KEY, \
            EdD BOOL, \
            Enrolled BOOL, \
            Graduate_Coursework BOOL, \
            JD BOOL, \
            MBA BOOL, \
            MD BOOL, \
            MEd BOOL, \
            Masters BOOL, \
            Other BOOL, \
            PhD BOOL, \
            avg_review_length FLOAT(5, 2), \
            badge_hours SMALLINT(5), \
            days_since_last_review SMALLINT(5), \
            has_badge BOOL, \
            has_rating BOOL, \
            hourly_rate SMALLINT(5), \
            lat DECIMAL(10, 7), \
            lon DECIMAL(10, 7), \
            name VARCHAR(50), \
            number_of_ratings SMALLINT(5), \
            number_of_reviews SMALLINT(5), \
            profile_picture BOOL, \
            rating FLOAT(2, 1), \
            zip_code VARCHAR(5), \
            zip_radius SMALLINT(3))")

        next(fin) # Ignore header
        pbar = max
        i=0
        for line in fin:
            line = line[:-1]
            print 'Reading line ' + str(i)
            cur.execute('INSERT INTO tutor_attributes_nyc VALUES(%s)' % line)
            i += 1

    fin.close()

con.close()
