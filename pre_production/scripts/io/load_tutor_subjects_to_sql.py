import pymysql as mdb
# from progressbar import ProgressBar

con = mdb.connect('localhost', 'insight', 'dataimpact', 'wyzantv1') #host, user, password, #database

with con:
    with open('../../data/tutor_subjects_table.csv') as fin:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS tutor_subjects_nyc")
        cur.execute("CREATE TABLE tutor_subjects_nyc( \
            ts_id INT PRIMARY KEY, \
            tutor_id INT, \
            tutoring_subject VARCHAR(64), \
            is_linked BOOL, \
            is_qualified BOOL)")

        next(fin) # Ignore header
        pbar = max
        i=0
        for line in fin:
            line = line[:-1]
            print 'Reading line ' + str(i)
            cur.execute('INSERT INTO tutor_subjects_nyc VALUES(%s)' % line)
            i += 1

    fin.close()

con.close()
