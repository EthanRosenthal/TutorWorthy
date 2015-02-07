CREATE TABLE zcta (
    zip CHAR(5) NOT NULL,
    city VARCHAR(64),
    state CHAR(2),
    latitude DECIMAL(6,4),
    longitude DECIMAL(6,4),
    timezone INT(11),
    dst CHAR(1),
    PRIMARY KEY (zip)
    );
