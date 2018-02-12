import psycopg2
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from src.data import html_parsing
import configparser


class PostgresConnection:
    """Establishes a connection to a Postgres DB using credentials in config.txt, in order to retrieve data for training.

    """
    def __init__(self):
        config = configparser.ConfigParser()
        config.read("../../config.txt")
        self.host = config.get('psql database', 'host')
        self.db_name = config.get('psql database', 'db_name')
        self.user = config.get('psql database', 'user')
        self.password = config.get('psql database', 'password')
        self.port = config.get('psql database', 'port')

    def get_db_data(self, sql_string):
        """Queries the DB specified when instantiating the PostgresConnection class, and converts to a pandas dataframe.

        Args:
            sql_string (str): the SQL query to execute

        Returns:
            pandas dataframe: the results of the query, using the column headers provided in the DB

        """
        connection_string = f"""
        host='{self.host}' 
        dbname='{self.db_name}' 
        user='{self.user}' 
        password='{self.password}' 
        port='{self.port}'
        """

        with psycopg2.connect(connection_string) as connection:
            cursor = connection.cursor()
            cursor.execute(sql_string)

            dataframe = pd.DataFrame(cursor.fetchall())
            dataframe.columns = [desc[0] for desc in cursor.description]

        return dataframe

    def set_db_data(self, sql_strings):
        """Writes to the DB specified when instantiating the PostgresConnection class using the supplied SQL string.

        Args:
            sql_strings (:obj:`list` of :obj:`str`): list of SQL strings to execute

        """
        connection_string = f"""
        host='{self.host}' 
        dbname='{self.db_name}' 
        user='{self.user}' 
        password='{self.password}' 
        port='{self.port}'
        """

        with psycopg2.connect(connection_string) as connection:
            cursor = connection.cursor()
            for query in sql_strings:
                cursor.execute(query)

            connection.commit()

    @staticmethod
    def format_profootball_dates(df, year):
        """pro-football-reference dates are in an inconsistent format for presentation. Converts to PST

        Args:
            df (pandas.DataFrame): dataframe to fix the dates of
            year (str): year to append to the datetime

        Returns:
            pandas.DataFrame with reformatted date in 'datetime' column

        """

        def time_apply(x):
            time_list = x.split(':')
            hour = int(time_list[0])
            minute = time_list[1][:-2]
            midi = time_list[1][2:]
            if midi == 'PM':
                hour = str(hour + 9)
            else:
                hour = str(hour - 3)
            if len(hour) == 1:
                hour = '0' + hour
            return hour + ':' + minute

        def date_apply(x):
            month, day = x.split()
            if len(day) == 1:
                day = '0' + day
            return month + ' ' + day

        df['Time'] = df['Time'].apply(lambda x: time_apply(x))
        df['Date'] = df['Date'].apply(lambda x: date_apply(x))
        df['datetime'] = df.Time + df.Date + year
        df['datetime'] = df.datetime.apply(lambda x: datetime.strptime(x, '%H:%M%B %d%Y'))
        return df.drop(['Day', 'Date', 'Time'], 1)

    def fill_matchup_table_with_games(self, week, year):
        """Gets the matchups of a particular week and year from www.pro-football-reference.com, and fills the matchup
        table in the DB with them.

        Args:
            week (str): week to get the matchups from
            year (str): year to get the matchups from

        """
        url = f'https://www.pro-football-reference.com/years/{year}/games.htm'
        schedule_html = requests.get(url)
        pro_soup = BeautifulSoup(schedule_html.content, 'html.parser')

        matchup_table = html_parsing.week_html_parsing(pro_soup)[0][1]
        matchup_table = matchup_table[matchup_table['Time'] != '']
        matchup_table = matchup_table.dropna()

        matchup_table = self.format_profootball_dates(matchup_table, year)

        week_matchups = matchup_table[matchup_table['Week'] == float(week)]
        sql_queries = []
        for i, row in week_matchups.iterrows():
            sql_queries.append("INSERT INTO \"2017_matchups\" (hometeam, awayteam, week, date) "
                               "VALUES ({}, {}, {}, {});".format(
                                row.Home.upper(), row.Visitor.upper(), row.Week, row.datetime))
        self.set_db_data(sql_queries)
        print('Table filled successfully.')
