import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from itertools import groupby
from src.data import psql_operations


def parse_dates_and_teams(week, year):
    """

    Args:
        week:
        year:

    Returns:

    """
    away_qbs = []
    home_qbs = []
    away_rbs = []
    home_rbs = []
    times = []

    starting_qb_map = {}
    starting_rb_map = {}
    alpha_sorted_teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
                          'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                          'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ',
                          'OAK', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WSH']
    db_connection = psql_operations.PostgresConnection()
    team_df = db_connection.get_db_data(f'SELECT * FROM \"2017_matchups\" WHERE week = {week} ORDER BY date')
    dates = team_df['date'].tolist()
    days = [x.day for x in dates]
    team_count_on_date = [len(list(group)) for key, group in groupby(days)]
    home_teams = team_df['hometeam'].tolist()
    away_teams = team_df['awayteam'].tolist()

    # ourlads uses ARZ instead of ARI for some stupid reason
    url = 'http://www.ourlads.com/nfldepthcharts/depthcharts.aspx'
    depth_chart_html = requests.get(url)
    player_soup = BeautifulSoup(depth_chart_html.content, 'lxml')
    qbs = player_soup.find_all('td', string=re.compile('QB'))
    rbs = player_soup.find_all('td', string=re.compile('RB'))

    # DET, NE, NYJ all list 2 RBs as starters. This eliminates the second starter for each. Also it is a horrible way
    # of doing so.
    for index in [11, 22, 26]:
        del rbs[index]

    for qbTag, team in zip(qbs, alpha_sorted_teams):
        last, first = qbTag.next_sibling.next_sibling.text.strip().split(' ')[:2]
        starting_qb_map[team] = first.lower().replace("'", "") + ' ' + last.lower()[:-1].replace("'", "")

    for rbTag, team in zip(rbs, alpha_sorted_teams):
        last, first = rbTag.next_sibling.next_sibling.text.strip().split(' ')[:2]
        starting_rb_map[team] = first.lower().replace("'", "") + ' ' + last.lower()[:-1].replace("'", "")

    for homeTeam in home_teams:
        home_qbs.append(starting_qb_map[homeTeam])
        home_rbs.append(starting_rb_map[homeTeam])

    for awayTeam in away_teams:
        away_qbs.append(starting_qb_map[awayTeam])
        away_rbs.append(starting_rb_map[awayTeam])

    write_players_to_db(home_teams, away_qbs, home_qbs, away_rbs, home_rbs, week)

    return [away_teams, team_count_on_date, home_teams, times, dates, away_qbs, home_qbs, away_rbs, home_rbs]


def week_html_parsing(soup):
    team_to_abbr = {
        'Oakland Raiders': 'oak',
        'New England Patriots': 'ne',
        'Buffalo Bills': 'buf',
        'Houston Texans': 'hou',
        'New Orleans Saints': 'no',
        'Carolina Panthers': 'car',
        'Cincinnati Bengals': 'cin',
        'Cleveland Browns': 'cle',
        'Green Bay Packers': 'gb',
        'Detroit Lions': 'det',
        'Seattle Seahawks': 'sea',
        'Jacksonville Jaguars': 'jax',
        'New York Jets': 'nyj',
        'Kansas City Chiefs': 'kc',
        'Denver Broncos': 'den',
        'Miami Dolphins': 'mia',
        'Tampa Bay Buccaneers': 'tb',
        'Minnesota Vikings': 'min',
        'Arizona Cardinals': 'ari',
        'New York Giants': 'nyg',
        'Tennessee Titans': 'ten',
        'Pittsburgh Steelers': 'pit',
        'Indianapolis Colts': 'ind',
        'Baltimore Ravens': 'bal',
        'Dallas Cowboys': 'dal',
        'San Diego Chargers': 'lac',
        'Los Angeles Chargers': 'lac',
        'St. Louis Rams': 'lar',
        'Los Angeles Rams': 'lar',
        'San Francisco 49ers': 'sf',
        'Chicago Bears': 'chi',
        'Washington Redskins': 'wsh',
        'Philadelphia Eagles': 'phi',
        'Atlanta Falcons': 'atl'
    }

    def parse_html_table(table):
        n_columns = 0
        n_rows = 0
        column_names = ['Week', 'Day', 'Date', 'Time', 'Visitor', '@', 'Home', 'link',
                        'PtsW', 'PtsL', 'YdsW', 'TOW', 'YdsL', 'TOL']

        # Find number of rows and columns
        # we also find the column titles if we can
        for row in table.find_all('tr'):

            # Determine the number of rows in the table
            td_tags = row.find_all('td')
            if len(td_tags) > 0:
                n_rows += 1
                if n_columns == 1:
                    # Set the number of columns for our table
                    n_columns = len(td_tags)

            # Handle column names if we find them
            # th_tags = row.find_all('th')
            # if len(th_tags) > 0 and len(column_names) == 0:
            #     for th in th_tags:
            #         column_names.append(th.get_text())

        # Safeguard on Column Titles
        # if len(column_names) > 0 and len(column_names) != n_columns:
        #     raise Exception("Column titles do not match the number of columns")

        columns = column_names if len(column_names) > 0 else range(0, n_columns)
        df = pd.DataFrame(columns=columns,
                          index=range(0, n_rows))
        row_marker = 0
        for row in table.find_all('tr'):
            column_marker = 1
            columns = row.find_all('td')
            first_col = row.find('th')
            df.iloc[row_marker, 0] = first_col.get_text()
            for column in columns:
                if column.get_text() in team_to_abbr:
                    df.iloc[row_marker, column_marker] = team_to_abbr[column.get_text()]
                else:
                    df.iloc[row_marker, column_marker] = column.get_text()
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1

        # Convert to float if possible
        for col in df:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

        return df

    return [(table['id'], parse_html_table(table)) for table in soup.find_all('table')]


def write_players_to_db(home_teams, away_qbs, home_qbs, away_rbs, home_rbs, week):
    """

    Args:
        home_teams:
        away_qbs:
        home_qbs:
        away_rbs:
        home_rbs:
        week:
    """
    db_connection = psql_operations.PostgresConnection()
    for home_team, away_qb, home_qb, away_rb, home_rb in zip(home_teams, away_qbs, home_qbs, away_rbs, home_rbs):
        query_strings = ['UPDATE \"2017_matchups\" SET awayqb=\'{}\' '
                         'WHERE hometeam=\'{}\' AND week={}'.format(away_qb, home_team, week),
                         'UPDATE \"2017_matchups\" SET homeqb=\'{}\' '
                         'WHERE hometeam=\'{}\' AND week={}'.format(home_qb, home_team, week),
                         'UPDATE \"2017_matchups\" SET awayrb=\'{}\' '
                         'WHERE hometeam=\'{}\' AND week={}'.format(away_rb, home_team, week),
                         'UPDATE \"2017_matchups\" SET homerb=\'{}\' '
                         'WHERE hometeam=\'{}\' AND week={}'.format(home_rb, home_team, week)]
        db_connection.set_db_data(query_strings)
    print('Starters updated successfully.')
