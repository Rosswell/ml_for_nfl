import numpy as np
import pandas as pd
import configparser
from src.data import psql_operations


class DataCleaner:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('../../config.txt')
        self.db_connection = psql_operations.PostgresConnection()
        self.player_map = self.db_connection.get_db_data('SELECT * FROM \"player_map_v2\"')
        self.team_map = self.db_connection.get_db_data('SELECT * FROM \"team_map_v2\"')

    def clean_data(self, raw_data, position):
        """Cleans the raw data to feed into the ML pipeline

        Args:
            raw_data (pandas.DataFrame): dataframe of the raw data, as scraped from the internets
            position (str): position that the raw_data_df represents, ie. 'qb' or 'rb'

        Returns:
            pandas.DataFrame: a cleaned dataframe whose stats reflect the cumulative summation for each player at that
            particular time in their career

        """

        # team mapping
        df_teams_mapped = pd.merge(raw_data, self.team_map, left_on='tm', right_on='team', suffixes=('', '_home'))
        df_teams_mapped = pd.merge(df_teams_mapped, self.team_map, left_on='opp_tm', right_on='team', suffixes=('', '_away'))
        # player mapping
        df_teams_players_mapped = pd.merge(df_teams_mapped, self.player_map, on='player', suffixes=('', '_player'))
        agg_data = df_teams_players_mapped.drop(['team', 'team_away', 'entry_id', 'player', 'tm', 'opp_tm'], 1)

        final_agg_data = pd.DataFrame(columns=list(agg_data.columns))
        player_list = agg_data['player_id'].unique()

        for player in player_list:
            player_df = agg_data[agg_data['player_id'] == player].sort_values(['season', 'week'])

            non_agg_cols = player_df[['week', 'season', 'win_loss', 'game_id', 'player_id', 'team_id', 'team_id_away']]
            player_df = player_df.drop(['week', 'season', 'game_id', 'player_id', 'team_id', 'team_id_away'], 1)

            player_df = player_df.rename(index=str, columns={'win_loss': 'agg_win_loss'})
            # creating a dummy row of zeros
            agg_dummy_row = pd.DataFrame({key: 0 for key in list(player_df.columns)}, index=[0])
            # concat the dummy data with the real data to aggregate
            agg_cols_with_dummy_data = pd.concat([agg_dummy_row, player_df], axis=0, ignore_index=True)
            # aggregating the aggregate-able data
            agg_cols_agg_applied = np.cumsum(agg_cols_with_dummy_data, axis=0).reset_index(drop=True)
            agg_cols_agg_applied_normed = pd.DataFrame(columns=list(agg_cols_agg_applied.columns))

            for i, row in agg_cols_agg_applied.iterrows():
                if i != 0:
                    agg_cols_agg_applied_normed = agg_cols_agg_applied_normed.append(row/i)
                else:
                    agg_cols_agg_applied_normed = agg_cols_agg_applied_normed.append(row)
            # dropping the final aggregate (since that's for a game in the future) and joining the agg cols
            # with the non agg cols
            agg_player_data = pd.concat([agg_cols_agg_applied_normed, non_agg_cols.reset_index(drop=True)],
                                        axis=1, join_axes=[agg_cols_agg_applied_normed.index]).iloc[:-1]
            final_agg_data = final_agg_data.append(agg_player_data, ignore_index=True)

        if 'rb' in position:
            # dropping stats irrelevant to RBs
            final_agg_data = final_agg_data.drop(['att', 'cmp', 'int', 'lng', 'rate', 'score_for', 'score_against',
                                                  'sk', 'sk_yds', 'td', 'win_loss', 'yds', 'agg_win_loss'], 1)
        else:
            # dropping stats irrelevant to QBs
            final_agg_data = final_agg_data.drop(['receiving_lng', 'receiving_td', 'receiving_tgt', 'receiving_yds'], 1)

        return final_agg_data

    def get_testing_data(self, cleaned_rb_data, cleaned_qb_data, week, year):
        """gathers and formats data for a particular week and year in order to perform game predictions on that data.

        Args:
            cleaned_rb_data (pandas.DataFrame): the cleaned and aggregated RB data used to form the training data
            cleaned_qb_data (pandas.DataFrame): the cleaned and aggregated QB data used to form the training data
            week (str): the week being tested
            year (str): the year being tested

        Returns:
            pandas.DataFrame: the testing data ready to be fed into the ML pipeline

        """
        testing_data = self.db_connection.get_db_data(f'SELECT * FROM \"2017_matchups\" WHERE week = {week} ORDER BY date')

        # since the matchups table currently has 2 rbs, drop the second of each
        testing_data = testing_data.drop(['homerb2', 'awayrb2'], 1)
        home_qb_data = self.testing_data_formatter(cleaned_qb_data, testing_data, 'homeqb', week, year)
        away_qb_data = self.testing_data_formatter(cleaned_qb_data, testing_data, 'awayqb', week, year)
        home_rb_data = self.testing_data_formatter(cleaned_rb_data, testing_data, 'homerb', week, year)
        away_rb_data = self.testing_data_formatter(cleaned_rb_data, testing_data, 'awayrb', week, year)
        home_player_data = pd.merge(home_qb_data, home_rb_data, on='game_id', suffixes=('_qb', '_rb'))
        away_player_data = pd.merge(away_qb_data, away_rb_data, on='game_id', suffixes=('_qb', '_rb'))

        merged_testing_data = pd.concat([home_player_data, away_player_data], ignore_index=True)
        return merged_testing_data.drop(['index_qb', 'index_rb', 'game_id'], 1)

    def testing_data_formatter(self, training_data, testing_data, testing_column, week, year):
        """gets the most up to date cumulative stats on the players in the testing data, and formats it such that it
        reflects the testing data

        Args:
            training_data (pandas.DataFrame): training data dataframe that was used to train the model
            testing_data (pandas.DataFrame): raw testing data dataframe that contains matchup information
            testing_column (str): the column of data to format, generally in home/away status + position,
            ie. 'homerb' or 'awayqb'
            week (str): the week that's being tested
            year (str): the year that's being tested

        Returns:
            pandas.DataFrame: a formatted dataframe for the position provided via the testing_column

        """
        game_ids = (testing_data['hometeam'] + '_' + testing_data['awayteam']).tolist()
        positional_data = pd.DataFrame(columns=training_data.columns)
        zeroed_data = pd.DataFrame({k: 0 for k in training_data.columns}, index=[0])

        for qb in testing_data[testing_column].tolist():
            player_id = self.player_map[self.player_map['player'] == qb]['player_id'].values[0]
            home_team = testing_data[testing_data[testing_column] == qb]['hometeam'].values[0].lower()
            away_team = testing_data[testing_data[testing_column] == qb]['awayteam'].values[0].lower()
            home_team_id = self.team_map[self.team_map['team'] == home_team]['team_id'].values[0]
            away_team_id = self.team_map[self.team_map['team'] == away_team]['team_id'].values[0]
            try:
                temp_df = training_data[training_data['player_id'] == player_id].iloc[-1]
                temp_df['player_id'] = player_id
                temp_df['season'] = int(year)
                temp_df['team_id'] = home_team_id
                temp_df['team_id_away'] = away_team_id
                temp_df['week'] = int(week)
                positional_data = positional_data.append(temp_df)
            except:
                zeroed_data['player_id'] = player_id
                zeroed_data['season'] = int(year)
                zeroed_data['team_id'] = home_team_id
                zeroed_data['team_id_away'] = away_team_id
                zeroed_data['week'] = int(week)
                positional_data = positional_data.append(zeroed_data)

        positional_data = positional_data.reset_index()
        positional_data['game_id'] = pd.Series(game_ids)
        return positional_data

    @staticmethod
    def aggregate_data(df):
        """This method aggregates the data such that each game observation is a rolling average of that individual's
        performance.

        Args:
            df (pandas.DataFrame): dataframe to aggregate the stats from

        Returns:
            pandas.DataFrame: the dataframe containing the aggregated data

        """
        name_list = list(df.player_id.unique())
        temp_df = df[:]
        for name in name_list:
            player_data = temp_df[temp_df['player_id'] == name][:].sort_values(['season', 'week'])
            non_agg_cols = player_data[['week', 'season', 'team_id', 'opponent_id', 'player_id', 'win_loss']]
            agg_cols = player_data.drop(['week', 'season', 'team_id', 'opponent_id', 'player_id'], 1)
            # tbh don't actually remember why this was necessary
            agg_cols['agg_win_loss'] = agg_cols['win_loss']
            agg_cols.drop('win_loss', 1, inplace=True)
            # this is just a row with zeroes to serve as the initial values of stats for each individual when they have
            # never played a game before
            agg_dummy_row = pd.DataFrame({key: 0 for key in list(agg_cols.columns)}, index=[0])
            agg_cols_with_dummy_data = pd.concat([agg_dummy_row, agg_cols], axis=0, ignore_index=True)
            agg_cols_agg_applied = np.cumsum(agg_cols_with_dummy_data, axis=0).reset_index(drop=True)
            agg_cols_agg_applied_normed = pd.DataFrame(columns=agg_cols_agg_applied.columns)

            for i, row in agg_cols_agg_applied.iterrows():
                if i != 0:
                    agg_cols_agg_applied_normed = agg_cols_agg_applied_normed.append(row / i)

            agg_player_data = pd.concat([agg_cols_agg_applied_normed, non_agg_cols.reset_index(drop=True)], axis=1,
                                        join_axes=[agg_cols_agg_applied_normed.index]).iloc[:-1]

            # creating dataframe for the first qb, otherwise combining it with existing aggregated data
            if name_list.index(name) != 0:
                training_data = pd.concat([training_data, agg_player_data], axis=0, ignore_index=True)
            else:
                training_data = agg_player_data
        return training_data
