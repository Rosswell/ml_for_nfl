# general necessities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cleaning the data
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# linear classifier models
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# nonlinear classification models
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# ensemble classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Grid Search for evaluating multiple parameters of the models while testing them
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# misc
from collections import OrderedDict
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import psycopg2
import datetime
import pickle
import configparser
from src.data import psql_operations


class MLOptimizer:
    def __init__(self, df, y_col, week, year, global_random_state=0):
        config = configparser.ConfigParser()
        config.read('../../config.txt')
        self.db_connection = psql_operations.PostgresConnection()
        self.player_map = self.db_connection.get_db_data('SELECT * FROM \"player_map_v2\"')
        self.team_map = self.db_connection.get_db_data('SELECT * FROM \"team_map_v2\"')
        self.week = week
        self.year = year
        self.X = df.drop(y_col, 1)
        self.y = df[y_col]
        feature_number = len(df.columns) - 1
        # observations = len(df)
        np.random.seed(global_random_state)
        self.data = OrderedDict([('Model', []),
                                 ('Training Score', []),
                                 ('Test Score', []),
                                 ('Fit Time', []),
                                 ('Best Model Params', []),
                                 ('Best Model Instance', [])])
        self.classification_dict = dict(LogisticRegression={
            'name': 'Logistic Regression',
            'model': LogisticRegression(),
            'params': {
                'model__warm_start': [True, False],
                'model__C': [0.01, 0.1, 1, 10, 100]
            }
        }, MLPClassifier={
            'name': 'MLPClassifier',
            'model': MLPClassifier(),
            'params': {
                # 'model__alpha': [0.1, 1, 10],
                # 'model__max_iter': [25, 50, 100, 500]
            }
        }, LDA={
            'name': 'LDA',
            'model': LDA(),
            'params': {
                'model__solver': ['svd', 'lsqr', 'eigen']
            }
        }, GaussianNB={
            'name': 'Gaussian Naive Bayes',
            'model': GaussianNB(),
            'params': {}
        }, MultinomialNB={
            'name': 'Multinomial Naive Bayes',
            'model': MultinomialNB(),
            'params': {}
        }, QDA={
            'name': 'QDA',
            'model': QDA(),
            'params': {}
        }, DecisionTreeClassifier={
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(),
            'params': {
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [2, 5, 10]
            }
        }, SVC={
            'name': 'RBF SVC',
            'model': SVC(),
            'params': {
                'model__C': [1, 10, 100],
                'model__degree': [2, 3, 4, 5],
                'model__gamma': [0.1, 1, 10],
                'model__kernel': ['rbf'],
                'model__probability': [True]
            }
        }, KNeighborsClassifier={
            'name': 'K Neighbors Classifier',
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': list(range(1, feature_number - 1))
            }
        }, RandomForest={
            'name': 'Random Forest Classifier',
            'model': RandomForestClassifier(),
            'params': {
                "model__max_depth": [3, None],
                "model__max_features": list(range(1, feature_number, 3)),
                "model__min_samples_split": [5, 10, 15],
                "model__min_samples_leaf": list(range(1, feature_number, 3)),
                "model__bootstrap": [True, False],
                "model__criterion": ["gini", "entropy"]
            }
        }, GradientBoostingClassifier={
            'name': 'Gradient Boosting Classifier',
            'model': GradientBoostingClassifier(),
            'params': {
                'model__loss': ['deviance', 'exponential'],
                'model__n_estimators': [50, 100, 150],
                'model__min_samples_leaf': [5, 10, 15],
                'model__min_samples_split': [5, 10, 15],
                'model__max_features': [feature_number, feature_number - 1, feature_number - 2],
                'model__subsample': [0.4, 0.5, 0.6, 0.7, 0.8]
            }
        }, AdaBoostClassifier={
            'name': 'Ada Boost Classifier',
            'model': AdaBoostClassifier(),
            'params': {
                'model__n_estimators': list(range(100, 300, 50)),
                'model__learning_rate': list(np.arange(0.7, 1.1, 0.1))
            }
        }, Custom1={
            'name': 'Random Forest Classifier',
            'model': RandomForestClassifier(),
            'params': {
                "model__n_estimators": [100],
                "model__max_features": [0.65],
                "model__min_samples_split": [2],
                "model__min_samples_leaf": [16],
                "model__bootstrap": [True],
                "model__criterion": ["entropy"]
            }
        }, Custom2={
            'name': 'Gradient Boosting Classifier',
            'model': GradientBoostingClassifier(),
            'params': {
                "model__learning_rate": [0.1],
                "model__max_features": [0.3],
                "model__max_depth": [2],
                "model__min_samples_leaf": [13],
                "model__min_samples_split": [10],
                "model__n_estimators": [100],
                "model__subsample": [0.55]
            }
        })
    #     # return data_suite

    def import_model(self, clf, clf_name):
        """
        Imports the model parameters to the classification dictionary
        :param clf: classification model; sklearn instance
        :param clf_name: model name; string
        """

        if 'pca' in clf.best_estimator_.named_steps:
            clf_name = str(clf.best_estimator_.named_steps['pca'].n_components_) + " Component PCA + " + clf_name
        clf_df = pd.DataFrame(clf.cv_results_).sort_values('rank_test_score')
        test_score = clf_df['mean_test_score'].values[0]
        train_score = clf_df['mean_train_score'].values[0]
        fit_time = clf_df['mean_fit_time'].values[0]
        self.data['Model'].append(clf_name)
        self.data['Test Score'].append(test_score)
        self.data['Training Score'].append(train_score)
        self.data['Fit Time'].append(fit_time)
        self.data['Best Model Params'].append(clf.best_estimator_.named_steps['model'].get_params())
        self.data['Best Model Instance'].append(clf.best_estimator_)

    def view(self):
        """
        Converts the data map into a dataframe ranked by test score for visualization
        :return: a dataframe with model data
        """
        return pd.DataFrame(self.data).sort_values('Test Score', ascending=False)

    def clear(self):
        """
        Erases the data in the model data dictionary and prints a successful message
        """
        self.data = OrderedDict([('Model', []),
                                 ('Training Score', []),
                                 ('Test Score', []),
                                 ('Fit Time', []),
                                 ('Best Model Params', []),
                                 ('Best Model Instance', [])])
        print("Ranking Cleared.")

    # TODO: should be in visualizations
    def plot_predictions(self, week_game_data=None, agg_data=None, week_titles=None, week='', year='', eval_preds=False,
                         plot_betting_preds=False, agg_test_data=None):
        """
        Creates and displays a plot of the Classification Models' Test and Training Accuracies, as well as Fit Time.
        """

        # getting data to graph
        def line_wrapper(tl):
            new_tl = []
            text_loop_counter = 0
            for text in tl:
                model_name = text._text
                if len(model_name) > 18 and 'PCA' in model_name:
                    model_name = model_name.replace(' +', '')
                    n = 3
                    strings_wrapped = model_name.split(' ')
                    model_name = ' '.join(strings_wrapped[:n]) + '\n+\n' + ' '.join(strings_wrapped[n:])
                    new_tl.append(text)
                    new_tl[text_loop_counter]._text = model_name
                elif len(model_name):
                    n = 2
                    strings_wrapped = model_name.split(' ')
                    model_name = ' '.join(strings_wrapped[:n]) + '\n' + ' '.join(strings_wrapped[n:])
                    new_tl.append(text)
                    new_tl[text_loop_counter]._text = model_name
                else:
                    new_tl.append(text)
                text_loop_counter += 1
            return new_tl

        if eval_preds:
            plotting_values, plotting_annots, week_titles, model_accuracies = self.evaluate_predictions(week, year)
            accuracy_vals = [str(round(val, 3) * 100) + '%' for val in model_accuracies]
            title_suffix = 'Results'
        else:
            # for plotting values for predictions
            x_train = agg_data.drop('win_loss', 1)
            y_train = agg_data['win_loss']
            x_test = agg_test_data.drop('win_loss', 1)
            y_test = agg_test_data['win_loss']
            # newData
            confidences, predictions = self.predict_week(week_game_data, x_train, y_train, x_test, True)
            # confidences, predictions = self.predictWeek(week_game_data, x_train, y_train)
            plotting_values, plotting_annots = self.combine_games(confidences, predictions)
            accuracy_vals = [round(val, 3) for val in self.view()['Test Score'].values[:5]]
            accuracy_vals = accuracy_vals + [np.mean(accuracy_vals)]
            accuracy_vals = [str(val * 100) + '%' for val in accuracy_vals]
            title_suffix = 'Predictions'

        if plot_betting_preds:
            # for plotting betting $/percentages
            betting_values = pd.DataFrame(columns=plotting_values.columns, index=plotting_values.index)
            for col in plotting_values.columns:
                betting_values[col] = plotting_values[col].apply(
                    lambda x: round(x / plotting_values[col].sum() * 100, 2))

            betting_annots = pd.DataFrame(columns=plotting_annots.columns, index=plotting_annots.index)
            for col in plotting_annots.columns:
                for row in plotting_annots.index:
                    betting_annots.loc[row, col] = plotting_annots.loc[row, col].split('\n')[0] + \
                                                   '\n' + str(betting_values.loc[row, col]) + '%'

            g = sns.heatmap(betting_values, annot=betting_annots, linewidths=1, cmap='RdYlGn', cbar=False, fmt="s")
            tl = g.get_xticklabels()
            g.set_xticklabels(line_wrapper(tl), size=10)
            tly = g.get_yticklabels()
            g.set_yticklabels(tly, rotation=0)
            g.set_title('NFL {} Week {} Betting Percentages'.format(year, week), size=16)

        # creating the graph figure and graphing heatmap subplots on them
        # NOTE: height ratios should be adjusted to scale with the number of teams that play on each day
        # plt.figure(figsize=(15, 15))
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10),
                                          gridspec_kw={'height_ratios': [.3, 3, .3]})

        # heatmap graphing
        plot_title = f.suptitle("NFL Week " + week + " Game " + title_suffix, fontsize=16)
        row_start = 0
        row_end_list = np.cumsum(week_titles['iterations'].values)
        min_value = plotting_values.min().values.min()
        max_value = plotting_values.max().values.max()
        for title, rowEnd, axis in zip(week_titles['date'].values, row_end_list, [ax1, ax2, ax3]):
            sns.heatmap(plotting_values.iloc[row_start:rowEnd],
                        annot=plotting_annots.iloc[row_start:rowEnd],
                        linewidths=1, cmap='RdYlGn', cbar=False, fmt="s",
                        ax=axis, vmin=min_value, vmax=max_value)
            axis.set_title(title, size=12)
            row_start = rowEnd

        # manipulating plot axes labels for better visualization
        for ax in [ax1, ax2, ax3]:
            ax.set_ylabel('')
            tl = ax.get_xticklabels()
            ax.set_xticklabels(line_wrapper(tl), size=10)
            tly = ax.get_yticklabels()
            ax.set_yticklabels(tly, rotation=0)

        # adding model accuracy text above the graph, and adjusting text locations
        plt.tight_layout()
        loop_index = 0
        for xLoc, modelName in zip(np.arange(0.11, 0.996, 0.146), list(plotting_values.columns)):
            if loop_index < 5:
                accuracy_text = 'Model Accuracy'
            else:
                accuracy_text = 'Average Accuracy'
            f.text(xLoc, 0.905, "{}:\n         {}".format(accuracy_text, accuracy_vals[loop_index]), size=10,
                   fontweight='bold')
            loop_index += 1
        f.subplots_adjust(top=0.87)
        plot_title.set_y(0.99)
        plot_title.set_x(0.55)
        plt.show()

    def optimize_model(self, model, pca=False):
        """
        Runs all the linear or nonlinear models, with PCA preprocessing on or off, and writes them to the
        best classification model dictionary
        :param model: classification model to use as sklearn method, string
        :param pca: whether to use pca or not, boolean
        """

        model_name = self.classification_dict[model]['name']
        model_instance = self.classification_dict[model]['model']
        model_params = self.classification_dict[model]['params'].copy()
        pipeline_steps = [('model', model_instance)]

        if pca:
            pca_clf = PCA()
            pipeline_steps.insert(0, ('pca', pca_clf))
            model_params['pca__n_components'] = [3, 4, 5, 6]
        pipe = Pipeline(steps=pipeline_steps)
        if 'MLP' in model_name:
            grid = GridSearchCV(pipe, model_params, verbose=1)
            scaled_x = StandardScaler().fit_transform(self.X)
            grid.fit(scaled_x, self.y)
        else:
            grid = GridSearchCV(pipe, model_params, verbose=1)  # , scoring='f1')
            grid.fit(self.X, self.y)
        self.import_model(grid, model_name)

    @staticmethod
    def get_ids(key, c, conn, new_player_id=None, team=True):

        # NOTE: should just be able to pass all teams/players and return a list of ids instead of this malarkey
        if team:
            team_str = key.lower()
            query = "SELECT team_id FROM team_map_v2 WHERE team='" + team_str + "'"
            # if away:
            #     team_str = '@ ' + team_str
            #     query = "SELECT team_id2 FROM team_map_v2 WHERE team='" + team_str + "'"
            c.execute(query)
            team_id = c.fetchall()[0][0]
            return team_id

        player_str = key.lower()
        query = "SELECT player_id FROM player_map_v2 WHERE player=\'{}\'".format(player_str)
        c.execute(query)
        player_id = c.fetchall()
        if not player_id:
            create_entry_query = "INSERT INTO player_map_v2 (player, player_id) " \
                                 "values(\'{}\', \'{}\')".format(player_str, new_player_id)
            c.execute(create_entry_query)
            conn.commit()
            c.execute("SELECT MAX(player_id) FROM player_map_v2")
            new_player_id = c.fetchall()[0][0]
            return new_player_id
        else:
            return player_id[0][0]

    # TODO: is this method even necessary? 
    def fetch_games(self, week, year):

        # this dict is for the teams and the QBs playing
        return_data_dict = {
            'away_team_id': [],
            'home_team_id': [],
            'home_qb_id': [],
            'away_qb_id': [],
            'home_rb_id': [],
            'away_rb_id': [],
            # 'homeRb2Id': [],
            # 'awayRb2Id': [],
            'season': [],
            'week': []
        }

        # this dict is for the date titles for future plot visualization
        return_title_dict = {
            'date': [],
            'iterations': []
        }
        
        results = self.db_connection.get_db_data("SELECT date, hometeam, awayteam, homeqb, awayqb, homerb, awayrb " \
                     "FROM \"2017_matchups\" " \
                     "WHERE week = \'{}\'".format(week))
        results = results.sort_values(['date', 'hometeam'])
        # eliminating teams on a bye

        home_teams = results['hometeam'].tolist()
        away_teams = results['awayteam'].tolist()
        home_qbs = results['homeqb'].tolist()
        away_qbs = results['awayqb'].tolist()
        home_rb1s = results['homerb'].tolist()
        away_rb1s = results['awayrb'].tolist()
        # homeRb2s = results['homerb2'].tolist()
        # awayRb2s = results['awayrb2'].tolist()
        times = results['date'].tolist()
        results['date'] = results.date.apply(lambda x: self.convert_datetime_to_titles(x))
        dates = results['date'].unique()
        team_count_on_date = [results['date'].value_counts()[value] for value in results['date'].unique()]

        # grabs the last index to use it for inserting new players, but would be easier if the table used an auto-inc column
        cursor.execute("SELECT MAX(player_id) FROM player_map_v2")
        new_player_id = cursor.fetchall()[0][0]
        for gameIndex, awayQb, homeQb, awayRb, homeRb in zip(range(len(away_teams)), away_qbs, home_qbs, away_rb1s,
                                                             home_rb1s):  # , awayRb2s, homeRb2s):
            # getting the ids of players and teams
            away_team_id = self.get_ids(away_teams[gameIndex], cursor, conn, new_player_id, team=True)
            home_team_id = self.get_ids(home_teams[gameIndex], cursor, conn, new_player_id, team=True)
            away_qb_id = self.get_ids(awayQb, cursor, conn, new_player_id, team=False)
            new_player_id += 1
            home_qb_id = self.get_ids(homeQb, cursor, conn, new_player_id, team=False)
            new_player_id += 1
            away_rb_id = self.get_ids(awayRb, cursor, conn, new_player_id, team=False)
            new_player_id += 1
            home_rb_id = self.get_ids(homeRb, cursor, conn, new_player_id, team=False)
            new_player_id += 1
            # awayRb2Id = self.getIds(awayRb2, cursor, conn, new_player_id, team=False, away=False)
            # new_player_id += 1
            # homeRb2Id = self.getIds(homeRb2, cursor, conn, new_player_id, team=False, away=False)
            # new_player_id += 1

            # filling the map with game data
            return_data_dict['away_team_id'].append(away_team_id)
            return_data_dict['home_team_id'].append(home_team_id)
            return_data_dict['home_qb_id'].append(home_qb_id)
            return_data_dict['away_qb_id'].append(away_qb_id)
            return_data_dict['home_rb_id'].append(home_rb_id)
            return_data_dict['away_rb_id'].append(away_rb_id)
            # return_data_dict['homeRb2Id'].append(homeRb2Id)
            # return_data_dict['awayRb2Id'].append(awayRb2Id)
            return_data_dict['season'].append(year)
            return_data_dict['week'].append(week)

        # filling the title map with dates and number of teams playing on the date
        return_title_dict['date'] = dates
        return_title_dict['iterations'] = team_count_on_date
        return_data_dict['timestamps'] = times

        return [pd.DataFrame(return_data_dict, index=list(range(len(return_data_dict['away_team_id'])))),
                pd.DataFrame(return_title_dict, index=list(range(len(return_title_dict['date']))))]

    @staticmethod
    def convert_datetime_to_titles(x):
        weekday = x.strftime("%A")
        month = x.strftime('%B')
        day = str(x.day)
        suffix = 'th'
        if day[0] != '1':
            if day[-1] == '1':
                suffix = 'st'
                if day[-2] == '1':
                    suffix = 'th'
            elif day[-1] == '2':
                suffix = 'nd'
                if day[-2] == '1':
                    suffix = 'th'
            elif day[-1] == '3':
                suffix = 'rd'
                if day[-2] == '1':
                    suffix = 'th'
        return weekday + ', ' + month + ' ' + day + suffix

    # TODO: should be in predict.py
    def evaluate_predictions(self, week, year):

        prediction_table = self.db_connection.get_db_data("SELECT * FROM \"{}\" "
                                                          "WHERE week = \'{}\'".format(year + '_predictions', week))
        prediction_table.sort_values('date', inplace=True)
        game = [homeTeam + ' vs. ' + awayTeam for homeTeam, awayTeam in
                zip(prediction_table['hometeam'], prediction_table['awayteam'])]
        prediction_table.index = game
        prediction_table.drop(['hometeam', 'awayteam'], 1, inplace=True)
        prediction_table['date'] = prediction_table.date.apply(lambda x: self.convert_datetime_to_titles(x))

        predictions = prediction_table.iloc[:, :6]
        results = prediction_table['results']

        def value_transform(val):
            if val == 'TIE':
                return 0.5
            elif val == 'BYE':
                return np.nan
            return 0

        plotting_vals = pd.DataFrame(index=[predictions.index])

        dates = list(prediction_table.date.unique())
        date_map = prediction_table.date.value_counts().to_dict()
        team_count_per_date = [date_map[date] for date in dates]
        model_accuracies = []
        for col in predictions.columns:
            plotting_vals[col] = np.where(predictions[col] == results, 1,
                                          predictions[col].apply(lambda x: value_transform(x)))
            model_accuracies.append(plotting_vals[col].sum() / len(plotting_vals[col]))

        weekly_plot_titles = pd.DataFrame({'date': dates, 'iterations': team_count_per_date})

        # sorting the tables by date
        plotting_vals['date'] = list(prediction_table['date'])
        plotting_vals = plotting_vals.sort_values('date', ascending=False)
        plotting_vals.drop('date', 1, inplace=True)
        predictions['date'] = list(prediction_table['date'])
        predictions = predictions.sort_values('date', ascending=False)
        predictions.drop('date', 1, inplace=True)
        weekly_plot_titles = weekly_plot_titles.sort_values('date', ascending=False)

        return [plotting_vals, predictions, weekly_plot_titles, model_accuracies]

    # TODO: should be in predict.py
    def predict_week(self, fetched_game_data, training_df_x, training_df_y, test_df_x, new_data=False):
        """
        Uses the top 5 prediction models to create dataframes of win/loss predictions and confidence scores
        :param test_df_x:
        :param new_data:
        :param fetched_game_data:
        :param training_df_x: aggregated training data for input variables; dataframe
        :param training_df_y: aggregated training data for output variable; dataframe
        :return:
        """

        if new_data:
            home_total_team_ids = list(
                np.append(fetched_game_data['homeTeamId'].values, fetched_game_data['awayTeamId'].values))

        else:
            # beginning to include RB data
            temp_df1 = pd.merge(fetched_game_data, self.player_map, how='inner', left_on='homeQbId', right_on='player_id')
            temp_df1.rename(index=str, columns={'player_name': 'homeQbName'}, inplace=True)
            temp_df2 = pd.merge(fetched_game_data, self.player_map, how='inner', left_on='homeRbId', right_on='player_id')
            temp_df2.rename(index=str, columns={'player_name': 'homeRbName'}, inplace=True)

            total_player_ids = list(
                np.append(fetched_game_data['homeQbId'].values, fetched_game_data['awayQbId'].values))
            home_total_team_ids = list(
                np.append(fetched_game_data['homeTeamId'].values, fetched_game_data['awayTeamId'].values))
            away_total_team_ids = list(
                np.append(fetched_game_data['awayTeamId'].values, fetched_game_data['homeTeamId'].values))
            prediction_data = pd.DataFrame(columns=training_df_x.columns)

            # Aggregate data sorting for the QBs starting on the week
            for playerId in total_player_ids:
                if len(training_df_x['player_id'][training_df_x['player_id'] == int(playerId)]) != 0:
                    # note: this would be where the rb data would be merged to the qb data
                    prediction_data = prediction_data.append(
                        training_df_x[training_df_x['player_id'] == int(playerId)].sort_values(['season', 'week']).iloc[-1:])
                else:
                    prediction_data = prediction_data.append(
                        pd.DataFrame(np.zeros((1, len(training_df_x.columns))), columns=training_df_x.columns))

            # resetting the index and replacing the incorrect team ids with the right ones - the current team ids
            # are those from the raw DB, so they're whatever team the player was on at that point, rather than their current team
            prediction_data.index = list(range(0, len(prediction_data)))
            prediction_data['team_id'] = home_total_team_ids
            prediction_data['opponent_id'] = away_total_team_ids

            # this is for 2qb predictions
            prediction_data2 = prediction_data.copy()
            prediction_data3 = pd.merge(prediction_data, prediction_data2, left_on='team_id', right_on='opponent_id',
                                        suffixes=('_home', '_away'))

        models = self.view()['Best Model Instance'][:5]
        model_names = self.view()['Model'][:5]
        self.save_models(models, model_names)

        # NOTE: should just have a map stored so we don't have to make all these queries all the time
        # gets the team abbreviations for the graph
        team_print_list = []
        for team_id in home_total_team_ids:
            team = self.team_map[self.team_map['team_id'] == team_id].values[0]
            if len(team) > 1:
                team = team[1][0]
            else:
                team = team[0][0]
            if len(team.split(' ')) > 1:
                team_print_list.append(team.split(' ')[1].upper())
            else:
                team_print_list.append(team.upper())

        # predicting the games and their associated confidence scores
        confidences_df = pd.DataFrame(index=team_print_list)
        predictions_df = pd.DataFrame(index=team_print_list)
        for model, modelName in zip(models, model_names):
            if isinstance(model, list):
                pipe = Pipeline(model[0], model[1])
                pipe.fit(training_df_x, training_df_y)
                # predictions = pipe.predict(prediction_data) # old data
                # confidences = list(pipe.predict_proba(prediction_data))
                predictions = pipe.predict(test_df_x)  # old data
                confidences = list(pipe.predict_proba(test_df_x))
            else:
                clf = model
                clf.fit(training_df_x, training_df_y)
                # predictions = clf.predict(prediction_data) old data
                # confidences = list(clf.predict_proba(prediction_data))
                predictions = clf.predict(test_df_x)
                confidences = list(clf.predict_proba(test_df_x))

            confidence_of_prediction = []
            for pred, conf in zip(predictions, confidences):
                if pred == 0:
                    confidence_of_prediction.append(conf[0])
                else:
                    confidence_of_prediction.append(conf[1])

            confidences_df = pd.concat([confidences_df,
                                       pd.Series(confidence_of_prediction, index=team_print_list, name=modelName).rename(
                                           modelName)], axis=1)
            predictions_df = pd.concat([predictions_df, pd.Series(predictions, index=team_print_list, name=modelName)],
                                       axis=1)
        predictions_df['Aggregate Prediction'] = predictions_df.mean(axis=1)
        confidences_df['Aggregate Prediction'] = confidences_df.mean(axis=1)

        # awayQbs, homeQbs, predictionDict, confidenceDict = self.parseDfsForDb(fetched_game_data, predictions_df, confidences_df)
        # self.saveToDb(self.year + '_predictions',
        #               fetched_game_data['timestamps'],
        #               team_print_list[0:len(team_print_list)//2],
        #               team_print_list[len(team_print_list)//2:],
        #               [self.week] * len(awayQbs),
        #               homeQbs,
        #               awayQbs,
        #               None,
        #               None,
        #               None,
        #               None,
        #               confidenceDict,
        #               predictionDict)
        return confidences_df, predictions_df

    @staticmethod
    def combine_games(confidence_df, prediction_df):
        """
        Takes the individual predictions of teams, takes the team with the highest confidence call,
        :param confidence_df:
        :param prediction_df:
        :return:
        """
        # creating the combined maps
        confidence_map = OrderedDict([(key, []) for key in confidence_df.columns])
        plotting_labels_map = OrderedDict([(key, []) for key in confidence_df.columns])
        # creating the combined teams index
        index = list(confidence_df.index)
        home_away_split = len(index) // 2
        combined_team_index = [homeTeam + ' vs. ' + awayTeam for homeTeam, awayTeam in
                               zip(index[:home_away_split], index[home_away_split:])]
        cols = list(confidence_df.columns)
        # choosing the highest confidence call, and creating two new dataframes, one with the highest confidence 
        # prediction and one with the highest confidence number
        for colIndex in range(len(confidence_df.columns)):
            for homeIndexStart, awayIndexStart in zip(range(0, home_away_split),
                                                      range(home_away_split, home_away_split * 2 + 1)):
                # collecting the predictions and confidences for a game
                home_confidence = confidence_df.iloc[homeIndexStart: homeIndexStart + 1, colIndex].values[0]
                away_confidence = confidence_df.iloc[awayIndexStart: awayIndexStart + 1, colIndex].values[0]
                home_prediction = prediction_df.iloc[homeIndexStart: homeIndexStart + 1, colIndex].values[0]
                away_prediction = prediction_df.iloc[awayIndexStart: awayIndexStart + 1, colIndex].values[0]

                # finding the more prediction with higher confidence
                max_conf = max(home_confidence, away_confidence)
                min_conf = min(home_confidence, away_confidence)
                # getting the total Confidence in the game call by assessing whether the predictions agree or not
                if home_prediction == away_prediction:
                    if home_confidence != 0 and away_confidence != 0:
                        total_confidence = (max_conf + (1 - min_conf)) / 2
                    else:
                        total_confidence = max_conf
                else:
                    if home_confidence != 0 and away_confidence != 0:
                        total_confidence = (max_conf + min_conf) / 2
                    else:
                        total_confidence = max_conf

                confidence_map[cols[colIndex]].append(total_confidence)
                # writes the home team to the label map if it has the higher confidence, otherwise write the away team
                if max_conf == home_confidence:
                    plotting_labels_map[cols[colIndex]].append(index[homeIndexStart] +
                                                               '\nConfidence: ' + "%.3f" % total_confidence)
                else:
                    plotting_labels_map[cols[colIndex]].append(index[awayIndexStart] +
                                                               '\nConfidence: ' + "%.3f" % total_confidence)

        # self.saveToDb(confidence_map, plotting_labels_map, week, year)
        return [pd.DataFrame(confidence_map, index=combined_team_index),
                pd.DataFrame(plotting_labels_map, index=combined_team_index)]

    def save_to_db(self, table_name, timestamps, home_team, away_team, week, away_qbs, home_qbs,
                   away_rb1s, home_rb1s, away_rb2s, home_rb2s,
                   confidence_dict=None, prediction_dict=None):

        write_choice = 'y'
        empty_check = self.db_connection.get_db_data("SELECT * FROM \"{}\" WHERE week = \'{}\'".format(table_name, week[0]))
        if empty_check:
            write_choice = input(
                'This week already exists in the {} table. Continue writing to DB? (y/n)\n'.format(table_name)).lower()

        # inserting data into the DB
        if write_choice == 'y':
            sql_queries = []
            for i in range(len(timestamps)):
                # fix this: time adjustment is not necessary when getting results from the DB rather than the internet
                if 1 == 2:
                    time = datetime.datetime.strptime(timestamps[i], '%Y-%m-%dT%H:%MZ') - datetime.timedelta(hours=7)
                else:
                    time = timestamps[i]
                wk = int(week[i])
                if 'matchups' in table_name:
                    sql_queries.append("INSERT INTO \"2017_matchups\" (date, week, hometeam, awayteam, homeqb, awayqb, homerb, awayrb, homerb2, awayrb2) "
                                       "VALUES({}, {}, {}, {}, {}, {}, {}, {}, {}, {});".format(
                                        time, wk, home_team[i], away_team[i], home_qbs[i].lower(), away_qbs[i].lower(), 
                                        home_rb1s[i].lower(), away_rb1s[i].lower(), home_rb2s[i].lower(), away_rb2s[i].lower()))
                # elif 'odds' in table_name:
                    # needs filling out - need to look into odds API first
                    # sql_queries.append("INSERT INTO \"2017_odds\" (date, week, hometeam, awayteam, favorite, underdog, favoriteteam) "
                    #                    "VALUES({}, {}, {}, {}, {}, {}, {});".format(time, wk, home_team[i], away_team[i]))
                elif 'predictions' in table_name:
                    # NOTE: doesn't need the Rbs, but may not even need the Qbs either, look into this
                    sql_queries.append("INSERT INTO \"2017_predictions\" "
                                       "(model1, model2, model3, model4, model5, agg, date, model1_conf, model2_conf, "
                                       "model3_conf, model4_conf, model5_conf, agg_conf, homeqb, awayqb, week, "
                                       "hometeam, awayteam) "
                                       "VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});".format(
                                        prediction_dict['model1'][i],
                                        prediction_dict['model2'][i],
                                        prediction_dict['model3'][i],
                                        prediction_dict['model4'][i],
                                        prediction_dict['model5'][i],
                                        prediction_dict['agg'][i],
                                        time,
                                        confidence_dict['model1_conf'][i],
                                        confidence_dict['model2_conf'][i],
                                        confidence_dict['model3_conf'][i],
                                        confidence_dict['model4_conf'][i],
                                        confidence_dict['model5_conf'][i],
                                        confidence_dict['agg_conf'][i],
                                        home_qbs[i].lower(),
                                        away_qbs[i].lower(),
                                        wk,
                                        home_team[i],
                                        away_team[i]))

                self.db_connection.set_db_data(sql_queries)

    def parse_dfs_for_db(self, game_data, prediction_df, confidence_df):

        prediction_map, confidence_map = self.combine_games(confidence_df, prediction_df)

        away_qbs = []
        home_qbs = []
        for awayQbId, homeQbId in zip(game_data['awayQbId'].values, game_data['homeQbId'].values):
            away_qbs.append(self.player_map['player'][self.player_map['player_id'] == awayQbId].values[0])
            home_qbs.append(self.player_map['player'][self.player_map['player_id'] == homeQbId].values[0])
        prediction_dict = {}
        confidence_dict = {}
        for modelCol, modelNum in zip(confidence_map.columns[:-1], list(range(1, 6))):
            prediction_label = 'model' + str(modelNum)
            confidence_label = 'model' + str(modelNum) + '_conf'
            prediction_dict[prediction_label] = []
            confidence_dict[confidence_label] = []
            for val in confidence_map[modelCol].values:
                pred, conf = val.split('\nConfidence: ')
                prediction_dict[prediction_label].append(pred)
                confidence_dict[confidence_label].append(conf)
        prediction_dict['agg'] = []
        confidence_dict['agg_conf'] = []
        for agg in confidence_map['Aggregate Prediction'].values:
            pred, conf = agg.split('\nConfidence: ')
            prediction_dict['agg'].append(pred)
            confidence_dict['agg_conf'].append(float(conf))

        return [away_qbs, home_qbs, prediction_dict, confidence_dict]

    def get_odds(self, week, year, total_bet):
        """
        Retrieves the predictions and determines winnings from the results in order to plot winnings from a week
        :param week: week to query; string
        :param year: year to query; string
        :param total_bet: total amount to bet on that week's games; int
        :return: none
        """

        # function to return the return from a scaled bet
        def betting_results(df, row, col, total_bet):

            scaled_bet = df.loc[row, col + '_conf'] * total_bet
            if df.loc[row, col] == df.loc[row, 'results']:  # winning case
                if df.loc[row, col] == df.loc[row, 'favoriteteam']:
                    return scaled_bet / (df.loc[row, 'favorite'] / 100) + scaled_bet  # winning + favorite
                else:
                    return scaled_bet * (df.loc[row, 'underdog'] / 100) + scaled_bet  # winning + underdog
            return 0  # (-scaled_bet)

        odds_df = self.db_connection("SELECT * FROM \"{}\" "
                                     "WHERE week = \'{}\'".format(year + '_odds', week))

        prediction_df = self.db_connection("SELECT * FROM \"{}\" "
                                           "WHERE week = \'{}\'".format(year + '_predictions', week))

        game = [homeTeam + ' vs. ' + awayTeam for homeTeam, awayTeam in zip(odds_df['hometeam'], odds_df['awayteam'])]
        odds_df.index = game
        prediction_df.index = game

        joined_df = odds_df.join(prediction_df, rsuffix='del')
        joined_df.drop(['datedel', 'weekdel', 'hometeamdel', 'awayteamdel'], 1, inplace=True)

        model_cols = ['model1', 'model2', 'model3', 'model4', 'model5', 'agg']
        for col in model_cols:
            conf_col = col + '_conf'
            joined_df[conf_col] = joined_df[conf_col] / joined_df[conf_col].sum()
        odds_map = {k: [] for k in model_cols}
        for col in model_cols:
            for row in joined_df.index:
                odds_map[col].append(round(betting_results(joined_df, row, col, total_bet), 2))

        odds_df = pd.DataFrame(odds_map, index=[joined_df.index])
        odds_annots = pd.DataFrame(columns=odds_df.columns, index=joined_df.index)

        sums = pd.Series(name='Total Return')
        sum_annots = pd.Series(name='Total Return')
        percent_gain = pd.Series(name='Percentage Return')
        percent_gain_annots = pd.Series(name='Percentage Return')
        for col in odds_df:
            total = odds_df[col].sum()
            gain = (total / total_bet - 1) * 100
            # adding the totals and return percentages to the repsective annotations and values series
            sums.set_value(col, total)
            percent_gain.set_value(col, gain)
            sum_annots.set_value(col, '$' + str(total))
            percent_gain_annots.set_value(col, str(gain) + '%')
            # transforming the annotation dataframe
            odds_annots[col] = odds_df[col].apply(lambda x: '$' + str(x))

        # adding the series to the respective dataframes for plotting
        for vals, annots in zip([sums, percent_gain], [sum_annots, percent_gain_annots]):
            odds_df = odds_df.append(vals)
            odds_annots = odds_annots.append(annots)

        f, axes = plt.subplots(3, 2, sharex='col', figsize=(10, 10), gridspec_kw={'height_ratios': [3, .4, .4],
                                                                                  'width_ratios': [30, 1],
                                                                                  'wspace': 0.1})
        sns.heatmap(odds_df.iloc[:-2], linewidths=1, cmap='RdYlGn',
                    annot=odds_annots.iloc[:-2], fmt='s', ax=axes[0][0], cbar_ax=axes[0][1])
        sns.heatmap(odds_df.iloc[-2].values.reshape(1, -1), linewidths=1, cmap='RdYlGn',
                    annot=odds_annots.iloc[-2].values.reshape(1, -1), fmt='s', ax=axes[1][0], cbar_ax=axes[1][1],
                    yticklabels=['Net Total'])
        sns.heatmap(odds_df.iloc[-1].values.reshape(1, -1), linewidths=1, cmap='RdYlGn',
                    annot=odds_annots.iloc[-1].values.reshape(1, -1), fmt='s', ax=axes[2][0], cbar_ax=axes[2][1],
                    xticklabels=odds_df.columns, yticklabels=['Percentage Return'])
        for ax in [axes[0][0], axes[1][0], axes[2][0]]:
            ytl = ax.get_yticklabels()
            ax.set_yticklabels(ytl, rotation=0)
        f.suptitle('NFL {} Week {} Betting Returns with ${}'.format(year, week, total_bet), fontsize=16)
        plt.show()

    @staticmethod
    def save_models(models, model_names):
        for model, name in zip(models, model_names):
            filename = (name + '.sav').replace(' ', '_')
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
