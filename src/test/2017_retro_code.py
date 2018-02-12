from src.data import make_dataset
from src.data import psql_operations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import ZeroCount


weeks = [str(x) for x in range(1, 17)]
year = '2017'
# loading df
db_connection = psql_operations.PostgresConnection()
rbDf = db_connection.get_db_data('SELECT * FROM rbs_v2')
qbDf = db_connection.get_db_data('SELECT * FROM qbs_v2')
# setting columns that duplicate QB columns in the RB data to zero to allow for data cleaning
for col in ['score_for', 'score_against', 'win_loss']:
    rbDf[col] = 0
cleaner = make_dataset.DataCleaner()
print('Initially cleaning and aggregating the RB data')
aggRbDf = cleaner.clean_data(rbDf, 'rbs_v2')

print('Initially cleaning and aggregating the QB data')
aggQbDf = cleaner.clean_data(qbDf, 'qbs_v2')

print('getting maps and stuff from the DB')
winnings_list = []
vegas_winnings_list = []
accuracy_list = []
vegas_accuracy_list = []
for week in weeks:
    print(f'calculating week {week}...')

    trainRb_Qb = pd.merge(aggQbDf, aggRbDf, on='game_id', suffixes=('_qb', '_rb')).sort_values('total_touches_rb')
    trainRb_Qb = trainRb_Qb.drop_duplicates(subset='game_id')
    testing_data = cleaner.get_testing_data(aggRbDf, aggQbDf, week, year)
    trainRb_Qb = trainRb_Qb.drop('game_id', 1)

    training_features = trainRb_Qb.drop('win_loss', axis=1).values
    training_target = trainRb_Qb['win_loss'].values
    prediction_features = testing_data.drop('win_loss', axis=1).values

    # Score on the training set was:0.6203702298990865
    pipeline = make_pipeline(
        ZeroCount(),
        RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.05, min_samples_leaf=12,
                               min_samples_split=11, n_estimators=100)
    )

    pipeline.fit(training_features, training_target)
    # 1 as a prediction means the home team won
    predictions = pipeline.predict(prediction_features)
    confidences = pipeline.predict_proba(prediction_features)

    print('determining the predicted winning teams')
    id_to_short_team = {team_id: team for team_id, team in zip(cleaner.team_map['team_id'], cleaner.team_map['team'])}

    halfway_index = int(len(predictions) / 2)
    # predictions
    round_1_predictions = []
    round_2_predictions = []
    current_index = 0
    for home_team_id, away_team_id, prediction in zip(list(testing_data['team_id_qb']),
                                                      list(testing_data['team_id_away_qb']),
                                                      predictions):
        home = id_to_short_team[home_team_id].upper()
        away = id_to_short_team[away_team_id].upper()
        # avoiding 2 loops, so switch rounds lives here:
        if current_index < halfway_index:
            # round 1
            if prediction == 1:
                round_1_predictions.append(home)
            else:
                round_1_predictions.append(away)
        else:
            if prediction == 1:
                round_2_predictions.append(home)
            else:
                round_2_predictions.append(away)
        current_index += 1

    print('assigning winning teams from the average of confidences')
    winning_team_predictions = []
    winning_team_confidences = []
    for round1_conf, round2_conf, round1, round2 in zip(confidences[:halfway_index], confidences[halfway_index:],
                                                        round_1_predictions, round_2_predictions):
        round1_conf = max(round1_conf)
        round2_conf = max(round2_conf)
        # both rounds of prediction agree, take the average of the confidences
        if round1 == round2:
            winning_team_predictions.append(round1)
            winning_team_confidences.append(np.average([round1_conf, round2_conf]))
        # rounds of prediction don't agree, therefore take the difference of the confidences
        else:
            if round1_conf > round2_conf:
                winning_team_predictions.append(round1)
                winning_team_confidences.append(round1_conf - round2_conf)
            else:
                winning_team_predictions.append(round2)
                winning_team_confidences.append(round2_conf - round1_conf)

    print('getting the results from the results table in the DB')
    results_df = db_connection.get_db_data(f'SELECT winner FROM results WHERE week={week}')
    team_to_abbr = {
        'raiders': 'oak',
        'patriots': 'ne',
        'bills': 'buf',
        'texans': 'hou',
        'saints': 'no',
        'panthers': 'car',
        'bengals': 'cin',
        'browns': 'cle',
        'packers': 'gb',
        'lions': 'det',
        'seahawks': 'sea',
        'jaguars': 'jax',
        'jets': 'nyj',
        'chiefs': 'kc',
        'broncos': 'den',
        'dolphins': 'mia',
        'buccaneers': 'tb',
        'vikings': 'min',
        'cardinals': 'ari',
        'giants': 'nyg',
        'titans': 'ten',
        'steelers': 'pit',
        'colts': 'ind',
        'ravens': 'bal',
        'cowboys': 'dal',
        'chargers': 'lac',
        'rams': 'lar',
        '49ers': 'sf',
        'bears': 'chi',
        'redskins': 'wsh',
        'eagles': 'phi',
        'falcons': 'atl'
    }

    print('assessing accuracy of the predictions')
    results = [team_to_abbr[long_team].upper() for long_team in list(results_df['winner'].values)]
    correct_predictions = []
    for team in winning_team_predictions:
        correct = True
        if team not in results: correct = False
        correct_predictions.append(correct)

    percent_money_allocation = [x / sum(winning_team_confidences) for x in winning_team_confidences]
    vegas_percent_money_allocation = [1 / len(winning_team_confidences)] * len(winning_team_confidences)

    odds_df = db_connection.get_db_data('SELECT * FROM \"2017_odds\" WHERE week=' + week)

    correct_vegas_predictions = []
    for team in list(odds_df['favoriteteam'].values):
        correct = True
        if team not in results: correct = False
        correct_vegas_predictions.append(correct)

    # calculating model winnings
    total_winnings = 0
    for correct, stake, predicted_team in zip(correct_predictions, percent_money_allocation, winning_team_predictions):
        fav_team = ''
        fav_odds = ''
        und_odds = ''
        winnings = 0
        stake *= 1000
        for i, row in odds_df.iterrows():
            if predicted_team in list(row):
                fav_team = row['favoriteteam']
                fav_odds = row['favorite']
                und_odds = row['underdog']

        if correct and predicted_team == fav_team:
            winnings = stake / (fav_odds / 100) + stake
        elif correct and predicted_team != fav_team:
            winnings = stake * (und_odds / 100) + stake
        else:
            winnings = 0

        total_winnings += winnings

    # calculating vegas winnings
    total_vegas_winnings = 0
    for correct, stake, predicted_team in zip(correct_vegas_predictions, vegas_percent_money_allocation,
                                              list(odds_df.loc[:, 'favoriteteam'])):
        fav_team = ''
        fav_odds = ''
        und_odds = ''
        winnings = 0
        stake *= 1000
        for i, row in odds_df.iterrows():
            if predicted_team in list(row):
                fav_team = row['favoriteteam']
                fav_odds = row['favorite']
                und_odds = row['underdog']

        if correct and predicted_team == fav_team:
            winnings = stake / (fav_odds / 100) + stake
        elif correct and predicted_team != fav_team:
            winnings = stake * (und_odds / 100) + stake
        else:
            winnings = 0

        total_vegas_winnings += winnings

    total_prediction_accuracy = np.average(correct_predictions)
    total_vegas_prediction_accuracy = np.average(correct_vegas_predictions)

    winnings_list.append(total_winnings)
    vegas_winnings_list.append(total_vegas_winnings)
    accuracy_list.append(total_prediction_accuracy)
    vegas_accuracy_list.append(total_vegas_prediction_accuracy)