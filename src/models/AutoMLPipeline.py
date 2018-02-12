from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier


class autoML():

    def runTpot(self, trainingDf, predictionDf, yCol, week, year, savedFilePath):
        X = trainingDf.drop(yCol, 1)
        y = trainingDf[yCol]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
        tpot = TPOTClassifier(generations=5, verbosity=2, scoring='f1')
        tpot.fit(X_train, y_train)
        print(tpot.score(X_test, y_test))
        tpot.export(savedFilePath)
        X_predict = predictionDf.drop(yCol, 1)
        results = tpot.predict(X_predict)
        print(X_predict)
        print(results)