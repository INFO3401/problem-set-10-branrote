################################################################################
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

################################################################################
# Worked with Michael and Aaron
# also used stack overflow
class AnalysisData:

    def _init_(self, filename):
        self.variables = []
        self.filename = filename

    def parseFile(self):
        self.dataset = pd.read_csv(self.filename)
        self.variables = self.dataset.colums

dataParser = AnalysisData('./candy-data.csv')
dataParser.parseFile()

################################################################################

class LinearAnalysis(object):

    def _init_(self, targetY):
        self.bestX = None
        self.targetY = targetY
        self.fit = None

    def runSimpleAnalysis(self, dataParser):

        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables:
            if column == self.targetY or column == 'competitorname':
                continue

            x_values = dataset[column].values.reshape(-1,1)
            y_values = dataset[self.targetY].values

            regr = LinearRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values)
            score = r2_score(y_values,preds)

            if score > best_pred:
                best_pred = score
                self.bestX = column

        self.fit = best_pred
        print(self.bestX)
        print(self.fit)

linear_analysis = LinearAnalysis(targetY='sugarpercent')
linear_analysis.runSimpleAnalysis(dataParser)
#from here i was going to add the regression and score

################################################################################
class LogisticAnalysis(object):

    def _init_(self, targetY):
        self.bestX = None
        self.targetY = targetY
        self.fit = None

    def runSimpleAnalysis(self, dataParser):

        dataset = dataParser.dataset

        best_pred = 0
        for column in dataParser.variables:
            if column == self.targetY or column == 'competitorname':
                continue

            x_values = dataset[column].values.reshape(-1,1)
            y_values = dataset[self.targetY].values

            regr = LogisticRegression()
            regr.fit(x_values, y_values)
            preds = regr.predict(x_values)
            score = r2_score(y_values,preds)

            if score > best_pred:
                best_pred = score
                self.bestX = column

        self.fit = best_pred
        print(self.bestX)
        print(self.fit)
        print(regr.coef_)
        print(regr.intercept_)

    def runMultipleRegression(self, dataParser):

        dataset = dataParser.dataset
        clean_dataset = dataset.drop([self.targetY, 'competitorname'], axis=1)
        x_values = clean_dataset.values
        y_values = dataset[self.targetY].values

        regr = LogisticRegression()
        regr.fit(x_values, y_values)
        preds = regr.predict(x_values)
        score = r2_score(y_values,preds)

        print(clean_dataset.columns)
        print(score)
        print(regr.coef_)
        print(regr.intercept_)


logistic_analysis = LogisticAnalysis(targetY='chocolate')
logistic_analysis.runSimpleAnalysis(dataParser)

multivariable_logistics = = LogisticAnalysis(targetY='chocolate')
multivariable_logistics,runMultipleRegression(dataParser)
###############################################################################
#Problem 3

# Linear Regression: y = .0044 + .2571

# Logistic Regression: 1/1 + e^ -(.059x - -3.088)

# Multiple Regression: -2.5286x1 -0.197x2 0.0394x3 -0.1654x4 0.4978x5 -0.4759x6
# 0.8151x7 -0.5997x8 -0.2581x9 0.3224 0.05388x11 -1.6826x12
###############################################################################
#Problem 4

# a.) Null = Caramel and Chocolate have similar sugar content.
#   independent = caramel (categorical)
#   independent = chocolare (categorical)
#   dependent = sugar (continuous)

# b.) Null = There is an equal amount in red and blue states.
#   independent = red states (discrete)
#   independent = blue states (discrete)
#   dependent = split ticket voters (discrete)

# c.)
#   independent = phones with short battery lives (categorical)
#   independent = phones with longer battery lives (categorical)
#   dependent = phone rate (continuous)
