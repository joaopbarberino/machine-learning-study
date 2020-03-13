import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


#  Read a csv and transform it into a pandas.DataFrame
df = pd.read_csv('http://bit.ly/kaggletrain')

#  Setting dataframe, removing rows where 'Embarked' is missing
df = df.loc[df.Embarked.notna(), ['Survived', 'Pclass', 'Sex', 'Embarked']]

#  Define the X of the model, our features, deleting the 'Survived' column, which is our output
X = df.drop('Survived', axis='columns')
print(X)

#  Define the Y of the model, which is the label we want to predict
y = df.Survived
print(y)

#  Make a column transformer to pre-process our data, selecting 'Sex' and 'Embarked' columns to
#  encode them with OneHotEncoder, the 'remainder' parameter decides what to do with the remaining
#  columns, as the default behavior is to drop them, we use 'passthrough' to just concatenate them
#  with the processed data, resulting:
#  For each possible value of a column, it creates a column with 0 or 1 - false or true:
#  ['sex_value_1', 'sex_value_2', 'embarked_value_1', 'embarked_value_2', 'embarked_value_3', 'untouched_Pclass']
#  [[0. 1. 0. 0. 1. 3.]
#   [1. 0. 1. 0. 0. 1.]
#   [1. 0. 0. 0. 1. 3.]
#   ...
#   [1. 0. 0. 0. 1. 3.]
#   [0. 1. 1. 0. 0. 1.]
#   [0. 1. 0. 1. 0. 3.]]

column_trans = make_column_transformer(
    (OneHotEncoder(), ['Sex', 'Embarked']),
    remainder='passthrough')

column_trans.fit_transform(X)

#  Build a Pipeline with a Logistic Regression model and our pre-processor
logreg = LogisticRegression(solver='lbfgs')
pipe = make_pipeline(column_trans, logreg)

#  Cross validate model with X and y, returning the average accuracy of the prediction
score = cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()
print(score)

#  Works as model.fit, but runs pre-processing as well
pipe.fit(X, y)

X_new = X.sample(5, random_state=99)

#  Works as model.predict, but runs pre-processing on the inserted data before prediction
predicts = pipe.predict(X_new)
print(predicts)
