import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#  Read a csv and transform it into a pandas.DataFrame
df = pd.read_csv('/Users/barberino/Downloads/extrato.csv')

#  Setting dataframe, removing unwanted columns
df = df.reindex(columns=[
    'Data',
    'Tag',
    'Valor',
    'Label',
])

df.Data = pd.to_datetime(df.Data, format="%d/%m/%Y")

# for index, x in enumerate(df.Data):
#     df.Data[index] = x.day

df.Data = df.Data.dt.day
print(df)

#  Define the X of the model, our features, deleting the 'Survived' column, which is our output
X = df.drop('Label', axis='columns')

#  Define the Y of the model, which is the label we want to predict
y = df.Label
# print(X, '\n\n', y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0002)

#  Make a column transformer to pre-process our data, selecting 'Sex' and 'Embarked' columns to
#  encode them with OneHotEncoder, the 'remainder' parameter decides what to do with the remaining
#  columns, as the default behavior is to drop them, we use 'passthrough' to just concatenate them
#  with the processed data, resulting:
#  For each possible value of a column, it creates a column with 0 or 1 - false or true.

column_trans = make_column_transformer((OneHotEncoder(), ['Tag', 'Valor']), remainder='passthrough')
# print(column_trans)
# print(column_trans.fit_transform(X))

#  Build a Pipeline with a Logistic Regression model and our pre-processor
logreg = LogisticRegression(solver='lbfgs', max_iter=50000)
pipe = make_pipeline(column_trans, logreg)

#  Works as model.fit, but runs pre-processing as well
pipe.fit(X, y)

#  Cross validate model with X and y, returning the average accuracy of the prediction
# score = cross_val_score(pipe, X_test, y_test, cv=5, scoring='accuracy').mean()
# print(score)

#  Works as model.predict, but runs pre-processing on the inserted data before prediction
print('\n\n')

df2 = pd.read_csv('/Users/barberino/Downloads/extrato_reposta.csv')
df2 = df2.reindex(columns=[
    'Data',
    'Tag',
    'Valor',
])
df2.Data = pd.to_datetime(df2.Data, format="%d/%m/%Y")
df2.Data = df2.Data.dt.day
print(df2, '\n\n\n')

predicts = pipe.predict(df2)
print(predicts)
