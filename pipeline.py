import dill
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from catboost import CatBoostClassifier


def create_features(df):
    df = df.copy()

    df['is_zero_loans'] = np.where(
        (df['is_zero_loans5'] == 0) & (df['is_zero_loans530'] == 0) & (df['is_zero_loans3060'] == 0) &
        (df['is_zero_loans6090'] == 0) & (df['is_zero_loans90'] == 0), 0, 1)
    df['is_zero_overlimit'] = np.where((df['is_zero_over2limit'] == 0) & (df['is_zero_maxover2limit'] == 0), 0, 1)

    return df

def main():
    print('Credit Scoring Model Pipeline')

    data = pd.read_csv('data/train_data.csv')
    X, y = data.drop(columns=['id', 'flag']), data['flag']

    model = CatBoostClassifier(eval_metric='AUC',
                               custom_metric='AUC:hints=skip_train~false',
                               verbose=100,
                               grow_policy='Lossguide',
                               bagging_temperature=10,
                               boosting_type='Plain',
                               depth=10)

    pipe = Pipeline(steps=[
        ('feature_creator', FunctionTransformer(create_features)),
        ('classifier', model)
    ])
    pipe.fit(X, y)

    with open('credit_scoring_pipe.pkl', 'wb') as file:
        dill.dump({'model': pipe}, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


