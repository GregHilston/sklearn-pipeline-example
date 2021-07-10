import logging
import typing

import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig()
logger = logging.getLogger(__name__)

random_state = 42


def sklearn_to_df(sklearn_dataset: sklearn.utils.Bunch) -> typing.Tuple[pd.DataFrame, pd.Series]:
    """Converts an sklearn premade dataset into a Pandas DataFrame

    based on: https://stackoverflow.com/a/46379878/1983957
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    target = pd.Series(sklearn_dataset.target)
    return df, target


def main(random_state):
    # loads our data into independent and dependent variables
    X, y = sklearn_to_df(load_boston())

    # splits our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # separates our columns by numeric and categorical
    categorical_columns = ["CHAS"]
    numeric_columns = [column for column in list(X) if column not in categorical_columns]

    # scales our data by only fitting on our training but scaling both training and testing
    # this avoids data leakage from our test set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train[X_train[numeric_columns]])
    X_test = scaler.transform(X_test)

    # trains our model
    reg = LinearRegression().fit(X, y)
    print(reg)


if __name__ == "__main__":
    main(random_state=random_state)
