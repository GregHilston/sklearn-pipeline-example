import logging
import typing

import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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


def main():
    # loads our data into independent and dependent variables
    X, y = sklearn_to_df(load_boston())

    # splits our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)


if __name__ == "__main__":
    main()
