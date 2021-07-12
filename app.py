import logging
import typing

import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    # We separate out our feature engineering Pipeline from our model Pipeline because we can fit the feature engineering Pipeline now
    # and then tag on different Model steps later, reusing the original feature_pipe object.
    feature_pipe = Pipeline([
        ("scaler", StandardScaler(numeric_columns)),
    ])

    model_pipe = Pipeline([
        ("decision tree regressor", DecisionTreeRegressor(random_state=random_state))
    ])

    pipe = Pipeline([
        feature_pipe,
        model_pipe
    ])

    # When we call `.fit()` on the Pipeline, we call `.fit()` on EVERY step in the pipeline, sequentially.
    # When we call `.predict()`, on the Pipeline, we call `.transform` on EVERY step in the pipeline, sequentially, except the last one if the last step extends a Model.
    # If you have another model, in the middle, this will break the pipeline unless we modify the Sklearn Pipeline code.

    # This fits the `pipe` object, in place, allowing me to pickle/dill the `pipe` object for inference.
    # trains our model
    pipe.fit(X_train, y_train)

    # evaluates our model
    logger.info(f"regressor R^2 score {pipe.score(X_test, y_test)}")


if __name__ == "__main__":
    main(random_state=random_state)
