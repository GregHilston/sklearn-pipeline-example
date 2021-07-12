import logging

import pickle
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from my_sklearn.my_sklearn import sklearn_to_df


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

random_state = 42


def main(random_state: int):
    # loads our data into independent and dependent variables
    X, y = sklearn_to_df(load_boston())

    # splits our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # read in our previously fitted pipe from disk
    pipe = pickle.load(open("pipe.pk", "rb"))

    # performs inference
    predictions = pipe.predict(X_test[0:1])
    logger.info(f"predictions {predictions}")

    # evaluates our model
    score = pipe.score(X_test, y_test)
    logger.info(f"regressor R^2 score {score}")


if __name__ == "__main__":
    main(random_state=random_state)