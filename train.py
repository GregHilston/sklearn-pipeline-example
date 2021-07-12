import logging

import pickle
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline

from my_sklearn.my_sklearn import sklearn_to_df


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

random_state = 42


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
    feature_pipe = make_pipeline(
        StandardScaler(numeric_columns),
        # Foo()
        # can add more
    )
    feature_pipe.fit(X_train, y_train)
    train_pipe = feature_pipe.transform(X_train)

    # could uncomment below
    # and use train_pipe along with test_pipe to perform custom
    # cross validation. This works, because feature_pipe was fitted ONLY on trained data
    # and test_pipe will ensure our test data is processed using only things learned
    # by training data
    # test_pipe = feature_pipe.transform(X_test)

    # This fits the `pipe` object, in place, allowing me to pickle/dill the `pipe` object for inference.
    # trains our model
    regressor = DecisionTreeRegressor(random_state=random_state)
    regressor.fit(train_pipe, y_train)

    pipe = make_pipeline(
        feature_pipe,
        regressor
    )

    # When we call `.fit()` on the Pipeline, we call `.fit()` on EVERY step in the pipeline, sequentially.
    # When we call `.predict()`, on the Pipeline, we call `.transform` on EVERY step in the pipeline, sequentially, except the last one if the last step extends a Model.
    # If you have another model, in the middle, this will break the pipeline unless we modify the Sklearn Pipeline code.

    # write out our fitted pipe to disk
    pickle.dump(pipe, open("pipe.pk", "wb"))


if __name__ == "__main__":
    main(random_state=random_state)
