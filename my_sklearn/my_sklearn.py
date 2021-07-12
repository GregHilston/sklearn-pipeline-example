import typing

import pandas as pd
import sklearn


def sklearn_to_df(sklearn_dataset: sklearn.utils.Bunch) -> typing.Tuple[pd.DataFrame, pd.Series]:
    """Converts an sklearn premade dataset into a Pandas DataFrame

    based on: https://stackoverflow.com/a/46379878/1983957
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    target = pd.Series(sklearn_dataset.target)
    return df, target