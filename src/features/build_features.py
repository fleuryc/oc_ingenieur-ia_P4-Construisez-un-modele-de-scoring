from typing import Union

import pandas as pd


def drop_impossible_values(
    dataframe: pd.DataFrame,
    constraints=dict[str, dict[str, Union[int, float]]],
) -> pd.DataFrame:
    """
    Drop values from a dataframe that have impossible or unlikely values.
    :param dataframe: The dataframe to be filtered.
    :param constraints: A dictionary of constraints to be applied.
    :return: The filtered dataframe.

    Example:
    constraints = {
        'age': {
            'min': 18,
            'max': 60
        }
    }

    """
    for col in dataframe.columns:
        if col in constraints:
            dataframe = dataframe[
                dataframe[col].between(
                    constraints[col]["min"], constraints[col]["max"]
                )
            ]
    return dataframe


def drop_outliers(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """ Remove outlier values from specified columns of DataFrame

        Compute the Inter-Quartile Ranges and set outliers to NaN
    """
    # compute quartiles and define range
    quartiles = dataframe[columns].quantile([0.25, 0.75])
    iqr = quartiles.loc[0.75] - quartiles.loc[0.25]
    limits = pd.DataFrame(
        {
            col: [
                quartiles.loc[0.25, col] - 1.5 * iqr[col],  # min
                quartiles.loc[0.75, col] + 1.5 * iqr[col],  # max
            ]
            for col in columns
        },
        index=["min", "max"],
    )

    df = dataframe.copy()

    # set to NaN data that are outside the range
    for col in columns:
        df.loc[:, col] = (
            dataframe[col]
            .where(limits.loc["min", col] <= dataframe[col])
            .where(dataframe[col] <= limits.loc["max", col])
        )

    return df
