from typing import Union, Optional

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


def scale_variables(dataframe: pd.DataFrame,) -> pd.DataFrame:
    """ Scale specified variables in a dataframe using scikit-learn's StandardScaler

        Returns a copy of the dataframe with scaled variables

        Parameters
        ----------
        dataframe: pd.DataFrame
            Dataframe to scale

        Returns
        -------
        pd.DataFrame
            Copy of the dataframe with scaled variables

        Raises
        ------
        ValueError
            If the dataframe contains non-numeric variables
    """
    # check if dataframe contains only numeric variables
    if not dataframe.select_dtypes(include=["number"]).columns.all():
        raise ValueError("Dataframe must contain only numeric variables")

    # scale dataframe
    scaler = StandardScaler()
    scaler.fit(dataframe)
    scaled_data = scaler.transform(dataframe)
    scaled_df = pd.DataFrame(
        scaled_data, columns=dataframe.columns, index=dataframe.index
    )
    return scaled_df


def impute_missing_values(dataframe: pd.DataFrame,) -> pd.DataFrame:
    """ Impute missing values in specified DataFrame with sci-kit learn's IterativeImputer

        :param dataframe: The dataframe to be imputed.

        :return: The imputed dataframe.

        Example:
        impute_missing_values(dataframe)
    """
    # define imputer
    imputer = IterativeImputer(n_nearest_features=20, verbose=2)
    # fit on the dataset
    imputer.fit(dataframe)
    # transform the dataset
    imputed_data = imputer.transform(dataframe.values)

    imputed_df = pd.DataFrame(
        imputed_data, columns=dataframe.columns, index=dataframe.index,
    )

    return imputed_df


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
