"""Helper functions, not project specific."""

from typing import Any, Final, Optional
import logging

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_empty_values(dataframe: pd.DataFrame) -> None:
    """ Plot a histogram of empty values percentage per columns of the input DataFrame.

    Args:
        dataframe: The DataFrame to plot.

    Returns:
        None.
    """
    num_rows = len(dataframe.index)
    if num_rows == 0:
        logging.warning("No data to plot.")
        return

    # Get the percentage of empty values per column
    columns_emptiness = (
        pd.DataFrame(
            {
                col: {
                    "count": dataframe[col].isna().sum(),
                    "percent": 100 * dataframe[col].isna().sum() / num_rows,
                }
                for col in dataframe.columns
            }
        )
        .transpose()
        .sort_values(by=["count"])
    )

    # Plot the bar chart with Plotly Express
    fig = px.bar(
        columns_emptiness,
        color="percent",
        y="percent",
        labels={
            "index": "column name",
            "percent": "% of empty values",
            "count": "# of empty values",
        },
        hover_data=["count"],
        title="Empty values per column",
        width=1200,
        height=800,
    )
    fig.show()


def plot_categories_bars(
    dataframe: pd.DataFrame,
    plot_columns: Optional[list[str]] = None,
    categorical_column: Optional[str] = None,
) -> None:
    """ Draw one bar chart per categorical or boolean variable, split per class and target.

        Arguments :
        - dataframe : Pandas DataFrame containing the data, including the categorical_columns and target column
        - plot_columns : list of columns to plot, if None, all bool & category columns are plotted
        - categorical_column : string representing the name of the variable containing the categories

        Returns : None
    """
    if plot_columns is None:
        plot_columns = dataframe.select_dtypes(
            include=["bool", "category"],
        ).columns.tolist()

    for col in plot_columns:
        df_g = (
            dataframe.groupby([col, categorical_column], dropna=False)
            .size()
            .reset_index()
        )
        df_g["percentage"] = (
            dataframe.groupby([col, categorical_column], dropna=False)
            .size()
            .groupby(level=0)
            .apply(lambda x: 100 * x / float(x.sum()))
            .values
        )
        df_g.columns = [col, categorical_column, "Count", "Percentage"]
        df_g.sort_values(
            by=["Count", "Percentage"],
            ascending=False,
            inplace=True,
        )
        fig = px.bar(
            df_g,
            x=col,
            y=["Count"],
            color=categorical_column,
            hover_data=["Percentage"],
            title=f"{col} Categories distribution and {categorical_column} ratio",
            width=800,
            height=400,
        )
        fig.show()