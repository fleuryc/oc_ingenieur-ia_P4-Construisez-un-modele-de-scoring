from typing import Optional


import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_boxes(
    dataframe: pd.DataFrame,
    plot_columns: Optional[list[str]] = None,
    categorical_column: Optional[str] = None,
) -> None:
    """ Draw one boxplot per numerical variable, split per categories.

        Arguments :
        - dataframe : Pandas DataFrame containing the data, including the categorical_column and numerical_columns
        - plot_columns : list of columns to plot, if None, all numerical columns are plotted
        - categorical_column : string representing the name of the variable containing the categories

        Returns : None
    """
    if plot_columns is None:
        plot_columns = dataframe.select_dtypes(
            include="number"
        ).columns.tolist()

    for i, col in enumerate(plot_columns):
        fig = px.box(
            dataframe,
            x=categorical_column,
            y=col,
            color=categorical_column,
            title="Variable distribution per TARGET",
            width=800,
            height=400,
        )
        fig.update_traces(boxmean="sd")
        fig.update_traces(notched=True)
        fig.show()


def plot_empty_values(dataframe: pd.DataFrame) -> None:
    """ Plot a histogram of empty values percentage per columns of the input DataFrame
    """
    num_rows = len(dataframe.index)
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


# Let's define a function to plot multiple BoxPlots
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
        df_g = dataframe.groupby([col, categorical_column]).size().reset_index()
        df_g["percentage"] = (
            dataframe.groupby([col, categorical_column])
            .size()
            .groupby(level=0)
            .apply(lambda x: 100 * x / float(x.sum()))
            .values
        )
        df_g.columns = [col, categorical_column, "Count", "Percentage"]
        df_g.sort_values(
            by=["Count", "Percentage"], ascending=False, inplace=True
        )
        fig = px.bar(
            df_g,
            x=col,
            y=["Count"],
            color=categorical_column,
            hover_data=["Percentage"],
            title="Categories distribution and TARGET ratio",
            width=800,
            height=400,
        )
        fig.show()
