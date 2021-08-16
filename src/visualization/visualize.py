import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


def plot_boxes(
    dataframe: pd.DataFrame,
    categorical_column: str,
    order_values: tuple[str] = None,
    num_cols: int = 3,
) -> None:
    """ Draw one boxplot per numerical variable, split per categories.

        Arguments :
        - dataframe : Pandas DataFrame containing the data, including the categorical_column and numerical_columns
        - categorical_column : string representing the name of the variable containing the categories
        - numerical_columns : list of strings representing the name of the numerical variables to plot
        - order_values : list of strings representing the values of the numerical variables to plot

        Returns : None
    """
    numerical_columns = dataframe.select_dtypes(
        include="number"
    ).columns.tolist()

    num_lines = int(np.ceil(len(numerical_columns) / num_cols))
    fig, axes = plt.subplots(
        num_lines, num_cols, figsize=(8 * num_cols, 8 * num_lines)
    )
    fig.suptitle(
        f"Numeric variables distribution, per { categorical_column }",
        fontsize=24,
    )

    for i, col in enumerate(numerical_columns):
        sns.boxplot(
            data=dataframe,
            x=categorical_column,
            y=col,
            order=order_values,
            showmeans=True,
            ax=axes[int(np.floor(i / num_cols)), i % num_cols],
        )


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
        height=600,
    )
    fig.show()


# Let's define a function to plot multiple BoxPlots
def plot_categories_bars(dataframe: pd.DataFrame) -> None:
    """ Draw one bar chart per categorical or boolean variable, split per class and target.

        Arguments :
        - dataframe : Pandas DataFrame containing the data, including the categorical_columns and target column

        Returns : None
    """
    for col in dataframe.select_dtypes(["bool", "category"]).columns:
        df_g = dataframe.groupby([col, "TARGET"]).size().reset_index()
        df_g["percentage"] = (
            dataframe.groupby([col, "TARGET"])
            .size()
            .groupby(level=0)
            .apply(lambda x: 100 * x / float(x.sum()))
            .values
        )
        df_g.columns = [col, "TARGET", "Count", "Percentage"]
        df_g.sort_values(
            by=["Count", "Percentage"], ascending=False, inplace=True
        )
        fig = px.bar(
            df_g,
            x=col,
            y=["Count"],
            color="TARGET",
            hover_data=["Percentage"],
            title="Categories distribution and target ration",
            width=1200,
            height=600,
        )
        fig.show()
