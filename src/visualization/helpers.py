"""Helper functions, not project specific."""

from typing import Optional
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.base import is_classifier, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


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
            y=col,
            points=False,
            color=categorical_column,
            title=f"{col} distribution per TARGET",
            width=800,
            height=400,
        )
        fig.update_traces(boxmean="sd")
        fig.update_traces(notched=True)
        fig.show()


def plot_classifier_results(
    classifier: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Plots the confusion_matrix, precision_recall_curve and roc_curve of a classifier on test data.

    Args:
        classifier (ClassifierMixin): sklearn Classifier
        X_test (pd.DataFrame): test data
        y_test (pd.Series): true values
    """
    if not is_classifier(classifier):
        logging.error(f"{classifier} is not a classifier.")
        raise ValueError(f"{classifier} is not a classifier.")


    _, ax = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(24, 8),
    )

    plot_confusion_matrix(
        classifier, X_test, y_test,
        ax=ax[0],
    )

    plot_precision_recall_curve(
        classifier,
        X_test,
        y_test,
        name=classifier.__class__.__name__,
        ax=ax[1],
    )

    plot_roc_curve(
        classifier,
        X_test,
        y_test,
        name=classifier.__class__.__name__,
        ax=ax[2],
    )

    plt.show()


def plot_pca_2d(
    data: pd.DataFrame,
    categories: pd.DataFrame,
) -> None:
    """ Draw a 2D PCA plot of the data and the feture importances.

        Arguments :
        - data : Pandas DataFrame containing the data
        - categories : Pandas DataFrame containing the categories

        Returns : None
    """
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Plot the data in the PCA space
    fig = px.scatter(
        x=data_pca[:, 0],
        y=data_pca[:, 1],
        color=categories.values.flatten(),
        symbol=categories.values.flatten(),
        title="PCA 2D",
        opacity=.2,
        width=1200,
        height=800,
    )

    # Plot the feature importances in the PCA space
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feature in enumerate(data.columns):
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=loadings[i, 0],
            y1=loadings[i, 1],
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
        )

    fig.show()
