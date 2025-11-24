import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import altair as alt
    import marimo as mo
    import numpy as np
    return (
        FunctionTransformer,
        OneHotEncoder,
        PCA,
        StandardScaler,
        alt,
        make_column_transformer,
        make_pipeline,
        mo,
        np,
        pd,
    )


@app.cell
def _(pd):
    df = pd.read_csv("../Tech_Use_Stress_Wellness.csv")
    df.replace({False: 0, True: 1}, inplace=True)
    return (df,)


@app.cell
def _(df):
    len(df.columns)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Preprocessing

    - Normalize numerical features to all have $\mu = 0$ and $\sigma = 1$
    - Apply one hot encoding to categorical data
    - Map boolean values to 0 or 1 respectively
    """)
    return


@app.cell
def _(OneHotEncoder, StandardScaler, df, make_pipeline, np):
    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OneHotEncoder())
    categorical_columns = df.select_dtypes(include=["object"]).columns.to_list()
    numerical_columns = (
        df.select_dtypes(include=np.number)
        .drop(columns=["user_id"])
        .columns.to_list()
    )
    return (
        categorical_columns,
        categorical_pipeline,
        numerical_columns,
        numerical_pipeline,
    )


@app.cell(hide_code=True)
def _(FunctionTransformer, np):
    # Right skewed transformation functions
    log_transformer = FunctionTransformer(np.log, feature_names_out="one-to-one")
    sqrt_transformer = FunctionTransformer(np.sqrt, feature_names_out="one-to-one")
    inverse_sqrt_transformer = FunctionTransformer(
        lambda n: 1 / np.sqrt(n), feature_names_out="one-to-one"
    )

    # Left skewed transformation functions
    square_transformer = FunctionTransformer()
    return


@app.cell
def _(
    categorical_columns,
    categorical_pipeline,
    df,
    make_column_transformer,
    numerical_columns,
    numerical_pipeline,
    pd,
):
    pre_processor = make_column_transformer(
        (numerical_pipeline, numerical_columns),
        (categorical_pipeline, categorical_columns),
        remainder="drop",
    )

    preprocessed_train_df = pd.DataFrame(
        data=pre_processor.fit_transform(df),
        columns=pre_processor.get_feature_names_out(),
        index=df.index,
    )
    return (preprocessed_train_df,)


@app.cell
def _(preprocessed_train_df):
    preprocessed_train_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### PCA
    """)
    return


@app.cell(hide_code=True)
def _(PCA, alt, np, pd):
    def plot_PCA_components(df: pd.DataFrame) -> alt.Chart:
        """df: Preprocessed dataframe
        Generates a plot to choose the appropriate number of components to pass to PCA
        """
        pca_model = PCA(n_components=20)
        pca_model.fit_transform(df)
        cum_sum_ratio = pca_model.explained_variance_ratio_.cumsum()
        # Create a DataFrame
        data = pd.DataFrame(
            {
                "Number of Components": np.arange(
                    1, len(cum_sum_ratio) + 1
                ),
                "Cumulative Explained Variance": cum_sum_ratio,
            }
        )

        # Find the first component count that explains >= 90% of the variance
        # This is a common threshold, but the user can visually adjust.
        # Use try-except for robustness if the array doesn't reach 90%
        try:
            threshold_90 = data[data["Cumulative Explained Variance"] >= 0.9].iloc[
                0
            ]
            component_90 = int(threshold_90["Number of Components"])
        except IndexError:
            # If 90% is never reached, use the last component for the rule placement
            component_90 = len(cum_sum_ratio)
        # Create the base chart
        chart = (
            alt.Chart(data)
            .encode(
                x=alt.X(
                    "Number of Components:O",
                    axis=alt.Axis(title="Number of Principal Components"),
                ),
                y=alt.Y(
                    "Cumulative Explained Variance:Q",
                    # axis=alt.Axis(
                    #     format=".0%",
                    #     title="Cumulative Explained Variance",
                   
                    # ),
                    scale=alt.Scale(domain=[0, 1])
                ),
                tooltip=[
                    "Number of Components",
                    alt.Tooltip("Cumulative Explained Variance", format=".2%"),
                ],
            )
            .properties(title="PCA Explained Variance vs. Number of Components")
        )

        # Add the line
        line = chart.mark_line(point=True).encode(color=alt.value("darkblue"))

        # Add a horizontal reference line for 90% variance (a common heuristic)
        rule_90_y = (
            alt.Chart(pd.DataFrame({"y": [0.9]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y")
        )

        # Add a vertical reference line for the component count that reaches 90%
        rule_90_x = (
            alt.Chart(pd.DataFrame({"x": [component_90]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x=alt.X("x:O", axis=None))
        )

        # Combine all layers
        final_chart = (line + rule_90_y + rule_90_x).interactive()

        return final_chart
    return (plot_PCA_components,)


@app.cell
def _(plot_PCA_components, preprocessed_train_df):
    plot_PCA_components(preprocessed_train_df)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The plot tells us that 15 components keeps 91.5% of variance
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
