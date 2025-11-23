import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.mixture import GaussianMixture
    import marimo as mo
    return OrdinalEncoder, StandardScaler, make_pipeline, mo, pd


@app.cell
def _(pd):
    df = pd.read_csv("../Tech_Use_Stress_Wellness.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Data Preprocessing
    """)
    return


@app.cell
def _(OrdinalEncoder, StandardScaler, df, make_pipeline):
    numerical_pipeline = make_pipeline(StandardScaler())
    categorical_pipeline = make_pipeline(OrdinalEncoder())
    categorical_columns = df.select_dtypes(include=["object"])
    numerical_columns = df.select_dtypes(include=["float64"])
    return categorical_columns, numerical_columns


@app.cell
def _(categorical_columns):
    categorical_columns
    return


@app.cell
def _(numerical_columns):
    numerical_columns
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
