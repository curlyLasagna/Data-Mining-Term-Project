import marimo

__generated_with = "0.18.1"
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
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import hamming_loss
    import altair as alt
    import marimo as mo
    import numpy as np
    return (
        FunctionTransformer,
        GaussianMixture,
        LogisticRegression,
        MultiLabelBinarizer,
        MultiOutputClassifier,
        OneHotEncoder,
        PCA,
        StandardScaler,
        alt,
        hamming_loss,
        make_column_transformer,
        make_pipeline,
        mo,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("Tech_Use_Stress_Wellness.csv")
    df.replace({False: 0, True: 1}, inplace=True)
    return (df,)


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

    preprocessed_df = pd.DataFrame(
        data=pre_processor.fit_transform(df),
        columns=pre_processor.get_feature_names_out(),
        index=df.index,
    )
    return (preprocessed_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ### PCA
    """)
    return


@app.cell
def _(PCA, alt, np, pd):
    def plot_PCA_components(df: pd.DataFrame, n: int) -> alt.Chart:
        """df: Preprocessed dataframe
        Generates a plot to choose the appropriate number of components to pass to PCA
        """
        pca_model = PCA(n)
        pca_model.fit_transform(df)
        cum_sum_ratio = pca_model.explained_variance_ratio_.cumsum()
        # Create a DataFrame
        data = pd.DataFrame(
            {
                "Number of Components": np.arange(1, len(cum_sum_ratio) + 1),
                "Cumulative Explained Variance": cum_sum_ratio,
            }
        )

        # Find the first component count that explains >= 90% of the variance
        threshold_90 = data[data["Cumulative Explained Variance"] >= 0.9].iloc[0]
        component_90 = int(threshold_90["Number of Components"])
        print(component_90)
        # Create the base chart
        chart = (
            alt.Chart(data)
            .encode(
                x=alt.X(
                    "Number of Components:Q",
                    axis=alt.Axis(title="Principal Components"),
                ),
                y=alt.Y(
                    "Cumulative Explained Variance:Q",
                    axis=alt.Axis(
                        format=".0%",
                        title="Cumulative Explained Variance",
                    ),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    "Number of Components",
                    alt.Tooltip("Cumulative Explained Variance", format=".2%"),
                ],
            )
            .properties(title="Principal Components")
        )

        # Add the line
        line = chart.mark_line(point=True).encode(color=alt.value("darkblue"))

        rule_90_y = (
            alt.Chart(pd.DataFrame({"y": [0.9]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y")
        )

        rule_90_x = (
            alt.Chart(pd.DataFrame({"x": [component_90]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x=alt.X("x:Q"))
        )

        # Combine all layers
        final_chart = (line + rule_90_y + rule_90_x).interactive()

        return final_chart
    return (plot_PCA_components,)


@app.cell
def _(plot_PCA_components, preprocessed_df):
    plot_PCA_components(preprocessed_df, 20).save(fp="cum_var.png", scale_factor=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The plot tells us that 15 components keeps 91.5% of variance of the data.
    90 to 95% is the recommended value since our goal is to accurately predict
    """)
    return


@app.cell
def _(PCA, preprocessed_df):
    pca_model = PCA(n_components=15)
    pca_arr = pca_model.fit_transform(preprocessed_df)
    return (pca_arr,)


@app.cell
def _(GaussianMixture, np, pca_arr):
    def getGMM_components(X) -> list:
        N = np.arange(1, 15)

        models = [None for n in N]
        for i in range(len(N)):
            models[i] = GaussianMixture(
                n_components=N[i], max_iter=1000, covariance_type="full"
            )
            models[i].fit(X)
        return models


    models = getGMM_components(pca_arr)

    AIC = [m.aic(pca_arr) for m in models]
    BIC = [m.bic(pca_arr) for m in models]

    # index of GMM with the best BIC score
    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    print("best fit converged:", gmm_best.converged_)
    print("BIC: n_components =  %i" % np.arange(1, 15)[i_best])
    return (gmm_best,)


@app.cell
def _(alt, pd):
    def line_plot(df: pd.DataFrame, title: str) -> alt.Chart:
        return (
            alt.Chart(df.reset_index())
            .mark_line()
            .encode(x="index:Q", y="value:Q")
        )
    return


@app.cell
def _(alt, pd):
    def plot_clusters(df: pd.DataFrame):
        alt.Chart(df).mark_circle().encode(alt.X(""))
    return


@app.cell
def _(gmm_best, pca_arr):
    gmm_best.predict_proba(pca_arr)
    return


@app.cell
def _(df, gmm_best, pd):
    def get_GMM_prob_df(data):
        return pd.DataFrame(
            gmm_best.predict_proba(data),
            columns=[f"P_cluster{i + 1}" for i in range(gmm_best.n_components)],
            index=df.index,
        )
    return (get_GMM_prob_df,)


@app.cell
def _(alt, np, pd):
    def plot_probability_distributions(df_probs: pd.DataFrame) -> alt.VConcatChart:
        """
        Analyzes the probability matrix from GMM and generates two histograms in Altair:
        1. Distribution of the Maximum Probability (P_max)
        2. Distribution of the Second Highest Probability (P_2nd_Max)

        Args:
            df_probs (pd.DataFrame): DataFrame containing only the GMM probability
                                     columns (P_Cluster_0 to P_Cluster_12).

        Returns:
            alt.VConcatChart: A vertically concatenated Altair chart showing both histograms.
        """
        NUM_CLUSTERS = 9
        CURRENT_THRESHOLD = 0.40  # The threshold the user is currently considering
        # 1. Calculate P_max and P_2nd_Max for each user

        # Sort probabilities descending for each row
        sorted_probs = np.sort(df_probs.values, axis=1)[:, ::-1]

        # Extract the highest probability (P_max) and the second highest (P_2nd_Max)
        analysis_df = pd.DataFrame(
            {"P_max": sorted_probs[:, 0], "P_2nd_Max": sorted_probs[:, 1]}
        )

        # 2. Define the base chart visualization elements
        base = alt.Chart(analysis_df).properties(width=400, height=250)

        # Common elements for the threshold line
        threshold_line = (
            alt.Chart(pd.DataFrame({"threshold": [CURRENT_THRESHOLD]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(
                x="threshold",
                tooltip=[
                    alt.Tooltip(
                        "threshold", format=".2f", title="Current Threshold"
                    )
                ],
            )
        )

        # --- Chart 1: P_max Distribution ---
        chart_p_max = base.encode(
            x=alt.X("P_max", bin=alt.Bin(maxbins=30), title="Probability (P_max)"),
            y=alt.Y("count()", title="Number of Users"),
            tooltip=[
                alt.Tooltip("P_max", bin=True, title="P_max Range"),
                "count()",
            ],
        ).mark_bar().properties(
            title="Distribution of Maximum Cluster Probability"
        ) + threshold_line.encode(x="threshold")  # Add threshold line to P_max

        # --- Chart 2: P_2nd_Max Distribution ---
        chart_p_2nd_max = base.encode(
            x=alt.X(
                "P_2nd_Max",
                bin=alt.Bin(maxbins=30),
                title="Probability (P_2nd Max)",
            ),
            y=alt.Y("count()", title="Number of Users"),
            tooltip=[
                alt.Tooltip("P_2nd_Max", bin=True, title="P_2nd Max Range"),
                "count()",
            ],
        ).mark_bar(color="darkorange").properties(
            title="Distribution of Second Highest Cluster Probability"
        ) + threshold_line.encode(x="threshold")  # Add threshold line to P_2nd_Max

        # 3. Concatenate and return both charts
        return (
            (chart_p_max & chart_p_2nd_max)
            .properties(title="Probability Distribution Analysis")
            .interactive()
        )
    return (plot_probability_distributions,)


@app.cell
def _(df, gmm_best, pca_arr, pd, plot_probability_distributions):
    plot_probability_distributions(
        pd.DataFrame(
            gmm_best.predict_proba(pca_arr),
            columns=[f"P_cluster{i + 1}" for i in range(gmm_best.n_components)],
            index=df.index,
        )
    ).save(fp="prob_dist.png", scale_factor=2)
    return


@app.cell
def _(pd):
    from typing import Dict, List, Tuple

    # --- Constants (Defined globally for scope, but can stay inside the function) ---
    CLUSTER_NAMES: Dict[int, str] = {
        0: "Cluster 1: Inactive & Socially Stressed",
        1: "Cluster 2: Sedentary & Low Mental Health",
        2: "Cluster 3: High-Stress Digital Life",
        3: "Cluster 4: Well-Adjusted",
        4: "Cluster 5: Low-Work Gaming Addict",
        5: "Cluster 6: Mental Health Crisis",
        6: "Cluster 7: Active & Low-Screen",
        7: "Cluster 8: Work/Study Screen Stress",
        8: "Cluster 9: Night Owl Gamer",
        9: "Cluster 10: Highly Content & Disconnected",
        10: "Cluster 11: Mindful but Sleep-Deprived",
    }
    PROBABILITY_THRESHOLD: float = 0.40


    # --- Intermediate Function: Returns Dataframe with Integer Labels ---
    def generate_multi_label_integers(prob_df: pd.DataFrame) -> pd.Series:
        """
        Intermediate function that assigns cluster *integer* labels (0 to N-1)
        to users based on a probability threshold.

        Args:
            prob_df (pd.DataFrame): DataFrame of GMM cluster probabilities.

        Returns:
            pd.Series: A Series of lists, where each list contains the assigned
                       cluster *indices* (integers).
        """

        # Check if the columns match the expected format (P_cluster1, P_cluster2, etc.)
        # This assumes column indices (0, 1, ...) map to cluster indices.

        def assign_integer_labels_by_threshold(row: pd.Series) -> List[int]:
            """Iterates through a single user's probabilities and assigns integer indices."""
            assigned_labels = []
            for col_idx, prob in enumerate(row):
                if prob >= PROBABILITY_THRESHOLD:
                    # The column index is the cluster index (0, 1, 2, ...)
                    assigned_labels.append(col_idx)
            return assigned_labels

        # Apply the function row-wise and return the resulting Series of integer lists
        multi_label_integers = prob_df.apply(
            assign_integer_labels_by_threshold, axis=1
        )
        multi_label_integers.name = "multi_cluster_indices"
        return multi_label_integers


    # --- Original (Now Modified) Final Function: Maps Integers to Names ---
    def generate_multi_label_series(prob_df: pd.DataFrame) -> pd.Series:
        """
        Transforms a DataFrame of GMM cluster probabilities into a Pandas Series
        where each row contains a list of descriptive cluster names that exceed a
        40% probability threshold, by calling an intermediate integer-labeling function.

        Args:
            prob_df (pd.DataFrame): DataFrame of cluster probabilities.

        Returns:
            pd.Series: A Series of lists, where each list contains the descriptive
                       assigned cluster names.
        """

        # 1. Call the intermediate function to get the integer labels
        integer_labels_series = generate_multi_label_integers(prob_df)

        # 2. Map the integer labels to the descriptive names
        def map_integers_to_names(integer_list: List[int]) -> List[str]:
            """Maps a list of integer indices to the corresponding descriptive names."""
            return [CLUSTER_NAMES.get(i, f"Cluster {i + 1}") for i in integer_list]

        # 3. Apply the mapping function to the series of integer lists
        multi_label_series = integer_labels_series.apply(map_integers_to_names)
        multi_label_series.name = "multi_cluster_labels"

        return multi_label_series
    return generate_multi_label_integers, generate_multi_label_series


@app.cell
def _(labeled_data):
    labeled_data
    return


@app.cell
def _(df, generate_multi_label_integers, get_GMM_prob_df, pca_arr, pd):
    unlabeled_data = pd.concat(
        [df, generate_multi_label_integers(get_GMM_prob_df(pca_arr))], axis=1
    ).drop(columns=["user_id"])
    unlabeled_data
    return


@app.cell
def _(df, generate_multi_label_series, get_GMM_prob_df, pca_arr, pd):
    labeled_data = pd.concat(
        [df, generate_multi_label_series(get_GMM_prob_df(pca_arr))], axis=1
    ).drop(columns=["user_id"])
    return (labeled_data,)


@app.cell
def _(labeled_data):
    grouped_labels = labeled_data.explode("multi_cluster_labels").groupby(
        "multi_cluster_labels"
    )
    dict = {group_val: labeled_data for group_val, labeled_data in grouped_labels}

    description_dict = {name: df.describe() for name, df in dict.items()}
    return


@app.cell
def _(
    generate_multi_label_integers,
    get_GMM_prob_df,
    pca_arr,
    pd,
    preprocessed_df,
    train_test_split,
):
    preprocessed_wo_labels_df = pd.concat(
        [preprocessed_df, generate_multi_label_integers(get_GMM_prob_df(pca_arr))], axis=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_wo_labels_df.drop(columns=["multi_cluster_indices"], axis=1),
        preprocessed_wo_labels_df["multi_cluster_indices"],
        test_size=0.3,
        random_state=42,
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    LogisticRegression,
    MultiLabelBinarizer,
    MultiOutputClassifier,
    X_test,
    X_train,
    y_train,
):
    # Encode multiple labels 
    mlb = MultiLabelBinarizer(classes=list(range(11)))
    prediction = MultiOutputClassifier(LogisticRegression()).fit(X_train, mlb.fit_transform(y_train)).predict(X_test)
    return mlb, prediction


@app.cell
def _(hamming_loss, mlb, prediction, y_test):
    # TODO: What does the hamming loss metric mean
    round(hamming_loss(mlb.fit_transform(y_test), prediction), 5)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
