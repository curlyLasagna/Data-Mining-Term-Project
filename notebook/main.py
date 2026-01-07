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
    from sklearn.metrics import hamming_loss, accuracy_score
    from sklearn.cluster import DBSCAN, KMeans
    import altair as alt
    import marimo as mo
    import numpy as np
    return (
        FunctionTransformer,
        GaussianMixture,
        KMeans,
        LogisticRegression,
        MultiLabelBinarizer,
        MultiOutputClassifier,
        OneHotEncoder,
        PCA,
        StandardScaler,
        accuracy_score,
        alt,
        hamming_loss,
        make_column_transformer,
        make_pipeline,
        mo,
        np,
        pd,
        silhouette_score,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("Tech_Use_Stress_Wellness.csv")
    df.replace({False: 0, True: 1}, inplace=True)
    return (df,)


@app.cell
def _(Union, alt, df, pd):
    def plot_column_barchart_altair(
        df: pd.DataFrame, column_name: str
    ) -> Union[alt.Chart, str]:
        """
        Generates an Altair bar chart for the distribution of a specified column.

        The function intelligently handles the column type:
        1. If the column has low granularity (few unique values), it plots a simple
           bar chart of unique value counts, sorted by count.
        2. If the column is numeric with high granularity, it plots a histogram
           (a binned bar chart) for a better visualization of the distribution.
        3. If the column has excessive unique values (e.g., ID columns), it returns
           an error message.

        Args:
            df (pd.DataFrame): The input pandas DataFrame.
            column_name (str): The name of the column to plot.

        Returns:
            alt.Chart: The generated Altair chart object.
            str: A message if the column is not suitable for a simple bar chart.
        """
        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in the DataFrame."

        is_numeric = pd.api.types.is_numeric_dtype(df[column_name])
        unique_count = df[column_name].nunique()

        # Heuristic for deciding between count bar chart and histogram/error
        if is_numeric and unique_count > 15 and unique_count < 1000:
            # Use a histogram for high-granularity numeric data
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(f"{column_name}:Q", bin=True, title=column_name),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[
                        alt.Tooltip(
                            f"{column_name}:Q", bin=True, title=column_name
                        ),
                        "count():Q",
                    ],
                )
                .properties(title=f"Histogram of {column_name}")
                .interactive()
            )
            return chart
        elif unique_count >= 1000:
            # Prevent plotting columns like user_id or highly unique strings
            return f"Column '{column_name}' has too many unique values ({unique_count}) for a simple bar chart. Consider grouping the data."
        else:
            # Simple bar chart for low-granularity data (categorical/discrete)
            # 1. Calculate the value counts and convert to a DataFrame for Altair
            counts_df = df[column_name].astype(str).value_counts().reset_index()
            counts_df.columns = [column_name, "Count"]

            # 2. Create the base chart
            base = alt.Chart(counts_df).properties(
                title=f"Distribution of {column_name}"
            )

            # 3. Create the bar chart
            bars = base.mark_bar().encode(
                # Encode the column value (as nominal) on the Y-axis, sorted by count
                y=alt.Y(f"{column_name}:N", title=column_name, sort="-x"),
                # Encode the count (as quantitative) on the X-axis
                x=alt.X("Count:Q", title="Count"),
                tooltip=[column_name, "Count"],
            )

            # 4. Add text labels to the bars
            text = bars.mark_text(
                align="left",
                baseline="middle",
                dx=3,  # Nudge text to the right
            ).encode(
                text="Count:Q",
                x=alt.X(
                    "Count:Q", axis=None
                ),  # Keep X-encoding for position, but hide axis ticks/labels
                color=alt.value("black"),
            )

            # 5. Combine and return
            return (bars + text).interactive()


    plot_column_barchart_altair(df, "work_related_hours").save(
        fp="work_related.png", scale_factor=2
    )
    return


@app.cell
def _(df):
    df
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


@app.cell
def _(alt, df, pd):
    def generate_altair_heatmap(df: pd.DataFrame) -> alt.Chart:
        """
        Generates a correlation heatmap for all numerical and boolean features
        in the dataset using Altair.

        The function standardizes feature handling and prepares the data for
        Altair's long-form data structure required for heatmaps.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            alt.Chart: An interactive Altair chart object representing the heatmap.
        """
        # 1. Select Numerical and Boolean Features
        numerical_cols = df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        features_for_correlation = [
            col for col in numerical_cols if col not in ["user_id"]
        ]

        df_corr = df[features_for_correlation].copy()

        # 2. Convert Boolean Features to Integers (True=1, False=0)
        for col in df_corr.select_dtypes(include=["bool"]).columns:
            df_corr[col] = df_corr[col].astype(int)

        # 3. Calculate the Correlation Matrix
        correlation_matrix = df_corr.corr()

        # 4. Convert the Wide-Format Correlation Matrix to Long-Format for Altair
        # Altair needs a column for 'Feature1', 'Feature2', and 'Correlation'
        correlation_matrix.index.name = "Feature1"
        correlation_matrix = correlation_matrix.reset_index()
        df_long = correlation_matrix.melt(
            id_vars="Feature1",
            value_vars=correlation_matrix.columns.drop("Feature1"),
            var_name="Feature2",
            value_name="Correlation",
        )

        # 5. Generate the Altair Heatmap

        # Define the base chart
        base = (
            alt.Chart(df_long)
            .encode(
                x=alt.X("Feature1", title="", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("Feature2", title=""),
                tooltip=[
                    "Feature1",
                    "Feature2",
                    alt.Tooltip("Correlation", format=".2f"),
                ],
            )
            .properties(
                title="Correlation Heatmap of Usage and Wellness Metrics",
                width=1000,
            )
        )

        # Create the heatmap rectangles
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "Correlation",
                scale=alt.Scale(
                    range="diverging", scheme="blueorange"
                ),  # Cool for correlation
                legend=alt.Legend(title="Correlation Value"),
            ),
        )

        # Add text labels for the correlation value
        text = base.mark_text().encode(
            text=alt.Text("Correlation", format=".2f"),
            # Determine text color based on background color (correlation value) for readability
            color=alt.condition(
                alt.datum.Correlation > 0.5, alt.value("white"), alt.value("black")
            ),
        )

        # Combine the heatmap and text layer
        chart = (heatmap + text).interactive()

        return chart


    generate_altair_heatmap(df)
    return


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
def _(Optional, PCA, alt, pd):
    def plot_pca_2d(
        df: pd.DataFrame, color_column: Optional[str] = None
    ) -> alt.Chart:
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df)

        # 4. Create result DataFrame
        pca_df = pd.DataFrame(
            data=principal_components,
            columns=["Principal Component 1", "Principal Component 2"],
        )

        # Add the color column back for plotting
        if color_column:
            # Resetting index ensures the indices align for concatenation
            col_series = df[color_column].reset_index(drop=True)

            # --- MODIFICATION START ---
            # Lambda function to take the first element if it's a list/tuple
            col_series = col_series.apply(
                lambda x: x[0]
                if isinstance(x, (list, tuple)) and len(x) > 0
                else x
            )
            # Ensure it is treated as an integer (optional, but good for consistency)
            # We use 'Int64' to handle potential NaNs safely if necessary, or just regular int
            try:
                col_series = col_series.astype(int)
            except (ValueError, TypeError):
                pass  # Keep as is if conversion fails
            # --- MODIFICATION END ---

        # Calculate explained variance for the axis titles
        explained_variance = pca.explained_variance_ratio_ * 100
        pc1_label = f"PC 1 ({explained_variance[0]:.2f}%)"
        pc2_label = f"PC 2 ({explained_variance[1]:.2f}%)"

        # 5. Generate Altair Plot
        chart = (
            alt.Chart(pca_df)
            .mark_circle(size=60)
            .encode(
                # Set the x-axis to the first principal component
                x=alt.X("Principal Component 1", title=pc1_label),
                # Set the y-axis to the second principal component
                y=alt.Y("Principal Component 2", title=pc2_label),
                # Add tooltips for interactivity
                tooltip=["Principal Component 1", "Principal Component 2"]
                + ([color_column] if color_column else []),
            )
            .properties(title="2D PCA Scatter Plot")
            .interactive()
        )  # Allow zooming and panning

        # Apply color encoding if a column was provided
        if color_column:
            chart = chart.encode(color=color_column)

        return chart
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
def _(PCA, preprocessed_df):
    pca_model = PCA(n_components=15)
    pca_arr = pca_model.fit_transform(preprocessed_df)
    return pca_arr, pca_model


@app.cell
def _(alt, models, np, pca_arr, pd, silhouette_score):
    def plot_silhouette_curve(
        pca_arr: np.ndarray, gmm_models: list, n_components_start: int = 1
    ) -> alt.Chart:
        """
        Generates an Altair line chart to visualize the Silhouette Scores
        for a list of fitted GMM models.

        Args:
            pca_arr (np.ndarray): The data used for clustering (PCA-transformed).
            gmm_models (list): A list of fitted GaussianMixture models.
            n_components_start (int): The starting number of components (usually 1).

        Returns:
            alt.Chart: An Altair chart showing the Silhouette score curve.
        """
        silhouette_scores = []
        for model in gmm_models[n_components_start:]:
            # Get the predicted labels for the current model
            labels = model.predict(pca_arr)
            # Calculate the silhouette score
            score = silhouette_score(pca_arr, labels)
            silhouette_scores.append(score)

        # The scores list starts from n_components = 2
        n_components = np.arange(
            n_components_start + 1, len(gmm_models) + n_components_start
        )

        # 1. Create DataFrame
        data = pd.DataFrame(
            {
                "Number of Components": n_components,
                "Silhouette Score": silhouette_scores,
            }
        )

        # 2. Find the optimal number of components (maximum score)
        i_best = np.argmax(silhouette_scores)
        best_n_components = data.iloc[i_best]["Number of Components"]
        max_score = data.iloc[i_best]["Silhouette Score"]

        # 3. Base chart setup
        base = (
            alt.Chart(data)
            .encode(
                x=alt.X("Number of Components:Q", title="Number of Components"),
                y=alt.Y("Silhouette Score:Q", title="Silhouette Score"),
                tooltip=[
                    "Number of Components",
                    alt.Tooltip("Silhouette Score", format=".3f"),
                ],
            )
            .properties(title="Silhouette Score vs. Number of GMM Components")
        )

        # 4. Line layer
        line = base.mark_line(point=True).encode(color=alt.value("darkgreen"))

        # 5. Point layer to highlight the maximum score (optimal components)
        min_point = base.mark_point(filled=True, size=150, color="orange").encode(
            opacity=alt.condition(
                alt.datum["Number of Components"] == best_n_components,
                alt.value(1.0),
                alt.value(0.0),
            )
        )

        # 6. Add a rule for the max score
        max_score_rule = (
            alt.Chart(pd.DataFrame({"y": [max_score]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y")
        )

        # 7. Combine all layers
        chart = (line + min_point + max_score_rule).interactive()
        return chart


    plot_silhouette_curve(pca_arr, models).save(
        fp="silhoutte_comparison.png", scale_factor=2
    )
    return


@app.cell
def _(GaussianMixture, np, pca_arr):
    def getGMM_components(X) -> list:
        N = np.arange(1, 15)

        models = [None for n in N]
        for i in range(len(N)):
            models[i] = GaussianMixture(
                n_components=N[i],
                max_iter=1000,
                covariance_type="full",
                random_state=42,
            )
            models[i].fit(X)
        return models


    models = getGMM_components(pca_arr)

    AIC = [m.aic(pca_arr) for m in models]
    BIC = [m.bic(pca_arr) for m in models]

    # index of GMM with the best BIC score
    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    best_GMM_cluster_count = np.arange(1, 15)[i_best]
    print("best fit converged:", gmm_best.converged_)
    print("BIC: n_components =  %i" % np.arange(1, 15)[i_best])
    return BIC, gmm_best, models


@app.cell
def _(BIC, alt, np, pd):
    def plot_bic_curve(bic_list: list, n_components_start: int = 1) -> alt.Chart:
        """
        Generates an Altair line chart to visualize the BIC values for GMM models.

        Args:
            bic_list (list): A list of BIC values.
            n_components_start (int): The starting number of components (usually 1).

        Returns:
            alt.Chart: An Altair chart showing the BIC curve with the minimum highlighted.
        """
        # 1. Create DataFrame
        data = pd.DataFrame(
            {
                "Number of Components": np.arange(
                    n_components_start, len(bic_list) + n_components_start
                ),
                "BIC": bic_list,
            }
        )

        # 2. Find the optimal number of components (minimum BIC)
        i_best = np.argmin(bic_list)
        best_n_components = data.iloc[i_best]["Number of Components"]

        # 3. Base chart setup
        base = (
            alt.Chart(data)
            .encode(
                x=alt.X("Number of Components:Q", title="Number of Components"),
                y=alt.Y("BIC:Q", title="Bayesian Information Criterion (BIC)"),
                tooltip=["Number of Components", alt.Tooltip("BIC", format=".2f")],
            )
            .properties(title="BIC vs. Number of GMM Components")
        )

        # 4. Line layer
        line = base.mark_line(point=True).encode(color=alt.value("darkblue"))

        # 5. Point layer to highlight the minimum BIC (optimal components)
        min_point = base.mark_point(filled=True, size=150, color="red").encode(
            opacity=alt.condition(
                alt.datum["Number of Components"] == best_n_components,
                alt.value(1.0),
                alt.value(0.0),
            )
        )

        # 6. Combine all layers
        chart = (line + min_point).interactive()
        return chart


    plot_bic_curve(BIC)

    # .save(fp="BIC_curve.png", scale_factor=2)
    return


@app.cell
def _(df, gmm_best, pd):
    def get_GMM_prob_df(data):
        """Gets a list of probabilities that a data point belongs to a cluster through GMM"""
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
        CURRENT_THRESHOLD = 0.10  # The threshold the user is currently considering
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
    )
    # .save(fp="prob_dist.png", scale_factor=2)
    return


@app.cell
def _(CLUSTER_NAMES, Dict, PCA, alt, gmm_best, pca_arr, pca_model, pd):
    def map_gmm_labels_to_names(
        integer_labels: pd.Series, cluster_names_map: Dict[int, str]
    ) -> pd.Series:
        """
        Maps integer GMM cluster labels (0, 1, 2, ...) to their descriptive names.

        Args:
            integer_labels (pd.Series): The integer cluster labels
                                        returned by gmm_best.predict(pca_arr).
            cluster_names_map (Dict[int, str]): A dictionary mapping the
                                                integer index to the cluster name.

        Returns:
            pd.Series: A Series of descriptive cluster names.
        """
        name_labels = integer_labels.map(
            lambda x: cluster_names_map.get(x, f"Unknown Cluster {x}")
        )
        name_labels.name = "Cluster Name"
        return name_labels


    def plot_gmm_clusters_2d(
        pca_arr: pd.DataFrame, cluster_labels: pd.Series, pca_model: PCA
    ) -> alt.Chart:
        """
        Generates a 2D scatter plot of GMM clusters using the first two PCA components.

        Args:
            pca_arr (pd.DataFrame): The data transformed into PCA space.
            cluster_labels (pd.Series): The integer cluster label (0 to N-1)
                                        predicted by gmm_best.predict(pca_arr).
            pca_model (PCA): The fitted PCA model object.

        Returns:
            alt.Chart: An interactive Altair chart object.
        """
        # 1. Create result DataFrame for plotting
        pca_df = pd.DataFrame(
            data=pca_arr[:, 0:2],
            columns=["Principal Component 1", "Principal Component 2"],
        )

        # 2. Add cluster labels and convert to string for discrete coloring
        # Using .reset_index(drop=True) ensures index alignment
        pca_df["Cluster"] = cluster_labels.astype(str).reset_index(drop=True)

        # 3. Calculate explained variance for the axis titles
        explained_variance = pca_model.explained_variance_ratio_ * 100
        pc1_label = f"PC 1"
        pc2_label = f"PC 2"

        # 4. Generate Altair Plot
        chart = (
            alt.Chart(pca_df)
            .mark_circle(size=60)
            .encode(
                # Set the x-axis to the first principal component
                x=alt.X("Principal Component 1", title=pc1_label),
                # Set the y-axis to the second principal component
                y=alt.Y("Principal Component 2", title=pc2_label),
                # Color by the cluster label (nominal/discrete data type)
                color=alt.Color(
                    "Cluster:N",
                    # title='GMM Cluster',
                    legend=alt.Legend(
                        orient="bottom", direction="horizontal", titleOrient="left"
                    ),
                ),
                # Add tooltips for interactivity
                tooltip=[
                    "Principal Component 1",
                    "Principal Component 2",
                    "Cluster",
                ],
            )
            .properties(title="Scatter Plot by Clusters", width=1000)
            .interactive()
        )  # Allow zooming and panning

        return chart


    # 1. Get the integer labels from the best GMM model
    gmm_integer_labels = pd.Series(gmm_best.predict(pca_arr))

    # 2. Map the integer labels to the descriptive names
    gmm_name_labels = map_gmm_labels_to_names(gmm_integer_labels, CLUSTER_NAMES)

    # 3. Visualize using the descriptive names
    (
        plot_gmm_clusters_2d(pca_arr, gmm_name_labels, pca_model).save(
            fp="gmm_clusters_tied_spherical.png", scale_factor=2
        )
    )
    return (gmm_integer_labels,)


@app.cell
def _(gmm_integer_labels, pca_arr, silhouette_score):
    silhouette_score(pca_arr, gmm_integer_labels)
    return


@app.cell
def _(pd):
    from typing import Dict, List, Tuple

    # --- Constants (Defined globally for scope, but can stay inside the function) ---
    CLUSTER_NAMES = {
        0: "Struggling & Disengaged",
        1: "High-Stress Youth",
        2: "Male High-Screen Workers",
        3: "Actively Managing Wellness",
        4: "Highly Healthy & Active Seniors",
        5: "Average & Health-Conscious",
        6: "Balanced & Mindful",
        7: "Naturally Well & Disengaged",
    }
    PROBABILITY_THRESHOLD: float = 0.20


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
    return (
        CLUSTER_NAMES,
        Dict,
        List,
        generate_multi_label_integers,
        generate_multi_label_series,
    )


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
    labeled_data
    return


@app.cell
def _(gmm_best, pca_arr, silhouette_score):
    silhouette_score(pca_arr, gmm_best.predict(pca_arr))
    return


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
        [preprocessed_df, generate_multi_label_integers(get_GMM_prob_df(pca_arr))],
        axis=1,
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
    mlb = MultiLabelBinarizer(classes=list(range(8)))
    prediction = (
        MultiOutputClassifier(LogisticRegression())
        .fit(X_train, mlb.fit_transform(y_train))
        .predict(X_test)
    )
    return mlb, prediction


@app.cell
def _(hamming_loss, mlb, prediction, y_test):
    # TODO: What does the hamming loss metric mean
    round(hamming_loss(y_true=mlb.fit_transform(y_test), y_pred=prediction), 5)
    return


@app.cell
def _(accuracy_score, mlb, prediction, y_test):
    accuracy_score(prediction, mlb.fit_transform(y_test))
    return


@app.cell
def _(labeled_data, pd):
    def get_cluster_label_counts(
        df: pd.DataFrame, column: str = "multi_cluster_labels"
    ) -> pd.DataFrame:
        """
        Explodes the list-based label column and returns a DataFrame with the
        count of each unique cluster label.
        """
        # 1. Explode the list column so each label is on a separate row
        exploded_series = df.explode(column)[column]

        # 2. Count occurrences and reset index to create a proper DataFrame
        counts_df = exploded_series.value_counts().reset_index()

        # 3. Rename columns for clarity
        counts_df.columns = ["Cluster Label", "Count"]

        return counts_df


    get_cluster_label_counts(labeled_data)
    return (get_cluster_label_counts,)


@app.cell
def _(alt, get_cluster_label_counts, labeled_data, pd):
    def plot_cluster_label_counts(counts_df: pd.DataFrame) -> alt.Chart:
        """
        Generates an Altair horizontal bar chart for cluster label counts.

        Args:
            counts_df (pd.DataFrame): DataFrame with 'Cluster Label' and 'Count'.

        Returns:
            alt.Chart: An interactive Altair chart object.
        """
        # 1. Base Chart
        base = (
            alt.Chart(counts_df)
            .encode(
                # Sort Cluster Label by Count in descending order ('sort="-x"')
                y=alt.Y("Cluster Label:N", sort="-x", title="Cluster"),
                x=alt.X("Count:Q", title="Number of Users"),
                tooltip=["Cluster Label", "Count"],
            )
            .properties(
                title="Distribution of Users Across Clusters",
                width=400,
                height=300,
            )
        )

        # 2. Bar Marks
        bars = base.mark_bar().encode(
            # Color by Cluster Label for visual separation
            color=alt.Color("Cluster Label:N", title="Cluster")
        )

        # 3. Add Text Labels
        text = base.mark_text(
            align="left",
            baseline="middle",
            dx=3,  # Nudge text to the right of the bar
        ).encode(
            x=alt.X(
                "Count:Q", axis=None
            ),  # Keep the x-encoding for position, but hide axis
            text=alt.Text("Count:Q"),
            color=alt.value("black"),  # Set text color
        )

        # Combine bars and text and make it interactive
        chart = (bars + text).interactive(bind_y=False)

        return chart


    plot_cluster_label_counts(get_cluster_label_counts(labeled_data))
    return


@app.cell
def _(labeled_data, pd):
    def get_label_count_distribution(
        df: pd.DataFrame, column: str = "multi_cluster_labels"
    ) -> pd.DataFrame:
        """
        Calculates the distribution of the number of labels assigned to each user.
        Groups by the count of labels in the list for each row.
        """
        # 1. Calculate the number of labels for each row
        label_counts_series = df[column].apply(len)

        # 2. Count the occurrences of each unique label count
        distribution_df = label_counts_series.value_counts().reset_index()

        # 3. Rename columns for clarity
        distribution_df.columns = ["Number of Labels", "Number of Users"]

        # 4. Sort by the number of labels (ascending)
        distribution_df = distribution_df.sort_values(
            by="Number of Labels", ascending=True
        )

        return distribution_df


    get_label_count_distribution(labeled_data)
    return


@app.cell
def _(KMeans, np, pca_arr):
    def get_kmeans_inertia(
        X: np.ndarray, n_components_range: np.ndarray = np.arange(1, 16)
    ) -> tuple[list, list]:
        """
        Fits K-Means models for a range of components and calculates the inertia (WSS).

        Args:
            X (np.ndarray): The data used for clustering (PCA-transformed).
            n_components_range (np.ndarray): The range of K values to test.

        Returns:
            tuple[list, list]: A tuple containing the list of fitted KMeans models
                               and the list of their corresponding inertia scores.
        """
        kmeans_models = []
        inertia_scores = []

        for k in n_components_range:
            # For k=1, max_iter=1 is used as it is mathematically trivial (all points in one cluster)
            # For k > 1, the algorithm runs normally.
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(X)
            kmeans_models.append(kmeans)
            inertia_scores.append(kmeans.inertia_)

        return kmeans_models, inertia_scores


    models_k_means, inertia_k_means = get_kmeans_inertia(pca_arr)
    return inertia_k_means, models_k_means


@app.cell
def _(alt, inertia_k_means, np, pd):
    def plot_elbow_curve(
        inertia_list: list, n_components_start: int = 1
    ) -> alt.Chart:
        """
        Generates an Altair line chart to visualize the K-Means Inertia (Elbow Curve).

        Args:
            inertia_list (list): A list of Inertia (WSS) values.
            n_components_start (int): The starting number of components (usually 1).

        Returns:
            alt.Chart: An Altair chart showing the Inertia curve.
        """
        # 1. Create DataFrame
        data = pd.DataFrame(
            {
                "Number of Components (K)": np.arange(
                    n_components_start, len(inertia_list) + n_components_start
                ),
                "Inertia (WSS)": inertia_list,
            }
        )

        # 2. Base chart setup
        base = (
            alt.Chart(data)
            .encode(
                x=alt.X(
                    "Number of Components (K):Q", title="Number of Clusters (K)"
                ),
                y=alt.Y(
                    "Inertia (WSS):Q",
                    title="Inertia (Within-Cluster Sum of Squares)",
                ),
                tooltip=[
                    "Number of Components (K)",
                    alt.Tooltip("Inertia (WSS)", format=".2f"),
                ],
            )
            .properties(title="K-Means Elbow Curve")
        )

        # 3. Line and Point layers
        chart = base.mark_line(point=True).encode(color=alt.value("darkred"))

        # 4. Combine and return
        return chart.interactive()


    # --- Execution ---
    plot_elbow_curve(inertia_k_means).save(fp="kmeans_elbow.png", scale_factor=2)
    return


@app.cell
def _(List, alt, models_k_means, np, pca_arr, pd, silhouette_score):
    def plot_kmeans_silhouette_curve(
        pca_arr: np.ndarray, kmeans_models: List, n_components_start: int = 1
    ) -> alt.Chart:
        """
        Generates an Altair line chart to visualize the Silhouette Scores
        for a list of fitted K-Means models.

        Args:
            pca_arr (np.ndarray): The data used for clustering (PCA-transformed).
            kmeans_models (list): A list of fitted KMeans models (K=1, K=2, ...).
            n_components_start (int): The starting K for the models list (usually 1).

        Returns:
            alt.Chart: An Altair chart showing the Silhouette score curve.
        """
        silhouette_scores = []
        # Silhouette score is not defined for n_components = 1, so we skip the first model
        for model in kmeans_models[n_components_start:]:
            # Get the predicted labels for the current model
            labels = model.predict(pca_arr)
            # Calculate the silhouette score
            score = silhouette_score(pca_arr, labels)
            silhouette_scores.append(score)

        # The K values start from 2 (n_components_start + 1)
        n_clusters = np.arange(
            n_components_start + 1, len(kmeans_models) + n_components_start
        )

        # 1. Create DataFrame
        data = pd.DataFrame(
            {
                "Number of Clusters (K)": n_clusters,
                "Silhouette Score": silhouette_scores,
            }
        )

        # 2. Find the optimal number of clusters (maximum score)
        i_best = np.argmax(silhouette_scores)
        best_n_clusters = data.iloc[i_best]["Number of Clusters (K)"]
        max_score = data.iloc[i_best]["Silhouette Score"]

        # 3. Base chart setup
        base = (
            alt.Chart(data)
            .encode(
                x=alt.X(
                    "Number of Clusters (K):Q", title="Number of Clusters (K)"
                ),
                y=alt.Y("Silhouette Score:Q", title="Silhouette Score"),
                tooltip=[
                    "Number of Clusters (K)",
                    alt.Tooltip("Silhouette Score", format=".3f"),
                ],
            )
            .properties(title="K-Means Silhouette Score vs. Number of Clusters")
        )

        # 4. Line layer
        line = base.mark_line(point=True).encode(color=alt.value("darkblue"))

        # 5. Point layer to highlight the maximum score (optimal components)
        min_point = base.mark_point(filled=True, size=150, color="orange").encode(
            opacity=alt.condition(
                alt.datum["Number of Clusters (K)"] == best_n_clusters,
                alt.value(1.0),
                alt.value(0.0),
            )
        )

        # 6. Add a rule for the max score
        max_score_rule = (
            alt.Chart(pd.DataFrame({"y": [max_score]}))
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(y="y")
        )

        # 7. Combine all layers
        chart = (line + min_point + max_score_rule).interactive()

        return chart


    plot_kmeans_silhouette_curve(pca_arr, models_k_means).save(
        fp="kmeans_silhoutte.png", scale_factor=2
    )
    return


@app.cell
def _(KMeans, np, pd):
    def get_kmeans_labels_series(
        pca_arr: np.ndarray, n_clusters: int = 8, index: pd.Index = None
    ) -> pd.Series:
        """
        Fits a KMeans model to the data and returns a Pandas Series of the
        hard cluster labels (integers 0 to n_clusters-1).

        Args:
            pca_arr (np.ndarray): The data used for clustering (PCA-transformed).
            n_clusters (int): The number of clusters (K) to use for K-Means.
            index (pd.Index): The index to assign to the resulting Series
                              (typically the index of the original DataFrame).

        Returns:
            pd.Series: A Series containing the integer cluster label for each row.
        """
        # 1. Initialize and fit the KMeans model
        kmeans_model = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
        )
        kmeans_model.fit(pca_arr)

        # 2. Predict the cluster labels
        labels = kmeans_model.predict(pca_arr)

        # 3. Create the Pandas Series
        labels_series = pd.Series(
            data=labels, index=index, name=f"kmeans_cluster_{n_clusters}"
        )

        return labels_series
    return


@app.cell
def _(KMeans, df, np, pca_arr, pd):
    def get_kmeans_labeled_series(
        pca_arr: np.ndarray,
        n_clusters: int = 8,
        index: pd.Index = None,
        to_label: bool = False,
    ) -> pd.Series:
        """
        Fits a KMeans model to the data and returns a Pandas Series of either
        integer cluster IDs or string cluster labels.

        Args:
            pca_arr (np.ndarray): The data used for clustering (PCA-transformed).
            n_clusters (int): The number of clusters (K) to use for K-Means.
            index (pd.Index): The index to assign to the resulting Series.
            to_label (bool): If True, maps the integer IDs to string labels
                             from the global CLUSTER_NAMES dictionary.

        Returns:
            pd.Series: A Series containing either integer cluster IDs or
                       string cluster labels for each row.
        """
        # 1. Initialize and fit the KMeans model
        kmeans_model = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300
        )
        kmeans_model.fit(pca_arr)

        # 2. Predict the cluster labels (integers 0 to n_clusters-1)
        labels = kmeans_model.predict(pca_arr)
        # Start with the integer labels as a Pandas Series
        labels_series = pd.Series(labels, index=index)

        two_clusters = {0: "Well-Adjusted", 1: "The Stressed and Screen-Dependent"}

        # 3. Apply labeling if requested
        if to_label:
            # Create a mapping dictionary for the clusters used
            label_map = {
                i: two_clusters.get(i, f"Cluster {i}") for i in range(n_clusters)
            }

            # 🌟 KEY CHANGE: Use the .map() method on the Pandas Series
            # This applies the dictionary lookup efficiently
            data_to_use = labels_series.map(label_map)
            name_suffix = "label"
        else:
            # Use the raw integer labels
            data_to_use = labels
            name_suffix = "id"

        # 4. Create the final Pandas Series
        final_series = pd.Series(data=data_to_use, index=index, name="cluster")

        return final_series


    kmeans_labels = get_kmeans_labeled_series(
        pca_arr, n_clusters=2, index=df.index, to_label=True
    )

    # 2. Concatenate the labels back to the original labeled_data DataFrame
    # This gives you a new column with the K-Means labels.
    labeled_data_with_kmeans = pd.concat([df, kmeans_labels], axis=1)
    labeled_data_with_kmeans
    return (kmeans_labels,)


@app.cell
def _(PCA, alt, kmeans_labels, pd, preprocessed_df):
    def plot_kmeans_clusters_pca(
        df_features: pd.DataFrame,
        cluster_labels: pd.Series,
        n_components_pca: int = 2,
        cluster_column_name: str = "cluster",
    ) -> alt.Chart:
        """
        Generates a 2D scatter plot of K-Means clusters using the first two PCA components.

        Args:
            df_features: The preprocessed data used for clustering (e.g., preprocessed_df).
            cluster_labels: A Series containing the K-Means cluster labels for each row.
            n_components_pca: The number of components to fit in PCA (must be >= 2 for 2D plot).
            cluster_column_name: The name to use for the cluster column in the plot.

        Returns:
            An interactive Altair Chart object.
        """

        # 1. Fit PCA and transform the data
        pca_model = PCA(n_components=n_components_pca)
        principal_components = pca_model.fit_transform(df_features)

        # 2. Create result DataFrame for plotting
        pca_df = pd.DataFrame(
            data=principal_components[:, 0:2],
            columns=["Principal Component 1", "Principal Component 2"],
        )

        # 3. Add cluster labels and convert to string for discrete coloring
        # Using .reset_index(drop=True) ensures index alignment with the PCA array
        pca_df[cluster_column_name] = cluster_labels.astype(str).reset_index(
            drop=True
        )

        # 4. Calculate explained variance for the axis titles
        explained_variance = pca_model.explained_variance_ratio_ * 100
        pc1_label = f"PC 1 ({explained_variance[0]:.2f}%)"
        pc2_label = f"PC 2 ({explained_variance[1]:.2f}%)"

        # 5. Generate Altair Plot
        chart = (
            alt.Chart(pca_df)
            .mark_circle(size=60)
            .encode(
                # Set the x-axis to the first principal component
                x=alt.X("Principal Component 1", title=pc1_label),
                # Set the y-axis to the second principal component
                y=alt.Y("Principal Component 2", title=pc2_label),
                # Color by the cluster label (nominal/discrete data type)
                color=alt.Color(
                    f"{cluster_column_name}:N", title="K-Means Cluster"
                ),
                # Add tooltips for interactivity
                tooltip=[
                    "Principal Component 1",
                    "Principal Component 2",
                    cluster_column_name,
                ],
            )
            .properties(
                title="2D PCA Scatter Plot of K-Means Clusters",
                width=600,
                height=400,
            )
            .interactive()
        )  # Allows zooming and panning

        return chart


    plot_kmeans_clusters_pca(preprocessed_df, kmeans_labels).save(
        fp="two_cluster_scatter.png", scale_factor=2
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
