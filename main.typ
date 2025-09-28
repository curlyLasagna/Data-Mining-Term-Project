#import "@preview/charged-ieee:0.1.4": ieee

#let pcite(key) = { cite(key, form: "normal") }
#set cite(form: "prose", style: "harvard-cite-them-right")
#show link: underline
#show: ieee.with(
  title: [Clustering Analysis of Screen Time and Wellness (Proposal)],
  abstract: [
    We explore the impact of device usage on personal wellness by leveraging a dataset that records device hours and technology habits alongside wellness metrics. Taking inspiration from customer segmentation in marketing, our approach segments users based on shared digital behaviors and wellness indicators to recommend personalized strategies for improving well-being using a trained classifier model.
  ],
  authors: (
    (
      name: "Luis Dale Gascon",
      department: [Computer Science],
      organization: [Towson University],
      email: "lgascon1@students.towson.edu",
    ),
  ),
  index-terms: ("wellness", "clustering", "technology usage", "mental health", "unsupervised learning"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction
Increased technology use has been associated with negative effects on mental health and general well-being. This project analyzes patterns of technology use across different devices (phones, laptops, tablets, and televisions) and various purposes (social media, work, entertainment, gaming), relating them to wellness metrics such as sleep quality, mood, stress, mental health scores, and additional lifestyle indicators. After analyzing the data, we will train a clustering model to classify users and suggest strategies for healthier technology use.

= Dataset

The dataset that we'll be working with contains 5,000 rows and is publicly available from #link("https://www.kaggle.com/datasets/nagpalprabhavalkar/tech-use-and-stress-wellness")[Kaggle]. It describes device usage hours, usage types, and a range of wellness metrics (sleep quality, mood, stress levels, mental health score, healthy eating, caffeine intake, weekly anxiety, weekly depression, and mindfulness).


= Related Works
#pcite(<analytics2040042>) compared several state-of-the-art clustering algorithms—including K-means, Gaussian Mixture Models (GMM), DBSCAN, agglomerative clustering, and BIRCH—for customer segmentation in the UK retail market. Their results showed that GMM, paired with PCA for dimensionality reduction, outperformed other approaches by achieving a Silhouette Score of 0.80. A score close to 1 indicates more clearly defined clusters.

Another study by #pcite(<10921704>) evaluated clustering algorithms not only using the Silhouette Score but also the Davies-Bouldin Index. A high Davies-Bouldin Index signifies that clusters are neither compact nor well-separated. This study noted that Gaussian Mixture Models face challenges when processing high-dimensional or large-scale data. They concluded that K-Means++ performed the best.

Research on the relationship between mental health and technology use, such as the paper by #pcite(<Lee2024DigitalWellbeing>), found that active digital use is positively correlated with symptoms of anxiety and that internet access is associated with increased levels of depression and anxiety, particularly among younger populations. However, the paper noted that a causal link between technology use and mental health remains inconclusive.

= Methods

We propose a clustering and classification approach for user segmentation. Gaussian Mixture Models (GMMs), which support soft clustering, will be employed to identify overlapping user clusters—an advantage over K-Means, which yields hard clusters—since individuals may exhibit multiple overlapping behaviors in their technology use.

Following the methodology from #pcite(<analytics2040042>) and considering the comments from #pcite(<10921704>) regarding GMMs and high dimensionality, we will reduce the dimensions of our dataset using PCA. To begin, we will employ the elbow method to determine the optimal number of principal components, rather than retaining an arbitrary number of dimensions.

Before fitting our clustering algorithm, we will determine the optimal number of Gaussian components by testing a range of component counts and analyzing the gradient of the Bayesian Information Criterion (BIC) #pcite(<Lavorini_2018>). Once clusters are identified, we will manually label each cluster based on shared characteristics.

After identifying our clusters, we will train a supervised classification model using the resulting cluster labels with a 70/30 train-test split.

== Toolset

We will use Python for its robust ecosystem. Polars will serve as our dataframe library for data manipulation, sklearn for our algorithms, and Altair for interactive visualization. Marimo will be used as our interactive prototyping notebook, and Streamlit for developing a prototype frontend once we are satisfied with the model. To ensure reproducibility, dependencies will be managed with uv and Nix.

= Expected Outcomes

We expect GMMs to reveal meaningful user clusters that reflect distinct profiles of technology use and wellness. Our framework will suggest targeted strategies to enable individuals to use technology more mindfully, leading to improved well-being. The methodology and findings may further inform the development of personalized digital wellness tools and advice.
