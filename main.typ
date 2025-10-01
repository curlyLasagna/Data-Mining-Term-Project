#import "@preview/charged-ieee:0.1.4": ieee

#let pcite(key) = { cite(key, form: "normal") }
#set cite(form: "prose", style: "harvard-cite-them-right")
#show link: underline
#show: ieee.with(
  title: [Clustering Analysis of Screen Time and Wellness (Proposal)],
  abstract: [
We examine how device usage affects personal wellness using a dataset that tracks hours spent on devices, technology habits, and related wellness metrics. Drawing from customer segmentation techniques in marketing, our approach groups users with similar digital behaviors and wellness patterns. Using a trained classifier model, we then recommend personalized strategies to help individuals enhance their well-being.
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
The rising use of technology has been linked to negative impacts on mental health and overall well-being. This proposal examines how people use technology across various devices such as phones, laptops, tablets, and televisions, and for different activities including social media, work, entertainment, and gaming. We plan to explore how these usage patterns relate to wellness metrics such as sleep quality, mood, stress levels, mental health assessments, and other lifestyle factors. Using this data, we want to identify user segments and train a classification model to recommend personalized strategies that promote healthier technology habits, taking inspiration from user segmentation on the field of marketing.

= Dataset

The dataset that we'll be working with contains 5,000 rows and is publicly available from #link("https://www.kaggle.com/datasets/nagpalprabhavalkar/tech-use-and-stress-wellness")[Kaggle]. It describes device usage hours, usage types, and a range of wellness metrics (sleep quality, mood, stress levels, mental health score, healthy eating, caffeine intake, weekly anxiety, weekly depression, and mindfulness).


= Related Works
#pcite(<analytics2040042>) compared several leading clustering algorithms, including K-means, Gaussian Mixture Models (GMM), DBSCAN, agglomerative clustering, and BIRCH, for customer segmentation in the UK retail market. Their findings indicated that GMM, when used alongside PCA for dimensionality reduction, outperformed other methods by achieving a Silhouette Score of 0.80, whereas K-means, BIRCH, and agglomerative algorithms all scored 0.64. A score closer to 1 indicates that the clusters are more defined.

Another study by #pcite(<10921704>) evaluated clustering performance using both the Silhouette Score and the Davies-Bouldin Index. A higher Davies-Bouldin Index implies that clusters are less compact and not well separated. This research observed that Gaussian Mixture Models encounter difficulties when handling high-dimensional or large-scale datasets, while K-means++ produced more reliable results even in the presence of high dimensionality and noise.

In the context of mental health, research such as this study by #pcite(<Lee2024DigitalWellbeing>) explored the relationship between technology use and psychological well-being. The study found that active digital engagement is positively associated with anxiety symptoms, and that access to the internet correlates with higher levels of depression and anxiety, especially among younger individuals. Nevertheless, the paper emphasized that there is no clear causal link established between technology use and mental health outcomes.

= Methods

We propose an approach that combines clustering and classification for user segmentation. To capture overlapping user groups, we will use Gaussian Mixture Models (GMMs), which offer soft clustering as opposed to K-Means, which provides hard clusters. This is important because individuals often display multiple, overlapping behaviors in their technology use.

Based on the methodology from #pcite(<analytics2040042>) and the insights from #pcite(<10921704>) regarding GMMs and high dimensionality, we will apply Principal Component Analysis (PCA) to reduce the dimensionality of our dataset before clustering. We will start by using the elbow method to identify the optimal number of principal components, with the help of the scree plot for visualization.

Before running the clustering algorithm, we will determine the optimal number of Gaussian components by evaluating a range of components and examining changes in the Bayesian Information Criterion (BIC) as  #pcite(<Lavorini_2018>) showcased in the author's Medium blog. Once the clusters are formed, we will assign labels to each group based on shared characteristics.

After clustering, we will train a supervised classification model using the cluster labels, with a 70/30 split for training and testing the data.

== Toolset

Our main tool of choice will be Python for its simple syntax and massive collection of popular packages from PyPi. Polars will serve as our dataframe library for data manipulation, a Rust alternative to Pandas, sklearn for the algorithms mentioned, and Altair for interactive visualization. Marimo will be used as our interactive prototyping notebook, and Streamlit for developing a prototype frontend once we are satisfied with the model. To ensure reproducibility, dependencies will be managed with uv, a Rust alternative to pip, and #link("https://devenv.sh/")[devenv].

= Expected Outcomes

We expect GMMs to reveal meaningful user clusters to show distinct profiles of technology use and wellness. Coming up with appropriate labels for these clusters will require thorough analysis and interpretation. Picking the proper strategies to enable individuals to use technology more mindfully, leading to improved well-being will be quite the challenge as usage patterns may be much more complex than expected. The methodology and findings may further inform the development of personalized digital wellness tools and advice.
