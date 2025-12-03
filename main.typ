#import "@preview/charged-ieee:0.1.4": ieee

#let pcite(key) = { cite(key, form: "normal") }
#set cite(form: "prose", style: "harvard-cite-them-right")
#show link: underline
#show: ieee.with(
  title: [Clustering Analysis of Screen Time and Wellness],
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

== Clustering Method of Choice
We propose an approach that combines clustering and classification for user segmentation. To capture overlapping user groups, we will be utilizing Gaussian Mixture Models (GMMs), which offer soft clustering as opposed to K-Means, which provides hard clusters. This is important because individuals often display multiple, overlapping behaviors in their technology use.

== Dimensionality Reduction
Based on the methodology from #pcite(<analytics2040042>) and the insights from #pcite(<10921704>) regarding GMMs and high dimensionality, we will apply Principal Component Analysis (PCA) to reduce the dimensionality of our dataset before clustering. 

To find the optimal number of components to keep during PCA, we will plot the value of cumulative explained variance per components from a range of 1 to 20, with a goal of choosing the number of components that keeps more than 90% of variance. 

Before running the clustering algorithm, we will determine the optimal number of Gaussian components by evaluating a range of components and examining changes in the Bayesian Information Criterion (BIC) as  #pcite(<Lavorini_2018>) showcased in the author's Medium blog. Once the clusters are formed from their probabilities, we will assign labels to each group based on observed characteristics that these data points share. This observation

After we obtain the probabilities from clustering, we will assign a threshold that assigns that datapoint to 1 or more labels. Our initial threshold is at 40%, so a data point could have at most 2 labels.  

Once the preprocessed dataset has each row labeled, we will split the dataset for test train split via holdout and train a multi-label classifier model to predict the label(s)  

== Toolset

Our primary programming language will be Python, chosen for its straightforward syntax and extensive ecosystem of packages available via PyPI. For data manipulation, we will use Pandas; for machine learning algorithms, scikit-learn; and for interactive visualizations, Altair. Marimo will serve as our interactive prototyping notebook, while Streamlit will be used to develop a prototype frontend once the model is ready. To ensure reproducibility, we will manage dependencies with uv (a Rust-based alternative to pip) and #link("https://devenv.sh/")[devenv].

= Analysis

#image("cum_var.png")

Our results show that 15 components is the lowest value that keeps more than 90% of variance at 91.5%.

After obtaining the probabilities generated by  
We wanted to set a threshold to set 
This tells us that the majority of users are not overlapping. 
We'll set the probability threshold at 40% 

