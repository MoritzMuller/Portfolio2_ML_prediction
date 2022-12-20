'''
Visualize and explore the beer dataset that we have created by merging beer reviews and beer details
'''
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import the dataset
data = pd.read_csv('data/beer_profile_and_ratings.csv')

## Manipulate the data
# check the data
data.head()
data.describe()

# drop all beers with less than 5 reviews
data = data[data['number_of_reviews'] >= 5]

## Check the distribution of observations across beer styles and breweries
# what is the distribution of number of beers across beer Style?
plt.figure(figsize=(30, 20))
sns.countplot(y='Style', data=data, order=data.Style.value_counts().index)
plt.title('Number of Beers across Beer Styles')
plt.show()  # linear distribution

# what is the distribution of number of beers across Breweries?
plt.figure(figsize=(50, 30))
sns.countplot(y='Brewery', data=data, order=data.Brewery.value_counts().index)
plt.title('Number of Beers across Breweries')
plt.show()  # log distribution

# what is the distribution of overall reviews?
plt.figure(figsize=(20, 10))
sns.displot(data['review_overall'], kde=False)
plt.title('Distribution of Overall Ratings')
plt.show()  # people are very critical! The highest rating in the data is a beer with 4.8/5

## Let's explore the data a little bit more

# there are multiple rating categories. Can we rely on the overall rating, or do ratings differ in the various categories?
# plot the distribution of ratings across the various categories
plt.figure(figsize=(20, 10))
sns.displot(data['review_aroma'], kde=False)
sns.displot(data['review_appearance'], kde=False)
sns.displot(data['review_palate'], kde=False)
sns.displot(data['review_taste'], kde=False)
sns.displot(data['review_overall'], kde=False)
plt.title('Distribution of Ratings across Categories')
plt.show()  # the distributions are very similar, so we should be able to rely on the overall rating. To be sure, lets plot the correlations

# plot the correlation matrix for all columns between review_aroma and review_overall to see whether the ratings are linearly correlated
plt.figure(figsize=(20, 10))
sns.heatmap(data.loc[:, ['review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'review_overall']].corr(), annot=True)
plt.title('Correlation between Ratings') # very high correlations across the bank, so we can disregard the other rating categories

# how much alcohol is in the beers?
plt.figure(figsize=(20, 10))
sns.displot(data['ABV'], kde=False)
plt.title('Distribution of Alcohol Content')
plt.show()

# just out of curiosity, what are the 5 highest rated beers?
data.sort_values(by='review_overall', ascending=False).head(5)

# what are the 10 beer Styles with the highest average ratings (put the rank below each label)?
plt.figure(figsize=(30, 20))
sns.barplot(x='review_overall', y='Style', data=data, order=data.groupby(
    'Style')['review_overall'].mean().sort_values(ascending=False).index[:10])
plt.title('Top 10 Beer Styles with Highest Average Ratings')
plt.show()


## Correlation analysis of the various taste notes
# plot correlation matrix for all columns between Astringency amd Malty and review_overall to see whether the ratings are linearly correlated with specific tast notes
plt.figure(figsize=(20, 10))
sns.heatmap(data.iloc[:, [8,
                          9,
                          10,
                          11,
                          12,
                          13,
                          14,
                          15,
                          16,
                          17,
                          18, 24]].corr(), annot=True)
plt.title('Correlation between taset notes and overall rating')
plt.show()  # it does not seem that there are specific taste notes that reviewers consistently relate to higher ratings

# How do individual taste notes correlate to the overall rating?
#plot the the datapoints for Astringency against review_overall to see whether there is a linear relationship
plt.figure(figsize=(20, 10))
sns.scatterplot(x='review_overall', y='Astringency', data=data)
plt.title('Astringency vs Review Overall')
plt.show()  # there is no linear relationship between Astringency and Review Overall, but good ratings are clustered around a certain range of astringency and the tail end of the distribution tends to correspond to higher values of this taste note

# Do the same for Sourness
plt.figure(figsize=(20, 10))
sns.scatterplot(x='review_overall', y='Sour', data=data)
plt.title('Sourness vs Review Overall')
plt.show() # similar behavior as Astringency


## PCA
## conduct a principal component analysis to check whether taste profile combinations can be simplified into fewer dimensions (e.g., maybe the combination of sourness and maltyness scores predicts an underlying taste)
X = data.iloc[:, [8,
                  9,
                  10,
                  11,
                  12,
                  13,
                  14,
                  15,
                  16,
                  17,
                  18]]
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)  # set the PCA to retain 95% of the variance
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3',
                           'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8'])

finalDf = pd.concat([principalDf, data[['Style']]], axis=1)

# Visualize the PCA
fig = px.scatter_3d(finalDf, x='principal component 1', y='principal component 2', z='principal component 3',
                    color='Style', opacity=0.7)
fig.show()  # it does not seem like there are relevant underlying dimensions that can be used to simplify the taste profile


## Cosine similarity
## Check which beer styles are most similar to each other
X = data.iloc[:, [8,
                  9,
                  10,
                  11,
                  12,
                  13,
                  14,
                  15,
                  16,
                  17,
                  18]]
cosine_sim = cosine_similarity(X, X)
indices = pd.Series(data.index, index=data['Name']).drop_duplicates()
# Visualize the similarity matrix
fig = px.imshow(cosine_sim)
fig.show()

# based on the cosine similarity, we can now write a function that finds the 10 most similar beer styles to a beer of choice
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    beer_indices = [i[0] for i in sim_scores]
    return data.iloc[beer_indices]
# Lets try which beers are most similar to "Doppelsticke", a beer that I know and like (the brewery is in my homecity)
get_recommendations(title='Doppelsticke') #good recommendations. The beers all share a similar taste profile. 

# Lets recap: 
# 1. We have a dataset of 2914 beers with 24 features, of which 11 are taste notes
# 2. The data is linearly distributed across beer styles and log distributed across breweries
# 3. The ratings are linearly correlated across the bank, so we can disregard the other rating categories
# 4. The PCA shoes that while there are some underlying dimensions in the taset profiles, they are not meaningfully related to the beer styles or taste notes.
# 5. The cosine similarity shows that there are meaningful similarities between beer styles, which can be used to recommend similar beers to a beer of choice.
# 6. There is a pattern in how the various taste notes are distributed across the ratings, but it is not linearly correlated with the overall rating.
# This makes it likely that a combination of taset profiles is related to better ratings. This could be a good starting point to predict the overall rating based on beer flavour profiles!
        
