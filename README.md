# Predicting beer reviews based on taste notes

Why do we like certain beers and not others? What makes a beer good an what makes a beer bad? And most importantly, can we predict how much we will like a beer based on its characteristics? These are the questions we will try to answer in this notebook. We will work with two very exciting datasets that contain taste information about a wide range of craft beers and their reviews on the popular website BeerAdvocate.com.

The first dataset contains ~1.5 million reviews of beers from the website BeerAdvocate.com . The second dataset contains taset information from the reviews on BeerAdvocate.com that contain the flavour profiles of the beers (retrieved here: https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information). Using fuzzy matching (shown in the script "01_FuzzyMatching.py"), we have matched the two datasets and can now work with a complete dataset that contains taset information of over 3000 beers and their (aggregated) review scores.

Ultimately, we will predict the review score of a beer based on its taste characteristics. We will use a variety of machine learning algorithms to do so, including linear regression, random forest, gradient boosting, and neural networks (RNN). To do so, we will first explore the data to better understand how taste notes relate to beer review scores. We will then use the taste notes to predict the review score of a beer. After that, we will try to use hyperparameter tuning to improve the performance of the model that we have chosen.

Files:
01_ -> Fuzzy matching of beer taste profiles and BeerAdvocate reviews (python file)
02_ -> Data visualization and exploration (Jupyter Notebook)
03_ -> Fitting a suitable linear regressor to the data to predict reviews (Jupyter Notebook)
