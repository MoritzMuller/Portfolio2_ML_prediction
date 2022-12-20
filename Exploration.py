# The purpose of the data integration is to create a new dataset that contains consumer ratings and reviews for different brews, combined with their detailed tasting profiles.
# The data integration is done by merging the two datasets on the beer name and brewery name.

import numpy as np
import pandas as pd
from pathlib import Path
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# import beer review data
data = pd.read_csv('data/beer_reviews.csv')
data.head()
data.info()

# We need to aggregate the data because it contains multiple reviews for a single beer. We'll do this getting the average scores of each brew.
df_scores = data.drop(['brewery_id', 'review_time', 'review_profilename',
                      'beer_style', 'beer_abv', 'beer_beerid'], axis=1).groupby(by=['brewery_name', 'beer_name'], as_index=False).mean()

# Get count of reviews per brew
df_scores_count = data.drop(['brewery_id', 'review_time', 'review_profilename', 'beer_style', 'beer_beerid',
                            'beer_abv'], axis=1).groupby(by=['brewery_name', 'beer_name'], as_index=False).count()
df_scores_count = df_scores_count['review_overall']

# Combine score count and score data
df_scores['number_of_reviews'] = df_scores_count
df_scores.head()
df_scores.info()

# import beer profile data
beer_profiles = pd.read_csv('data/beer_data_set.csv')
beer_profiles.info()
beer_profiles.head()

# Our goal is to match as many brews from beer_profiles with the available beer ratings data from df_scores.
# To create a unique identifier for every single brew on both data frames, we need to concatenate the brewery name and the beer name. For instance, the Amber beer by Alaskan Brewing Co. on the first row of df_profile above, will be identified as Alaskan Brewing Co. Amber.
# Before we proceed with that, however, there are several checks we need to perform to make sure that the concatenation will yield unique identifiers that are (somewhat) consistent accross both data frames for optimum results:
# 1.Check rows with missing data on Name column on df_profile.
# 2.Match brewery names from both data frame

# Display `beer_profile` rows with null values (missing beer names) and find fitting replacement names
beer_profiles[beer_profiles['Name'].isnull()]
# Look up data on 'Monastyrskiy Kvas' from `data`
data[data['brewery_name'] == 'Monastyrskiy Kvas']
# Highlight important row
data[data['brewery_name'] == 'Monastyrskiy Kvas'].style.apply(
    lambda x: ['background: lightgreen' if x.name == 40781 else '' for i in x],
    axis=1)
# Look up data on 'Stella Artois' from `data`
data[data['brewery_name'] == 'Stella Artois']
# Highlight important row
data[data['brewery_name'] == 'Stella Artois'].style.apply(
    lambda x: ['background: lightgreen' if x.name == 55651 else '' for i in x],
    axis=1)

# From looking at the tables, we know that the appropriate review data match for each brew lies in the row with highest the number_of_reviews on df_scores (see highlighted rows above). Let's go ahead and apply an identical label format for the two brews on beer_profiles:
# Edit `Name` column for `Stella Artois` and `Monastyrskiy Kvas` on `beer_profiles`
beer_profiles.loc[[1803, 2150],
                  'Name'] = beer_profiles.loc[[1803, 2150], 'Brewery']

# Match brewery names from both data frame
# Create new column on `df_profile` indicating whether the brewery name exists on `df_scores`
beer_profiles['brewery_review_exists'] = np.where(
    beer_profiles['Brewery'].isin(list(df_scores['brewery_name'].unique())), 1, 0)


# Formatting for better display
slice_ = 'brewery_review_exists'


def highlight_indicator(val):
    pink = 'background-color: pink' if val < 1 else ''
    return pink


beer_profiles.head(10).style.set_properties(**{'background-color': '#ffffb3'}, subset=slice_)\
    .applymap(highlight_indicator, subset=[slice_])

# Create new data frame (`brewery_no_scores`) listing breweries on `df_profile` with no exact match on `df_scores`
brewery_no_scores = pd.DataFrame(
    beer_profiles[beer_profiles['brewery_review_exists'] == 0]['Brewery'].unique()).set_axis(['Brewery'], axis=1)
brewery_no_scores

# We want to check if some of the breweries on the previous list actually have corresponding data on df_scores without having 
# the exact same brewery name labels (caused by typos, character errors, inconsistent labelling, etc.). This is where the Python library FuzzyWuzzy comes in handy. 
# We will create a new function to find "fuzzy" matches for the remaining 543 breweries on df_profile.

def fuzzy_merge(df_1, df_2, key1, key2, threshold=90, limit=1):
    """
    :param df_1: the left table to join
    :param df_2: the right table to join
    :param key1: key column of the left table
    :param key2: key column of the right table
    :param threshold: how close the matches should be to return a match, based on Levenshtein distance
    :param limit: the amount of matches that will get returned, these are sorted high to low
    :return: dataframe with boths keys and matches
    """
    s = df_2[key2].tolist()
    
    m = df_1[key1].apply(lambda x: process.extract(x, s, limit=limit))    
    df_1['matches'] = m
    
    m2 = df_1['matches'].apply(lambda x: ', '.join([i[0] for i in x if i[1] >= threshold]))
    df_1['matches'] = m2
    
    return df_1

# Create new column on `df_scores` indicating whether the brewery name exists on `df_profile`
df_scores['brewery_profile_exists'] = np.where(df_scores['brewery_name'].isin(list(beer_profiles['Brewery'].unique())), 1, 0)

# Create new data frame (`brewery_no_profile`) listing breweries on `df_scores` with no exact match on `df_profile`
brewery_no_profile = pd.DataFrame(df_scores[df_scores['brewery_profile_exists']==0]['brewery_name'].unique()).set_axis(['Brewery'], axis=1)

# # (Uncomment to let pandas display all rows and column content for all data frames)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# # (Default setting)
# pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_colwidth', 50)

#Get fuzzy matches for 543 breweries
#Warning: Takes a long time to run
fuzzy_match_brewery = fuzzy_merge(brewery_no_scores, brewery_no_profile, key1='Brewery',key2='Brewery', threshold=88, limit=1)
fuzzy_match_brewery.to_csv('data/Brewery Fuzzy Match List.csv', index=False)

#manually check the results and save them to a new csv file "Brewery Names Fuzzy Match List.csv" (87 proper matches)

# Load table containing saved brewery name matches ("Brewery Name Fuzzy Match List.csv")
fuzzy_match_brewery = pd.read_csv("data/Brewery Name Fuzzy Match List.csv")
fuzzy_match_brewery.info()

# Save table as dictionary
fuzzy_match_dict = dict(zip(fuzzy_match_brewery['Brewery'], fuzzy_match_brewery['matches']))

# Replace relevant brewery names in `df_profile`
beer_profiles['Brewery'] = beer_profiles['Brewery'].replace(fuzzy_match_dict)

# Additionally, manually replace "Yuengling Brewery " with "Yuengling Brewery" (no space) in both `df_profile` and `df_scores`
beer_profiles['Brewery'] = beer_profiles['Brewery'].replace({'Yuengling Brewery ': 'Yuengling Brewery'})
df_scores['brewery_name'] = df_scores['brewery_name'].replace({'Yuengling Brewery ': 'Yuengling Brewery'})

# Correcting 'brewery_review_exists' indicator column values on `df_profile` after using fuzzy matches
beer_profiles['brewery_review_exists'] = np.where(beer_profiles['Brewery'].isin(list(df_scores['brewery_name'].unique())), 1, 0)

# Correcting 'brewery_profile_exists' indicator column values on `df_scores` after using fuzzy matches
df_scores['brewery_profile_exists'] = np.where(df_scores['brewery_name'].isin(list(beer_profiles['Brewery'].unique())), 1, 0)

# remove all rows from `beer_profiles` with no match on `df_scores` (and vice-versa)
# Remove all brews from breweries that are not listed in either `df_profile` or `df_scores`
df_scores.drop(df_scores[df_scores['brewery_profile_exists']==0].index, axis=0, inplace=True)
beer_profiles.drop(beer_profiles[beer_profiles['brewery_review_exists']==0].index, axis=0, inplace=True)

##Match beer names
# Create new identifier column in `df_scores` called 'beer_name_full'
# Fill column by concatenating `brewery_name` and `beer_name`
df_scores['beer_name_full'] = df_scores['brewery_name'] + ' ' + df_scores['beer_name']

# Check if all generated brew names in df_scores are unique:
df_scores['beer_name_full'].nunique() == len(df_scores.index)

# Create new identifier column in `df_profile` called 'Beer Name (Full)' 
# Fill column by concatenating `Brewery` and `Name`
beer_profiles['Beer Name (Full)'] = beer_profiles['Brewery'] + ' ' + beer_profiles['Name']

# Check if all generated brew names in `df_profile` are unique:
beer_profiles['Beer Name (Full)'].nunique() == len(beer_profiles.index)
# Check duplicated brew name
beer_profiles[beer_profiles['Beer Name (Full)'].duplicated()]

# List rows with duplicated brew name
beer_profiles[beer_profiles['Beer Name (Full)']=='Sweetwater Tavern & Brewery Crazy Jackass Ale']


# Highlight row with error
beer_profiles[beer_profiles['Beer Name (Full)']=='Sweetwater Tavern & Brewery Crazy Jackass Ale'].style.apply(
    lambda x: ['background: pink' if x.name == 4056 else '' for i in x], 
    axis=1)

# Remove duplicate row containing error (missing data)
beer_profiles.drop(4056, inplace=True)

#Let's match!
# Create new column on `df_profile` indicating whether the complete brew name exists on `df_scores`
beer_profiles['beer_review_exists'] = np.where(beer_profiles['Beer Name (Full)'].isin(list(df_scores['beer_name_full'])), 1, 0)


# Formatting for better display
slice_ = 'beer_review_exists'

beer_profiles.head(10).style.set_properties(**{'background-color': '#ffffb3'}, subset=slice_)\
                         .applymap(highlight_indicator, subset=[slice_])        
                         
# Create new data frame (`beer_no_scores`) listing brews on `df_profile` with no exact match on `df_scores`
beer_no_scores = pd.DataFrame(beer_profiles[beer_profiles['beer_review_exists']==0]['Beer Name (Full)']).set_axis(['Beer Name (Full)'], axis=1)
beer_no_scores

# Create new column on `df_scores` indicating whether the complete brew name exists on `df_profile`
df_scores['beer_profile_exists'] = np.where(df_scores['beer_name_full'].isin(list(beer_profiles['Beer Name (Full)'])), 1, 0)

# Create new data frame (`beer_no_profile`) listing brews on `df_scores` with no exact match on `df_profile`
beer_no_profile = pd.DataFrame(df_scores[df_scores['beer_profile_exists']==0]['beer_name_full']).set_axis(['beer_name_full'], axis=1)

# Get fuzzy matches for 2460 brews
# Warning: Takes a long time to run
#fuzzy_match_beer_name = fuzzy_merge(beer_no_scores, beer_no_profile, 'Beer Name (Full)', 'beer_name_full', threshold=87, limit=1)
#fuzzy_match_beer_name.to_csv('data/Beer Name Fuzzy Match List.csv', index=False)

#after manual check, we find 1088 valid matches
fuzzy_match_beer_name = pd.read_csv('data/Beer Name Fuzzy Match List.csv')
fuzzy_match_beer_name

# Save table as dictionary
fuzzy_match_dict = dict(zip(fuzzy_match_beer_name['Beer Name (Full)'], fuzzy_match_beer_name['matches']))

# Replace relevant brewery names in `df_profile`
beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace(fuzzy_match_dict)

# Additionally, manually replace some beer names (containing spacing errors) in both `df_profile` and `df_scores`
beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Panil  Barriquée (Sour Version)': 'Panil Barriquée (Sour Version)'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Panil Panil Barriquée (Sour Version)': 'Panil Barriquée (Sour Version)'})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Panil  Barriquée (Italy Version)': 'Panil Barriquée (Italy Version)'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Panil Panil Barriquée  (Italy Version)': 'Panil Barriquée (Italy Version)'})
                                                                                                                                   
beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Unibroue  17 Grande Réserve': 'Unibroue 17 Grande Réserve'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Unibroue Unibroue 17 Grande Réserve': 'Unibroue 17 Grande Réserve'})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Schorschbräu  Schorschbock 57%': 'Schorschbräu Schorschbock 57%'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Schorschbräu Schorschbräu Schorschbock 57%': 'Schorschbräu Schorschbock 57%'})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({"Alexander Keith's  India Pale Ale": "Alexander Keith's India Pale Ale"})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({"Alexander Keith's Alexander Keith's India Pale Ale": "Alexander Keith's India Pale Ale"})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Schorschbräu  Schorschbock 31%': 'Schorschbräu Schorschbock 31%'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Schorschbräu Schorschbräu Schorschbock 31%': 'Schorschbräu Schorschbock 31%'})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Klosterbrauerei Andechs Andechser Dunkles Weissbier': 'Klosterbrauerei Andechs Andechs Weissbier Dunkel'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Klosterbrauerei Andechs Andechser  Dunkles Weissbier': 'Klosterbrauerei Andechs Andechs Weissbier Dunkel'})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({"St. Georgenbräu St. Georgenbräu Keller Bier": "St. Georgenbräu Buttenheimer Keller Bier"})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({"St. Georgenbräu St. Georgenbräu  Keller Bier": "St. Georgenbräu Buttenheimer Keller Bier"})

beer_profiles['Beer Name (Full)'] = beer_profiles['Beer Name (Full)'].replace({'Weisses Bräuhaus G. Schneider & Sohn GmbH Schneider Weisse Mein Alkoholfreies': 'Weisses Bräuhaus G. Schneider & Sohn GmbH Schneider Weisse Mein Alkoholfreies'})
df_scores['beer_name_full'] = df_scores['beer_name_full'].replace({'Weisses Bräuhaus G. Schneider & Sohn GmbH Schneider Weisse  Mein Alkoholfreies': 'Weisses Bräuhaus G. Schneider & Sohn GmbH Schneider Weisse Mein Alkoholfreies'})

# Correcting 'beer_review_exist' indicator column values on `df_profile` after using fuzzy matches
beer_profiles['beer_review_exists'] = np.where(beer_profiles['Beer Name (Full)'].isin(list(df_scores['beer_name_full'].unique())), 1, 0)

# Correcting 'beer_profile_exists' indicator column values on `df_scores` after using fuzzy matches
df_scores['beer_profile_exists'] = np.where(df_scores['beer_name_full'].isin(list(beer_profiles['Beer Name (Full)'].unique())), 1, 0)

#Remove all beers with incomplete data
df_scores.drop(df_scores[df_scores['beer_profile_exists']==0].index, axis=0, inplace=True)
beer_profiles.drop(beer_profiles[beer_profiles['beer_review_exists']==0].index, axis=0, inplace=True)

len(df_scores.index)==len(beer_profiles.index) # Perfect, same length!

## Joining the two dataframes
# Drop columns that are redundant or no longer neccessary
beer_profiles.drop(['key', 'Style Key', 'brewery_review_exists', 'beer_review_exists', 'Ave Rating'], axis=1, inplace=True)
df_scores.drop(['brewery_name', 'beer_name', 'brewery_profile_exists', 'beer_profile_exists'], axis=1, inplace=True)


df_scores.rename(columns={"beer_name_full": "Beer Name (Full)"}, inplace=True)

# Join data frames to make new dataset
df_final = pd.merge(beer_profiles, df_scores, how='left', on=['Beer Name (Full)'])

# Re-arrange column order
df_final = df_final[['Name', 'Style', 'Brewery', 'Beer Name (Full)', 
                     'Description', 'ABV', 'Min IBU', 'Max IBU', 
                     'Astringency', 'Body', 'Alcohol', 
                     'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty', 
                     'review_aroma','review_appearance', 'review_palate', 'review_taste', 'number_of_reviews', 'review_overall']]

# Resulting dataset:
df_final.info()

#save final dataset
df_final.to_csv('data/beer_profile_and_ratings.csv', index=False)