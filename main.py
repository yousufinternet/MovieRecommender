#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# read imdb datasets
title_basics = pd.read_csv(
    'IMDBDataSets/title.basics.tsv', sep='\t', na_values='\\N')

# a small fix in title basics dataframe
nonnull_runtimeminutes = title_basics.runtimeMinutes[title_basics.runtimeMinutes.notna()]
text_indexes = nonnull_runtimeminutes[
    nonnull_runtimeminutes.astype(str).str.match(r'^\D.*')].index
title_basics.loc[text_indexes, 'genres'] = title_basics.loc[text_indexes, 'runtimeMinutes'].tolist()
title_basics.loc[text_indexes, 'runtimeMinutes'] = np.nan

title_ratings = pd.read_csv(
    'IMDBDataSets/title.ratings.tsv', sep='\t', na_values='\\N')

# add ratings to title basics
title_basics = title_basics.join(title_ratings.set_index('tconst'), on='tconst')

# fix data types

print(title_basics.columns)
print(title_basics[title_basics.averageRating.notna() & title_basics.numVotes.notna()].shape)

fil_title_basics = title_basics[['tconst', 'primaryTitle', 'originalTitle', 'isAdult', 'startYear', 'genres', 'averageRating', 'numVotes']].dropna(how='any')

print(fil_title_basics.shape)
