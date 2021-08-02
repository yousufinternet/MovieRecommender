#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

title_basics = pd.read_csv(
    'IMDBDataSets/title.basics.tsv', sep='\t', na_values='\\N')

title_ratings = pd.read_csv(
    'IMDBDataSets/title.ratings.tsv', sep='\t', na_values='\\N')


nonnull_runtimeminutes = title_basics.runtimeMinutes[title_basics.runtimeMinutes.notna()]
text_indexes = nonnull_runtimeminutes[
    nonnull_runtimeminutes.astype(str).str.match(r'^\D.*')].index

title_basics.loc[text_indexes, 'genres'] = title_basics.loc[
    text_indexes, 'runtimeMinutes'].tolist()
title_basics.loc[text_indexes, 'runtimeMinutes'] = np.nan

ignored_cols = ['titleType', 'isAdult', 'endYear', 'runtimeMinutes', 'numVotes']
fil_title_basics = title_basics[
    title_basics.titleType.isin(['movie', 'tvMovie'])][
        title_basics.isAdult == 0][
            [c for c in title_basics.columns if c not in ignored_cols]]

fil_title_basics = fil_title_basics.join(title_ratings.set_index('tconst'), on='tconst')

fil_title_basics.dropna(how='any', inplace=True)

fil_title_basics = fil_title_basics[(fil_title_basics.averageRating > 4) & (fil_title_basics.numVotes > 1000)]

name_basics = pd.read_csv(
    'IMDBDataSets/name.basics.tsv',
    sep='\t', na_values='\\n',
    usecols=['nconst', 'primaryName']).set_index('nconst')

title_crew = pd.read_csv(
      'IMDBDataSets/title.crew.tsv', sep='\t', na_values='\\N')


def get_names(nconsts):
    '''
    given imdb name keys, return actual names
    I have merged first and middle names on purpose to increase matching accuracy
    '''
    names_lst = []
    for nconst in nconsts.split(','):
        try:
            names_lst.append(''.join(name_basics.loc[nconst, 'primaryName'].split()))
        except KeyError:
            continue
    if names_lst:
        return ' '.join(names_lst)
    else:
        return np.nan


title_crew.dropna(how='any', inplace=True)
title_crew[~title_crew.directors.astype(str).str.isnumeric()]
title_crew[~title_crew.writers.astype(str).str.isnumeric()]

title_crew.directors = title_crew.directors.apply(get_names)
title_crew.writers = title_crew.writers.apply(get_names)
title_crew.dropna(how='any', inplace=True)

fil_title_basics = fil_title_basics.join(title_crew.set_index('tconst'), on='tconst')
fil_title_basics.startYear = fil_title_basics.startYear.apply(int)
fil_title_basics.dropna(how='any', inplace=True)

fil_title_basics.genres = fil_title_basics.genres.apply(lambda x: ' '.join(x.split(',')))

movies_overview = pd.read_csv('IMDBDataSets/movies_metadata.csv', usecols=['imdb_id', 'original_language', 'overview']).set_index('imdb_id').dropna(how='any')

fil_title_basics = fil_title_basics.join(movies_overview, on='tconst')

with open('FilteredTitles_Overview', 'wb') as f:
    pickle.dump(fil_title_basics, f)

