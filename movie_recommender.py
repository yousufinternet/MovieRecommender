#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def get_title_from_index(idx, movies_df):
    return movies_df.loc[idx, ['primaryTitle', 'startYear']]


def get_user_movie(movies_df):
    while True:
        user_movie = input('Please enter a movie name you like:')
        sim_movies = movies_df[
            movies_df.primaryTitle.str.contains(user_movie, case=False)]
        if len(sim_movies) == 0:
            print('No movies with this title found')
            continue
        elif len(sim_movies) == 1:
            user_movie = sim_movies
        else:
            print('Please choose from matchings below:')
            print("\n".join(f'{i}. {title} ({startYear})'
                            for i, (title, startYear)
                            in enumerate(sim_movies[['primaryTitle', 'startYear']].itertuples(False, None))))
            while True:
                user_input = input('Enter movie number: ')
                if user_input.isnumeric() and int(user_input) < min(len(sim_movies), 10):
                    user_movie = sim_movies.iloc[int(user_input)]
                    break
                else:
                    print('incorrect input!')
                    continue
            break
    return user_movie


with open('Filtered_Titles', 'rb') as f:
    fil_title_basics = pickle.load(f)
fil_title_basics.index = range(len(fil_title_basics))

user_movie = get_user_movie(fil_title_basics)

important_features = [
    'primaryTitle', 'genres', 'startYear', 'directors', 'writers']


def combine_features(row):
    """
    combine a pandas series to be a single string
    """
    return " ".join(str(row[feature]) for feature in important_features)


fil_title_basics['CombinedFeatures'] = fil_title_basics.apply(combine_features, axis=1)

cv = CountVectorizer()
count_matrix = cv.fit_transform(fil_title_basics.CombinedFeatures)

print(count_matrix)
print(fil_title_basics)

print(fil_title_basics.shape)
print(count_matrix.shape)
print(user_movie.name)
print(count_matrix[user_movie.name])

cosine_sim = cosine_similarity(count_matrix, count_matrix[user_movie.name])

cosine_sim_idx = list(enumerate(cosine_sim))
best_20 = sorted(cosine_sim_idx, key=lambda x: x[1], reverse=True)[1:20]

for i, movie in enumerate(best_20):
    movie_title = get_title_from_index(movie[0], fil_title_basics)
    print(f'{i:0>2f}.{movie_title.primaryTitle} ({movie_title.startYear}). SCORE: {movie[1]}')
