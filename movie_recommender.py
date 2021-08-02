#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def get_title_from_index(idx, movies_df):
    return movies_df.loc[idx, ['tconst', 'primaryTitle', 'startYear', 'CombinedFeatures', 'averageRating']]


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
                            in enumerate(sim_movies[['primaryTitle', 'startYear']].iloc[:15].itertuples(False, None))))
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


def get_movie_rating(idx, movies_df):
    return movies_df.loc[idx, ['averageRating', 'numVotes']]


with open('FilteredTitles_Overview', 'rb') as f:
    fil_title_basics = pickle.load(f)

fil_title_basics.dropna(how='any', inplace=True)
fil_title_basics.index = range(len(fil_title_basics))

important_features = [
    'primaryTitle', 'genres', 'startYear', 'directors',
    'writers', 'original_language']


def combine_features(row):
    """
    combine a pandas series to be a single string
    """
    return " ".join(str(row[feature]) for feature in important_features)


fil_title_basics['CombinedFeatures'] = fil_title_basics.apply(combine_features, axis=1)

user_movie = get_user_movie(fil_title_basics)

cv = CountVectorizer()
count_matrix = cv.fit_transform(fil_title_basics.CombinedFeatures)
overview_count_matrix = cv.fit_transform(fil_title_basics.overview)

cosine_sim = cosine_similarity(count_matrix, count_matrix[user_movie.name])
cosine_sim_overview = cosine_similarity(overview_count_matrix, overview_count_matrix[user_movie.name])

combined_cosine_sim = list(zip(cosine_sim, cosine_sim_overview))
cosine_sim_idx = list(enumerate(combined_cosine_sim))
# best_20 = sorted(cosine_sim_idx, key=lambda x: (x[1][0]*2+x[1][1])/3, reverse=True)[1:11]
def scores_sorting_func(x):
    cosine_scores = sum(x[1])
    movie_rating = get_movie_rating(x[0], fil_title_basics)
    corr_movie_rat = movie_rating.averageRating/80
    corr_num_votes = movie_rating.numVotes / (fil_title_basics.numVotes.max()*8)
    final_score = cosine_scores + corr_movie_rat + corr_num_votes
    if final_score > 2:
        print(get_title_from_index(x[0], fil_title_basics))
        print(cosine_scores)
        print(corr_movie_rat, corr_num_votes)
        print(final_score)
    return final_score / 2.25

best_20 = sorted(cosine_sim_idx, key=scores_sorting_func, reverse=True)[1:11]

print(user_movie['CombinedFeatures'])
for i, movie in enumerate(best_20):
    movie_title = get_title_from_index(movie[0], fil_title_basics)
    print(f'{i+1:0>2}.{movie_title.primaryTitle} ({movie_title.startYear}).\n IMDB RATING: {movie_title.averageRating} SCORE: {movie[1]}')
    print(f'https://imdb.com/title/{movie_title.tconst}')
