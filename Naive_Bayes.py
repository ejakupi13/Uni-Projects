## Load movies data
import pandas as pd
df = pd.read_csv("/Users/danielkhan/PycharmProjects/WM/Codes/train_with_9_bins.csv")
df.head()

## links.csv is useful when you want to merge movielens dataset with tmdb dataset based on its id
print('Total rows: ', len(df))

print('Unique rows based on user_id: ', len(df.user_id.unique()))
print('Unique rows based on product_id: ', len(df.product_id.unique()))
df.head(5)

# Limit user ratings that have rated more than 30 movies
# If you have more than 20m user ratings, it will be problematic to pivot the rating dataframe later for collaborative filtering
# That is you're userId/movieId pivot table will be super sparse, this is one of the ways to limit the users while not losing too many rated movies
# But there are also other ways to do with sparse matrix (e.g. SVD)

# ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 50 and len(x)<1600)
ratings_f = df.groupby('user_id').filter(lambda x: len(x) >= 10)
ratings_f = ratings_f.groupby('product_id').filter(lambda x: len(x) >= 5)
print(len(ratings_f.groupby(['user_id']).size()))
print(len(ratings_f.groupby(['product_id']).size()))
#ratings_f.groupby(['userId']).size()

# List the movieId after the ratings count filtering
# Preserve 100.00% of the original movies in ratings dataframe
# But after filtering, preserve only  82.13% of the users
game_list_rating = ratings_f.product_id.unique().tolist()
game_filter = df[df.product_id.isin(game_list_rating)]
print('Preserved rate of the Game : {0:.2f} %'.format(len(ratings_f.product_id.unique())/len(game_filter.product_id.unique()) * 100))
print('Preserved rate of the users : {0:.2f} %'.format(len(ratings_f.user_id.unique())/len(df.user_id.unique()) * 100))

# Limit the user rating count which is greater than 0

ratings_f = ratings_f[['product_id', 'user_id', 'bin']].copy()
ratings_f.head()

# Use the function pivot to come up with the matrix (movieId, userId) and the rating
# Be careful for this pivot table, currently since we only have 610 users
# What if the number of users are greater than 200m, it will be super sparse, and will face the memory issue since having limited RAM
ratings_f2 = ratings_f.pivot(index = 'product_id', columns ='user_id', values = 'bin').fillna(0)
#ratings_f2.head(10)

def create_pivot_table(pd_df):
    data = pd_df.values
    #print(data.shape)
    rows, row_pos = np.unique(data[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)
    pivot_matrix = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    pivot_matrix[row_pos, col_pos] = data[:, 2]
    print(pivot_matrix.shape)
    return pivot_matrix

import numpy as np
# ratings_f2: item-user matrix (9724, 610)
ratings_f2 = create_pivot_table(ratings_f)
# ratings_f2.T: user-item matrix (610, 9724)
#print(ratings_f2)

from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(ratings_f2.T, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print('Shape of User Similarity Matrix:', user_correlation.shape)
# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(ratings_f2, metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0
print('Shape of Item Similarity Matrix:', item_correlation.shape)

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        mean_item_rating = ratings.mean(axis=0)
        ratings_diff = (ratings - mean_item_rating[np.newaxis, :])
        pred = mean_item_rating[np.newaxis, :] + ratings_diff.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    elif type == 'content':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred.clip(min=0)

#train_data_matrix  = ratings_f2.T.as_matrix(columns = ['userId', 'movieId', 'rating'])
user_prediction = predict(ratings_f2.T, user_correlation, type='user')
item_prediction = predict(ratings_f2.T, item_correlation, type='item')

user_pred_df = pd.DataFrame(user_prediction, columns = list(game_filter.product_id.unique()))
item_pred_df = pd.DataFrame(item_prediction, columns = list(game_filter.product_id.unique()))


def recommend_movies(pred_df, userID, movies, original_ratings, num_recommendations, method_name):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # User ID starts at 1, not 0
    sorted_user_predictions = pred_df.iloc[user_row_number].sort_values(ascending=False)  # User ID starts at 1

    # Get the user's data and merge in the movie information.
    user_data = original_ratings[original_ratings.user_id == (userID)]
    # user_full = (user_data.merge(movies, how = 'left', left_on = 'product_id', right_on = 'product_id').
    #                sort_values(['bin'], ascending=False)
    #           )

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (df[~df['product_id'].isin(user_data['product_id'])].
                           merge(pd.DataFrame(sorted_user_predictions).rename_axis('product_id').reset_index(),
                                 how='left',
                                 left_on='product_id',
                                 right_on='product_id').
                           rename(columns={user_row_number: method_name}).
                           sort_values(method_name, ascending=False).
                           iloc[:num_recommendations, :]
                           )

    return user_data, recommendations

print('Collaborative Filtering RS (user-based)')
user_data, recommendations  = recommend_movies(user_pred_df, 2, game_filter, ratings_f, len(game_filter), 'CF_user_pred_rating')
recommendations

