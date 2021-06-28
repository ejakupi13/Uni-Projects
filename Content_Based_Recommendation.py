# content-based recommender system

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from itertools import chain, repeat

chainer = chain.from_iterable


# ---------------------- START functions ----------------------
# Function that takes in movie title as input and outputs most similar movies
# def get_recommendations(prod, cosine_sim):

#     # Get the pairwise similarity scores of games
#     sim_scores = list(enumerate(cosine_sim[prod]))

#     # Sort games based on the similarity scores
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the scores of the 10 most similar games
#     sim_scores = sim_scores[1:11]

#     # Get the game indices
#     movie_indices = [i[0] for i in sim_scores]

#     # Return the top 10 most similar movies
#     return games_data.iloc[movie_indices]


def create_pivot_table(pd_df):
    data = pd_df.values
    rows, row_pos = np.unique(data[:, 4], return_inverse=True)  # col 4: internal user id
    cols, col_pos = np.unique(data[:, 1], return_inverse=True)  # col 1: real product id
    pivot_matrix = np.zeros((len(rows), len(cols)), dtype=data.dtype)
    pivot_matrix[:] = np.nan
    pivot_matrix[row_pos, col_pos] = data[:, 2]
    return pivot_matrix


def recommend_movies(pred_df, userID, games, original_ratings, num_recommendations):
    # get internal id of user
    uid_internal = map_uid.at[userID, 'index']

    sorted_user_predictions = pred_df.iloc[uid_internal].sort_values(ascending=False)

    # get user's rated games
    user_data = original_ratings[original_ratings.user_id == int(userID)]

    # optional: merge game info
    # user_full = (user_data.merge(games, how='left', left_on='product_id', right_on='product_id').
    #                 sort_values(['rating'], ascending=False)
    #             )

    # Recommend the highest predicted rating games that the user hasn't seen yet.
    recommendations = games_data[~games_data['product_id'].isin(user_data['product_id'])].merge(
        pd.DataFrame(sorted_user_predictions).rename_axis('product_id').reset_index(), how='left',
        on='product_id').rename(columns={uid_internal: 'pred'}).sort_values('pred', ascending=False).iloc[
                      :num_recommendations, :]

    return recommendations


# ---------------------- END functions ----------------------

data = pd.read_csv("/Users/danielkhan/Google Drive/Web Mining Project/Dataset/filter_for_algorithms.csv",
                   usecols=['product_id', 'user_id', 'title_x', 'publisher', 'genres', 'tags', 'specs', 'bin'])
data['product_id'] = data['product_id'].astype(int)  # product_id is initially a float with .0s
print("Total users: " + str(data['user_id'].nunique()))
print("Total products: " + str(data['product_id'].nunique()))

# dataframe storing games info
games_data = data[['product_id', 'title_x', 'publisher', 'genres']].copy()
games_data.fillna("", inplace=True)
games_data.drop_duplicates(inplace=True)
games_data.set_index(pd.Series(np.arange(len(games_data))), inplace=True)
games_data['metadata'] = games_data[['publisher', 'genres']].apply(lambda x: ' '.join(x), axis=1)

# produce tfidf matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(games_data['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=games_data.index.tolist())

# compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# recommendations = get_recommendations(1, cosine_sim)
# print(recommendations)

# create internal product id
map_pid = games_data.reset_index()
map_pid = map_pid[['index', 'product_id']].copy()

# create internal user id
map_uid = data[['user_id']].copy().drop_duplicates(keep='first')
map_uid.set_index(pd.Series(np.arange(len(map_uid))), inplace=True)
map_uid.reset_index(inplace=True)
map_uid.set_index('user_id', inplace=True)

user_item = data[['user_id', 'product_id', 'bin']].copy()
user_item = user_item.merge(map_pid, on='product_id', how='left')
user_item = user_item.merge(map_uid, on='user_id', how='left')

# user_item_matrix = user_item[['user_id','index','bin']].pivot(index='user_id', columns='index', values = 'bin').fillna(0)
user_item_matrix2 = create_pivot_table(user_item)

# pred = user_item_matrix2.dot(cosine_sim) / np.array([np.abs(cosine_sim).sum(axis=1)])

mean_item_rating = np.nanmean(user_item_matrix2, axis=0)
ratings_diff = (user_item_matrix2 - mean_item_rating[np.newaxis, :])

array = np.zeros((len(map_uid), len(map_pid)))
index = 0
for r1 in ratings_diff:
    if index %1000 == 0:
        print(index)
    sum_rated = np.nansum(np.where(np.isnan(r1), r1, cosine_sim), axis=1)
    dp = np.where(np.isnan(r1), 0.0, r1).dot(np.where(np.isnan(cosine_sim), 0.0, cosine_sim))
    dp = (np.nan_to_num(r1)).dot(cosine_sim)
    prod = np.nan_to_num(dp / sum_rated)
    array[index] = prod
    index = index + 1

pred = mean_item_rating[np.newaxis, :] + array

# pred = mean_item_rating[np.newaxis, :] + ratings_diff.dot(cosine_sim) / np.array([np.abs(cosine_sim).sum(axis=1)])
# print(pred)
content_prediction = pred.clip(min=0)

content_pred_df = pd.DataFrame(content_prediction, columns=list(map_pid['product_id']))

# dict in the format {uid: [prod1, prod2, ..., prodN]}, stores all user's recommendations
results = []

# iterate over all users
for index, user in enumerate(data['user_id'].unique().tolist()):
    if index % 1000 == 0:
        print(index)
    recommendations = recommend_movies(content_pred_df, user, games_data, user_item, 10)
    prod_rating = [(user, i, j) for i, j in
                   zip(recommendations['product_id'].tolist(), recommendations['pred'].tolist())]
    results.extend(prod_rating)

# df_results = pd.DataFrame({'user_id': list(chainer(repeat(k, len(v)) for k, v in results.items())), 'product_ratings': list(chainer(results.values()))})
df_results = pd.DataFrame(results, columns=['user_id', 'product_id', 'rating'])
# df_results['product_ratings'].str[1:-1].str.split(',', expand=True).astype(float)

# output results
df_results.to_csv("cb5000.csv", index=False)

print(df_results.head(100))

# test recommendations for one user
recommendations = recommend_movies(content_pred_df, 76561198022842797, games_data, user_item, 10)
print(recommendations)
