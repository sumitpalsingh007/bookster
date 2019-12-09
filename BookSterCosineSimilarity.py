import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
books_cols = ['id','book_id','best_book_id','work_id','books_count','isbn','isbn13','authors','original_publication_year','original_title','title','language_code','average_rating','ratings_count','work_ratings_count','work_text_reviews_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url']
books = pd.read_csv('../BookSter/books.csv', encoding = "ISO-8859-1")
print("books shape",books.shape)
print("books head",books.head(5))

ratings = pd.read_csv('../BookSter/ratings.csv', encoding = "ISO-8859-1")
print("rating shape",ratings.shape)
print('rating head',ratings.head(5))

book_tags = pd.read_csv('../BookSter/book_tags.csv', encoding = "ISO-8859-1")
print('books tags shape',book_tags.shape)
print("books tags head",book_tags.head())

tags = pd.read_csv('../BookSter/tags.csv')
#print(tags.tail())

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
tags_join_DF.head()

to_read = pd.read_csv('../BookSter/to_read.csv')
to_read.head()

# Calculating the unique users and ratings
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.book_id.unique().shape[0]
print("unique users : ", n_users)
print("rated books : ", n_items)
print("ratings :", ratings.rating.shape[0])

# Creating user matrix for calculating similarity  between user and item
data_matrix = np.zeros((n_users, n_items))

for line in ratings.itertuples():
    #print(line[1]-1, line[2]-1, line[3])
    data_matrix[line[2]-1, line[1]-1] = line[3]

print("user-item matrix  : ")
print(pd.DataFrame(data_matrix).head(10))
print("size of user item matix : ", data_matrix.shape)

# Calculating cosine similarity using  pairwise distance
from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(data_matrix, metric='cosine', n_jobs=100)
item_similarity = pairwise_distances(data_matrix.T, metric='cosine', n_jobs=100)

print("")
print("size of user_similarity : ", user_similarity.shape)
print("user_similarity : ")
print(pd.DataFrame(user_similarity).head(100))
print("size of item_similarity : ", item_similarity.shape)
print("item_similarity : ")
print(pd.DataFrame(item_similarity).head(100))

# defining prediction function
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# making  predictions
user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')

print("user_prediction shape: ", pd.DataFrame(user_prediction).shape)
print("user_prediction : ")
print(pd.DataFrame(user_prediction).head(100))
print("item_prediction shape: ", pd.DataFrame(item_prediction).shape)
print("item_prediction shape : ")
print(pd.DataFrame(item_prediction).head(100))