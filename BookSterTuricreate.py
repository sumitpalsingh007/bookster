import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../BookSter"))

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
books_cols = ['id','book_id','best_book_id','work_id','books_count','isbn','isbn13','authors','original_publication_year','original_title','title','language_code','average_rating','ratings_count','work_ratings_count','work_text_reviews_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url']
books = pd.read_csv('../BookSter/books.csv', encoding = "ISO-8859-1")
print(books.head)
print(books.shape)
print(books.columns)

ratings = pd.read_csv('../BookSter/ratings.csv', encoding = "ISO-8859-1")
print("rating shape",ratings.shape)

book_tags = pd.read_csv('../BookSter/book_tags.csv', encoding = "ISO-8859-1")
print(book_tags.head())

tags = pd.read_csv('../BookSter/tags.csv')
print(tags.tail())

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

# using turicreate for the above code
import turicreate as tc
train_data = tc.SFrame(ratings)

popularity_model = tc.popularity_recommender.create(train_data, user_id='user_id', item_id='book_id', target='rating')

popularity_recomm = popularity_model.recommend(users=[1,2,3,4,5],k=5)
popularity_recomm.print_rows(num_rows=25)

#Training the model
item_sim_model = tc.item_similarity_recommender.create(train_data, user_id='user_id', item_id='book_id', target='rating', similarity_type='cosine')

#Making recommendations
item_sim_recomm = item_sim_model.recommend(users=ratings.user_id.unique(),k=5)
item_sim_recomm.print_rows(num_rows=100)