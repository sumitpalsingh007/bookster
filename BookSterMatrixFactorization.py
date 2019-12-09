import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../BookSter"))

# Any results you write to the current directory are saved as output.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
books_cols = ['id','book_id','best_book_id','work_id','books_count','isbn','isbn13','authors','original_publication_year','original_title','title','language_code','average_rating','ratings_count','work_ratings_count','work_text_reviews_count','ratings_1','ratings_2','ratings_3','ratings_4','ratings_5','image_url','small_image_url']
books = pd.read_csv('../BookSter/books.csv', encoding = "ISO-8859-1")
#print(books.head)
#print(books.shape)
#print(books.columns)

ratings = pd.read_csv('../BookSter/ratings.csv', encoding = "ISO-8859-1")
#print("rating shape",ratings.shape)
#print("rating head",pd.DataFrame(ratings).head)

book_tags = pd.read_csv('../BookSter/book_tags.csv', encoding = "ISO-8859-1")
#print(book_tags.head())

tags = pd.read_csv('../BookSter/tags.csv')
#print(tags.tail())

tags_join_DF = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
#tags_join_DF.head()

to_read = pd.read_csv('../BookSter/to_read.csv')
#to_read.head()

# Calculating the unique users and ratings
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.book_id.unique().shape[0]
print("unique users : ", n_users)
print("rated books : ", n_items)
print("ratings :", ratings.rating.shape[0])

class MF():

    # Initializing the user-book rating matrix, no. of latent features, alpha and beta.
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    # Initializing user-feature and book-feature matrix
    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # List of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
        self.sgd()
        mse = self.mse()
        training_process.append((i, mse))
        if (i+1) % 20 == 0:
            print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    # Computing total mean squared error
    def mse(self):
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    # Ratings for user i and book j
    def get_rating(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Full user-book rating matrix
    def full_matrix(self):
        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)


R = np.array(ratings.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0))
print("input matrix")
print(R)
#R = np.array(ratings.reset_index().pivot(index=['user_id','book_id'], columns='book_id', values='rating').fillna(0))
mf = MF(R, K=20, alpha=0.001, beta=0.01, iterations=100)
training_process = mf.train()
print()
print("user_id x book_id")
result = mf.full_matrix();
#print(result.size)
print(result)