# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:31:02 2022

@author: hp
"""

import numpy as np
import pandas as pd
import os

os.chdir(r"C:\Users\hp\OneDrive\Desktop\Book Recommendation System")

# Reading Data from CSV File

books = pd.read_csv("Data/Books.csv")
ratings = pd.read_csv("Data/Ratings.csv")
users = pd.read_csv("Data/Users.csv")


# Checking for the null values
books.isnull().sum()
ratings.isnull().sum()
users.isnull().sum()

# Checking for the duplicated values
books.duplicated().sum()
ratings.duplicated().sum()
users.duplicated().sum()

###############################################################################
# Popularity Based Recommendation System
###############################################################################

ratings_with_name = ratings.merge(books, on = 'ISBN')

# for popuarity based Recommendation system we are considering a books which 
# has more than 250 users voting
# We are considering higest 50 books has avg voting more than others

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_rating'}, inplace = True)


avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace = True)

popular_df = num_rating_df.merge(avg_rating_df, on = 'Book-Title')


popular_df = popular_df[popular_df['num_rating']>=250].sort_values('avg_rating',ascending=False).head(50)

# There is a book with same title and different ISBN
# So we are droping such books
popular_df = popular_df.merge(books, on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_rating','avg_rating']]
popular_df['avg_rating'] = popular_df['avg_rating'].round(1)

###############################################################################
# Collaberative Filtering
###############################################################################
# Here we are taking in consideration a books which has more than 50 reviews and 
# Users who voted for atleast 200 books

x = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
knowlegeble_users = x[x].index

# Fetching knowlegeble users from rating_with_name
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(knowlegeble_users)]

y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt = pt.fillna(0)



from sklearn.metrics.pairwise import cosine_similarity
similarity_score = cosine_similarity(pt)


def recommendation(book_name):
    #fetching index
    index = np.where(pt.index == book_name)[0][0]
    distance = similarity_score[index]
    #Enumerating the items so we would not loose the index of the books 
    similar_items = sorted(list(enumerate(distance)), key=lambda x:x[1], reverse=True)[1:6]
    
    data = []
    for i in similar_items:
       # print(pt.index[i[0]]) #pt.index[i[0]] -> to fetching the books name
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data
     

# Hardcoded
# Enter a book name to find similar books 
recommendation("1984")

'''
import pickle
pickle.dump(popular_df, open("popular.pkl", "wb"))
pickle.dump(pt, open("pt.pkl", "wb"))
pickle.dump(books, open("books.pkl", "wb"))
pickle.dump(similarity_score, open("similarity_score.pkl", "wb"))

'''