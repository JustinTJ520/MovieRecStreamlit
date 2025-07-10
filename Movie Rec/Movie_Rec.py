#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st



def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title


def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    movies["similarity"] = similarity

    movies["title_match"] = movies["clean_title"].apply(lambda x: 1 if title in x else 0)
    movies["final_score"] = movies["similarity"] + (movies["title_match"] * 0.5)

    results = movies.sort_values(by="final_score", ascending=False).head(5)
    return results[["movieId", "title", "genres", "clean_title", "final_score"]]


def find_similar_movies(movie_id):
    movie_genres = movies[movies["movieId"] == movie_id]["genres"].values[0]
    
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] >= 4)]["userId"].unique()

    if len(similar_users) == 0:
        print("No users found who liked this movie highly. Falling back to content-based recommendations.")
    
        recommendations = movies[movies["genres"].str.contains(movie_genres.split("|")[0], na=False)].sample(10)
        recommendations["score"] = 0.5
        return recommendations[["score", "title", "genres"]]
    
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] >= 3.5)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]

    if len(similar_user_recs) == 0:
        print("No movies passed the 5% similarity threshold. Falling back to content-based recommendations.")
        
        recommendations = movies[movies["genres"].str.contains(movie_genres.split("|")[0], na=False)].sample(10, random_state=42)
        recommendations["score"] = 0.5
        return recommendations[["score", "title", "genres"]]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 3.5)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    recommendations = rec_percentages.merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
    recommendations = recommendations[recommendations["genres"].str.contains(movie_genres.split("|")[0], na=False)]

    if recommendations.empty:
        print("No movies matched genre filtering. Falling back to content-based recommendations.")
        
        recommendations = movies[movies["genres"].str.contains(movie_genres.split("|")[0], na=False)].sample(10, random_state=42)
        recommendations["score"] = 0.5
        return recommendations[["score", "title", "genres"]]
    
    return recommendations.sort_values("score", ascending=False).head(10)


movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

movie_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

movies = movies.merge(movie_tags, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")

movies["clean_title"] = movies["title"].apply(clean_title)

movies["combined_features"] = movies["clean_title"] + " " + movies["genres"] + " " + movies["tag"]

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english")

tfidf = vectorizer.fit_transform(movies["combined_features"])

average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
average_ratings.rename(columns = {"rating": "avg_rating"}, inplace = True)
movies = movies.merge(average_ratings, on = "movieId", how = "left")
movies["avg_rating"] = movies["avg_rating"].fillna(0)

st.title("Movie Recommendation System")

st.header("Search for a Movie")
search_title = st.text_input("Enter movie title:")
if st.button("Search"):
    results = search(search_title)
    st.write(results)

st.header("Get Recommendations")
title = st.text_input("Enter a movie title for recommendations:")
if st.button("Get Recommendations"):
    results = search(title)
    if not results.empty:
        movie_id = results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)
        st.write(recommendations)
    else:
        st.write("No movie found!")
