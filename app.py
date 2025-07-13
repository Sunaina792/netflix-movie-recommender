# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import plotly.express as px
# import plotly.graph_objects as go

# # Page configuration
# st.set_page_config(
#     page_title="Netflix Movie Recommender",
#     page_icon="🎬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         color: #E50914;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .movie-card {
#         background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin: 1rem 0;
#         border-left: 4px solid #E50914;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     }
#     .movie-title {
#         color: #E50914;
#         font-size: 1.5rem;
#         font-weight: bold;
#         margin-bottom: 0.5rem;
#     }
#     .movie-info {
#         color: #ffffff;
#         margin: 0.3rem 0;
#     }
#     .rating-badge {
#         background: #E50914;
#         color: white;
#         padding: 0.2rem 0.6rem;
#         border-radius: 15px;
#         font-size: 0.9rem;
#         font-weight: bold;
#     }
#     .genre-tag {
#         background: #2d2d2d;
#         color: #ffffff;
#         padding: 0.2rem 0.5rem;
#         border-radius: 12px;
#         font-size: 0.8rem;
#         margin: 0.1rem;
#         display: inline-block;
#         border: 1px solid #E50914;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load dataset
# @st.cache_data
# def load_netflix_data():
#     try:
#         df = pd.read_csv("netflix_cleaned.csv")

#         # Map your actual columns
#         df['title'] = df['Title']
#         df['genre'] = df['Genre']
#         df['description'] = df['Overview']
#         df['rating'] = pd.to_numeric(df['Vote_Average'], errors='coerce')

#         # Extract year from Release_Date
#         df['year'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.year

#         # Handle missing values
#         df['rating'] = df['rating'].fillna(0)
#         df['description'] = df['description'].fillna("No description available")
#         df['genre'] = df['genre'].fillna("Unknown")
#         # df['year'] = df['year'].fillna(df['year'].median())

#         return df

#     except FileNotFoundError:
#         st.error("❌ File 'netflix_cleaned.csv' not found.")
#         return None
#     except Exception as e:
#         st.error(f"❌ Error loading dataset: {str(e)}")
#         return None
    
# df = load_netflix_data()

# if df is None:
#     st.stop()
# else:
#     st.success("✅ Dataset loaded successfully.")

# # Header
# st.markdown('<h1 class="main-header">🎬 Netflix Movie Recommender</h1>', unsafe_allow_html=True)

# # Sidebar
# st.sidebar.header("🎯 Filters & Search")

# search_query = st.sidebar.text_input("🔍 Search by Title")

# # Genre filter
# if 'genre' in df.columns:
#     genres = []
#     for genre_list in df['genre'].dropna():
#         genres.extend([g.strip() for g in str(genre_list).split(',')])
#     unique_genres = sorted(set(genres))
# else:
#     unique_genres = []

# selected_genre = st.sidebar.selectbox("🎭 Genre", ["All"] + unique_genres)

# # # Year filter
# # if 'year' in df.columns:
# #     min_year = int(df['year'].min())
# #     max_year = int(df['year'].max())
# #     if min_year < max_year:
# #         selected_year = st.sidebar.slider("📅 Year", min_year, max_year, (min_year, max_year))
# #     else:
# #         selected_year = (min_year, max_year)
# #         st.sidebar.info(f"Only one year available: {min_year}")

# # else:
# #     selected_year = None

# # Rating filter
# if 'rating' in df.columns:
#     min_rating = float(df['rating'].min())
#     max_rating = float(df['rating'].max())
#     selected_rating = st.sidebar.slider("⭐ Min Rating", min_rating, max_rating, min_rating, step=0.1)
# else:
#     selected_rating = None

# # Filter the dataframe
# filtered_df = df.copy()

# if search_query:
#     filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]

# if selected_genre != "All":
#     filtered_df = filtered_df[filtered_df['genre'].str.contains(selected_genre, case=False, na=False)]

# # if selected_year:
# #     filtered_df = filtered_df[
# #         (filtered_df['year'] >= selected_year[0]) & 
# #         (filtered_df['year'] <= selected_year[1])
# #     ]

# if selected_rating is not None:
#     filtered_df = filtered_df[filtered_df['rating'] >= selected_rating]

# # Recommendations
# def get_recommendations(title, df, top_n=5):
#     df['features'] = df['genre'] + ' ' + df['description']
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(df['features'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
#     if title not in df['title'].values:
#         return df.head(top_n)

#     idx = df[df['title'] == title].index[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     movie_indices = [i[0] for i in sim_scores[1:top_n+1]]

#     return df.iloc[movie_indices]

# # Layout
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.header("📚 Movie Catalog")
#     if not filtered_df.empty:
#         for _, movie in filtered_df.iterrows():
#             title = movie.get('title', 'Unknown')
#             rating = movie.get('rating', 'N/A')
#             # year = movie.get('year', 'Unknown')
#             desc = movie.get('description', 'No description')
#             genres = movie.get('genre', '')

#             st.markdown(f"""
#             <div class="movie-card">
#                 <div class="movie-title">{title}</div>
#                 <div class="movie-info">
#                 <div class="movie-info">{desc[:150]}...</div>
#                 <div style="margin-top: 0.5rem;">
#                     {''.join([f'<span class="genre-tag">{g}</span>' for g in str(genres).split(',')])}
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.warning("No movies found for selected filters.")

# with col2:
#     st.header("🎯 Recommendations")
#     selected_movie = st.selectbox("Pick a movie", df['title'].dropna().unique().tolist())

#     if st.button("Recommend"):
#         recs = get_recommendations(selected_movie, df)
#         st.subheader(f"Movies similar to '{selected_movie}':")
#         for _, rec in recs.iterrows():
#             st.markdown(f"""
#             <div style="background: #2d2d2d; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #E50914;">
#                 <strong style="color: #E50914;">{rec['title']}</strong><br>
#                 <small style="color: #cccccc;">{rec['year']} • ⭐ {rec['rating']}</small><br>
#                 <small style="color: #ffffff;">{rec['description'][:100]}...</small>
#             </div>
#             """, unsafe_allow_html=True)

#     st.header("📊 Stats")
#     if 'rating' in df.columns:
#         st.plotly_chart(px.histogram(df, x='rating', title='Rating Distribution'), use_container_width=True)

#     if 'year' in df.columns:
#         st.plotly_chart(px.histogram(df, x='year', title='Release Year Distribution'), use_container_width=True)

#     if unique_genres:
#         genre_counts = pd.Series([g for sublist in df['genre'].dropna().str.split(',') for g in sublist]).value_counts().head(8)
#         st.plotly_chart(px.bar(x=genre_counts.index, y=genre_counts.values, title='Top Genres'), use_container_width=True)

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center; color: #666666; padding: 1rem;'>
#         Netflix Movie Recommender System • Built with Streamlit<br>
#         Using TF-IDF + Cosine Similarity for content-based recommendations
#     </div>
#     """, 
#     unsafe_allow_html=True
# )

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #E50914;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .movie-title {
        color: #E50914;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .movie-info {
        color: #ffffff;
        margin: 0.3rem 0;
    }
    .genre-tag {
        background: #2d2d2d;
        color: #ffffff;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.1rem;
        display: inline-block;
        border: 1px solid #E50914;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_cleaned.csv")

    df['title'] = df['Title']
    df['genre'] = df['Genre'].fillna("Unknown")
    df['description'] = df['Overview'].fillna("No description available")
    df['rating'] = pd.to_numeric(df['Vote_Average'], errors='coerce').fillna(0)
    df['year'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.year

    return df

df = load_data()

# ---------- Header ----------
st.markdown('<h1 class="main-header">🎬 Netflix Movie Recommender</h1>', unsafe_allow_html=True)

# ---------- Sidebar Filters ----------
st.sidebar.header("🎯 Filters & Search")
search_query = st.sidebar.text_input("🔍 Search by Title")

# Genre
genres = []
for g in df['genre'].dropna():
    genres.extend([x.strip() for x in str(g).split(',')])
unique_genres = sorted(set(genres))
selected_genre = st.sidebar.selectbox("🎭 Genre", ["All"] + unique_genres)

# Rating
min_rating = float(df['rating'].min())
max_rating = float(df['rating'].max())
selected_rating = st.sidebar.slider("⭐ Min Rating", min_rating, max_rating, min_rating, step=0.1)

# ---------- Filtered Data ----------
filtered_df = df.copy()

if search_query:
    filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]

if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['genre'].str.contains(selected_genre, case=False, na=False)]

filtered_df = filtered_df[filtered_df['rating'] >= selected_rating]

# ---------- Recommendation Function ----------
def get_recommendations(title, df, top_n=5):
    df['features'] = df['genre'] + ' ' + df['description']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    if title not in df['title'].values:
        return df.head(top_n)

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
    return df.iloc[movie_indices]

# ---------- Layout ----------
# No columns now — just a single full-width container
st.header("🎯 Recommendations")
selected_movie = st.selectbox("Pick a movie", df['title'].dropna().unique().tolist())

if st.button("Recommend"):
    recs = get_recommendations(selected_movie, df)
    st.subheader(f"Movies similar to '{selected_movie}':")
    for _, rec in recs.iterrows():
        st.markdown(f"""
        <div style="background: #2d2d2d; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 3px solid #E50914;">
            <strong style="color: #E50914;">{rec['title']}</strong><br>
            <small style="color: #cccccc;">{rec['year']} • ⭐ {rec['rating']}</small><br>
            <small style="color: #ffffff;">{rec['description'][:100]}...</small>
        </div>
        """, unsafe_allow_html=True)


# # ---------- Movie Catalog ----------
# with col1:
#     st.header("📚 Movie Catalog")
#     if not filtered_df.empty:
#         for _, movie in filtered_df.iterrows():
#             st.markdown(f"""
#             <div class="movie-card">
#                 <div class="movie-title">{movie['title']}</div>
#                 <div class="movie-info">{movie['description'][:150]}...</div>
#                 <div style="margin-top: 0.5rem;">
#                     {''.join([f'<span class="genre-tag">{g.strip()}</span>' for g in str(movie['genre']).split(',')])}
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
#     else:
#         st.warning("No movies found for selected filters.")



# ---------- Charts ----------
    st.header("📊 Stats")

    st.plotly_chart(px.histogram(df, x='rating', title='Rating Distribution'), use_container_width=True)

    st.plotly_chart(px.histogram(df, x='year', title='Release Year Distribution'), use_container_width=True)

    genre_counts = pd.Series([g.strip() for sublist in df['genre'].dropna().str.split(',') for g in sublist]).value_counts().head(8)
    st.plotly_chart(px.bar(x=genre_counts.index, y=genre_counts.values, title='Top Genres'), use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666; padding: 1rem;'>
    Netflix Movie Recommender System • Built with Streamlit<br>
    Using TF-IDF + Cosine Similarity for content-based recommendations
</div>
""", unsafe_allow_html=True)
