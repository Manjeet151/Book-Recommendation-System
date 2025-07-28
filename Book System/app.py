from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
app = Flask(__name__)

books = pd.read_csv("Books.csv", encoding="latin1")#latin1 encoding helps your computer understand 
tags = pd.read_csv("Tags.csv", encoding="latin1")#and read those special characters correctly.
book_tags = pd.read_csv("Book_Tags.csv", encoding="latin1")

# Merge datasets
book_tags_merged = pd.merge(book_tags, tags, on="tag_id", how="left")
book_data = pd.merge(books, book_tags_merged, on="book_id", how="left")

# Combine tags for each book and remove duplicates
book_data['combined_tags'] = book_data.groupby('book_id')['tag_name'].transform(lambda x: ' '.join(x))
book_data = book_data.drop_duplicates(subset=['book_id'])

# Create a lowercase title column for case-insensitive matching
book_data['lower_title'] = book_data['title'].str.lower()

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(book_data['combined_tags'].fillna(""))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)




def get_recommendations_by_title(title, cosine_sim=cosine_sim, top_n=5):
    lower_title = title.lower()

    # Use fuzzy matching to find closest title
    all_titles = book_data['lower_title'].tolist()
    closest_title, score, idx = process.extractOne(lower_title, all_titles, score_cutoff=60)

    if closest_title is None:
        return pd.DataFrame()  # no close match found

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # skip the input book itself
    book_indices = [i[0] for i in sim_scores]
    
    return book_data.iloc[book_indices][['title', 'author', 'image-url', 'rating']]


@app.route('/')
def home():
    # Safely get top books based on 'rating'
    try:
        top_books = books.nlargest(50, "rating")[['title', 'author', 'image-url','download-url', 'rating']]
    except KeyError:
        top_books = pd.DataFrame(columns=['title', 'author', 'image-url', 'rating'])

    return render_template('index.html', books=top_books.values)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        user_input = request.form['user_input']
        recommendations = get_recommendations_by_title(user_input)
        
        if not recommendations.empty:
            data = recommendations.values
            message = None
        else:
            data = []
            message = "No recommendations found. Please try another title."

        return render_template('recommend.html', data=data, message=message)

    return render_template('recommend.html')

if __name__ == '__main__':
    app.run(debug=True)
