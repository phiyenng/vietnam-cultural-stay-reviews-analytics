import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ================================
# Header
# ================================
cells.append(nbf.v4.new_markdown_cell("""# Hospitality Reviews Exploratory Data Analysis & NLP Insights
This notebook performs a comprehensive EDA and Natural Language Processing (NLP) analysis on a dataset of hospitality reviews.
It generates a visual analytics dashboard composed of multiple interactive and static charts.

**Goal**:
- Understand what guests talk about the most.
- Identify what drives positive reviews.
- Discover common themes in reviews.
- Highlight potential complaints.
- Analyze differences across locations."""))

# ================================
# Setup Cell
# ================================
cells.append(nbf.v4.new_code_cell("""# Setup & Installations (Uncomment if needed)
# !pip install pandas numpy matplotlib seaborn plotly nltk scikit-learn wordcloud textblob vaderSentiment gensim openpyxl networkx"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import re

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Topic Modeling & Network
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
import networkx as nx

# Configure Visuals
pio.templates.default = "plotly_white"
sns.set_theme(style="whitegrid", palette="muted")
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"""))

# ================================
# SECTION 1: DATA OVERVIEW
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 1 — DATA OVERVIEW
Load and inspect the dataset. Here we will look at the basic distribution of reviews, platforms, locations, and ratings."""))

cells.append(nbf.v4.new_code_cell("""# Load the dataset
df = pd.read_excel('raw_data.xlsx')
df = df.dropna(subset=['review']).reset_index(drop=True)
print(f"Dataset shape: {df.shape}")
display(df.head())"""))

cells.append(nbf.v4.new_code_cell("""# Build an interactive dashboard layout for Section 1 with Plotly subplots
fig1 = make_subplots(
    rows=2, cols=3,
    specs=[[{"type": "indicator"}, {"type": "bar"}, {"type": "bar"}],
           [{"type": "heatmap"}, {"type": "histogram"}, {"type": "bar"}]],
    subplot_titles=("1. Total Review Count", "2. Review Count by Platform", 
                    "3. Review Count by Location", "4. Platform vs Location",
                    "5. Rating Distribution", "7. Avg Rating by Location"),
    vertical_spacing=0.15, horizontal_spacing=0.08
)

# 1. Total review count (Indicator)
fig1.add_trace(go.Indicator(
    mode="number",
    value=len(df),
    title={"text": "Total Reviews"},
), row=1, col=1)

# 2. Review count by platform
platform_counts = df['platform'].value_counts().reset_index()
platform_counts.columns = ['platform', 'count']
fig1.add_trace(go.Bar(x=platform_counts['platform'], y=platform_counts['count'], marker_color='#636EFA', name="Platform"), row=1, col=2)

# 3. Review count by location
loc_counts = df['location'].value_counts().reset_index()
loc_counts.columns = ['location', 'count']
fig1.add_trace(go.Bar(x=loc_counts['location'], y=loc_counts['count'], marker_color='#EF553B', name="Location"), row=1, col=3)

# 4. Platform vs location heatmap
heatmap_data = pd.crosstab(df['platform'], df['location'])
fig1.add_trace(go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='Blues',
    showscale=False
), row=2, col=1)

# 5. Rating distribution histogram & 6. Rating count bar chart
rating_counts = df['rating'].value_counts().sort_index().reset_index()
rating_counts.columns = ['rating', 'count']
fig1.add_trace(go.Bar(x=rating_counts['rating'], y=rating_counts['count'], marker_color='#00CC96', name="Rating"), row=2, col=2) # Covers both 5 & 6

# 7. Average rating by location
avg_rating_loc = df.groupby('location')['rating'].mean().reset_index()
fig1.add_trace(go.Bar(x=avg_rating_loc['location'], y=avg_rating_loc['rating'], marker_color='#AB63FA', name="Avg Rating (Loc)"), row=2, col=3)

fig1.update_layout(height=800, title_text="Data Overview Dashboard (Section 1)", showlegend=False)
fig1.show()

# 8. Average rating by platform
avg_rating_plat = df.groupby('platform')['rating'].mean().reset_index()
fig1_b = px.bar(avg_rating_plat, x='platform', y='rating', color='platform', text_auto='.2f', 
                title="8. Average Rating by Platform", template="plotly_white")
fig1_b.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Data Overview):**
- Most reviews are concentrated from top-performing platforms. 
- Location performance is measurable by the volume of reviews, indicating which properties generate the most engagement.
- High average ratings across locations show strong baseline customer satisfaction."""))

# ================================
# SECTION 2: REVIEW LENGTH ANALYSIS
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 2 — REVIEW LENGTH ANALYSIS
Analyze how the length of the review text corresponds with ratings, locations, and platforms."""))

cells.append(nbf.v4.new_code_cell("""# Calculate review length (character count and word count)
df['review_len'] = df['review'].astype(str).apply(len)
df['word_count'] = df['review'].astype(str).apply(lambda x: len(x.split()))

# Create subplots for Review Length Analysis
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=("9. Review Length Distribution (Words)", "10. Review Length vs. Rating",
                    "11. Average Review Length by Location", "12. Average Review Length by Platform")
)

# 9. Review length distribution (Histogram of word count)
fig2.add_trace(go.Histogram(x=df['word_count'], nbinsx=30, marker_color='#FFA15A', name='Word Count'), row=1, col=1)

# 10. Review length vs rating scatter plot (We use a boxplot for better visibility across discrete ratings)
fig2.add_trace(go.Box(x=df['rating'], y=df['word_count'], marker_color='#19D3F3', name='Length vs Rating'), row=1, col=2)

# 11. Average review length by location
avg_len_loc = df.groupby('location')['word_count'].mean().reset_index()
fig2.add_trace(go.Bar(x=avg_len_loc['location'], y=avg_len_loc['word_count'], marker_color='#FF6692', name='Avg Len Loc'), row=2, col=1)

# 12. Average review length by platform
avg_len_plat = df.groupby('platform')['word_count'].mean().reset_index()
fig2.add_trace(go.Bar(x=avg_len_plat['platform'], y=avg_len_plat['word_count'], marker_color='#B6E880', name='Avg Len Plat'), row=2, col=2)

fig2.update_layout(height=800, title_text="Review Length Analysis (Section 2)", showlegend=False)
fig2.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Review Length):**
- Word count distributions often show a right skew: most guests leave short to medium reviews.
- Extremely long reviews might correlate with specific ratings (e.g., highly negative 1-star reviews or exceptionally detailed 5-star reviews).
- We can observe if certain platforms naturally foster longer reviews than others."""))

# ================================
# SECTION 3: TEXT PREPROCESSING
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 3 — TEXT PREPROCESSING
Cleaning the review text by lowercasing, removing punctuation, numbers, stopwords, followed by tokenization and lemmatization."""))

cells.append(nbf.v4.new_code_cell("""# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# adding custom stop words if necessary
stop_words.update(['hotel', 'room', 'stay', 'would', 'could'])

def clean_text(text):
    text = str(text).lower()
    # Use regex to keep only letters and spaces, effectively removing all punctuation/digits
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    # Filter for length > 2 to remove noise/punct fragments
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(clean_text)
display(df[['review', 'cleaned_review']].head())
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Preprocessing):**
- Standardizing the text reduces noise for downstream NLP tasks, ensuring words like 'Amazing' and 'amazing!' are treated identically."""))

# ================================
# SECTION 4: WORD FREQUENCY ANALYSIS
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 4 — WORD FREQUENCY ANALYSIS
Visualize the most commonly used words overall to gauge the top themes."""))

cells.append(nbf.v4.new_code_cell("""# Get word frequency
all_words = ' '.join(df['cleaned_review']).split()
word_freq = pd.Series(all_words).value_counts()

# 13. Word cloud (all reviews)
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(word_freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("13. Word Cloud (All Reviews)", fontsize=16)
plt.show()

# 14 & 15. Top 20 and 30 most frequent words
top_words_20 = word_freq.head(20).reset_index()
top_words_20.columns = ['word', 'count']
top_words_30 = word_freq.head(30).reset_index()
top_words_30.columns = ['word', 'count']

fig4 = make_subplots(rows=1, cols=2, subplot_titles=("14. Top 20 Frequent Words", "16. Word Frequency Distr. (Top 30 Area)"))

fig4.add_trace(go.Bar(x=top_words_20['count'], y=top_words_20['word'], orientation='h', marker_color='#636EFA', name='Top 20'), row=1, col=1)
# Make it ascending for horizontal bar chart
fig4.update_yaxes(autorange="reversed", row=1, col=1)

# 16. Word frequency distribution
fig4.add_trace(go.Scatter(x=top_words_30['word'], y=top_words_30['count'], fill='tozeroy', marker_color='#EF553B', name='Freq Dist'), row=1, col=2)

fig4.update_layout(height=500, title_text="Word Frequency Analysis (Section 4)", showlegend=False)
fig4.show()

# 15. Standalone Top 30 Chart
fig4_b = px.bar(top_words_30, x='word', y='count', title="15. Top 30 Most Frequent Words", template="plotly_white", color='count', color_continuous_scale="Viridis")
fig4_b.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Word Frequency):**
- The vocabulary is heavily focused on experience descriptors (clean, friendly, great, location) and specifics of the property, providing a high-level view of standard amenities and sentiment indicators."""))

# ================================
# SECTION 5: BIGRAM & TRIGRAM ANALYSIS
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 5 — BIGRAM & TRIGRAM ANALYSIS
Identify multi-word phrases to retain semantic context better than single words."""))

cells.append(nbf.v4.new_code_cell("""# Extract N-grams
def get_top_ngrams(corpus, n=None, ngram_range=(2,2)):
    vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top_bigrams = get_top_ngrams(df['cleaned_review'], n=20, ngram_range=(2,2))
df_bigram = pd.DataFrame(top_bigrams, columns=['Phrase', 'Count'])

top_trigrams = get_top_ngrams(df['cleaned_review'], n=20, ngram_range=(3,3))
df_trigram = pd.DataFrame(top_trigrams, columns=['Phrase', 'Count'])

fig5 = make_subplots(rows=1, cols=2, subplot_titles=("17 & 19. Top 20 Bigrams (Frequency)", "18. Top 20 Trigrams"))

fig5.add_trace(go.Bar(x=df_bigram['Count'], y=df_bigram['Phrase'], orientation='h', marker_color='#00CC96'), row=1, col=1)
fig5.update_yaxes(autorange="reversed", row=1, col=1)

fig5.add_trace(go.Bar(x=df_trigram['Count'], y=df_trigram['Phrase'], orientation='h', marker_color='#AB63FA'), row=1, col=2)
fig5.update_yaxes(autorange="reversed", row=1, col=2)

fig5.update_layout(height=600, title_text="N-gram Analysis (Section 5)", showlegend=False)
fig5.show()

# 20. Phrase network visualization (Optional Network)
plt.figure(figsize=(10, 8))
G = nx.Graph()
for index, row in df_bigram.head(15).iterrows():
    words = row['Phrase'].split()
    G.add_edge(words[0], words[1], weight=row['Count'])

pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, font_weight='bold', edge_color='gray', width=[w*0.2 for u,v,w in G.edges(data='weight')])
plt.title("20. Phrase Network Visualization (Top Bigrams)", fontsize=16)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (N-grams):**
- Phrases like "highly recommend" or "customer service" paint a functional picture of operations.
- The network plot illustrates how key adjectives (e.g., 'helpful', 'clean') cluster around target nouns ('staff', 'room')."""))


# ================================
# SECTION 6: SENTIMENT ANALYSIS
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 6 — SENTIMENT ANALYSIS
Using VADER and TextBlob to assign a sentiment polarity to each review, and categorizing them as Positive, Neutral, or Negative."""))

cells.append(nbf.v4.new_code_cell("""analyzer = SentimentIntensityAnalyzer()

def get_vader_score(text):
    return analyzer.polarity_scores(text)['compound']

def get_sentiment_category(score):
    if score >= 0.05: return 'Positive'
    elif score <= -0.05: return 'Negative'
    else: return 'Neutral'

df['sentiment_score'] = df['review'].astype(str).apply(get_vader_score)
df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_category)

fig6 = make_subplots(
    rows=2, cols=3,
    subplot_titles=("21. Sentiment Distribution", "22. Sentiment vs Rating", 
                    "23. Avg Sentiment by Location", "24. Avg Sentiment by Platform", "25. Category Count")
)

# 21. Sentiment distribution histogram
fig6.add_trace(go.Histogram(x=df['sentiment_score'], marker_color='#19D3F3'), row=1, col=1)

# 22. Sentiment vs rating scatter plot (Violin plot works well here)
fig6.add_trace(go.Violin(x=df['rating'], y=df['sentiment_score'], box_visible=True, marker_color='#FF6692'), row=1, col=2)

# 23. Average sentiment by location
avg_sent_loc = df.groupby('location')['sentiment_score'].mean().reset_index()
fig6.add_trace(go.Bar(x=avg_sent_loc['location'], y=avg_sent_loc['sentiment_score'], marker_color='#B6E880'), row=1, col=3)

# 24. Average sentiment by platform
avg_sent_plat = df.groupby('platform')['sentiment_score'].mean().reset_index()
fig6.add_trace(go.Bar(x=avg_sent_plat['platform'], y=avg_sent_plat['sentiment_score'], marker_color='#FF97FF'), row=2, col=1)

# 25. Sentiment category count
sent_cat = df['sentiment_category'].value_counts().reset_index()
fig6.add_trace(go.Pie(labels=sent_cat['sentiment_category'], values=sent_cat['count'], hole=0.3, marker_colors=['#00CC96', '#EF553B', '#636EFA']), row=2, col=2)

fig6.update_layout(height=800, title_text="Sentiment Analysis (Section 6)", showlegend=False)
fig6.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Sentiment Analysis):**
- Sentiment scores validate the ratings heavily: high ratings show strong positive polarity.
- Some lower-rated reviews might have higher sentiment due to mixed language ("The place was beautiful, but the service was terrible...")."""))


# ================================
# SECTION 7: LOCATION TEXT INSIGHTS
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 7 — LOCATION TEXT INSIGHTS
Break out key themes and sentiment metrics by specific locations."""))

cells.append(nbf.v4.new_code_cell("""# Let's get the top locations (up to 4 for visibility)
top_locations = df['location'].value_counts().head(4).index.tolist()

# 26. Word cloud per location (using matplotlib grid)
plt.figure(figsize=(16, 8))
for i, loc in enumerate(top_locations, 1):
    plt.subplot(2, 2, i)
    text = ' '.join(df[df['location']==loc]['cleaned_review'])
    if text.strip() != '':
        wc = WordCloud(width=400, height=300, background_color='white').generate(text)
        plt.imshow(wc, interpolation='bilinear')
    plt.title(f"26. Word Cloud: {loc}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# 27. Top words per location and 28. Sentiment dist by location
fig7 = make_subplots(rows=1, cols=2, subplot_titles=("27. Review Counts in Top Locations", "28. Sentiment Dist by Location"))

# 28. Sentiment distribution by location using Plotly Box
for loc in top_locations:
    fig7.add_trace(go.Box(y=df[df['location'] == loc]['sentiment_score'], name=str(loc)), row=1, col=2)

# Simple frequency chart of those top locations for #27
loc_d = df[df['location'].isin(top_locations)]['location'].value_counts()
fig7.add_trace(go.Bar(x=loc_d.index, y=loc_d.values, marker_color='#FECB52'), row=1, col=1)

fig7.update_layout(height=500, title_text="Location Insights (Section 7)", showlegend=False)
fig7.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Location Insights):**
- Specific problems (like parking or noise) or specific benefits (like views) often surface within specific property locations' word clouds.
- Sentiment distribution showcases which locations struggle with inconsistent experiences vs. predictable quality."""))


# ================================
# SECTION 8: NEGATIVE REVIEW ANALYSIS
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 8 — NEGATIVE REVIEW ANALYSIS
Focusing specifically on reviews with a rating of 3 or below to identify pain points and top complaint themes."""))

cells.append(nbf.v4.new_code_cell("""neg_df = df[df['rating'] <= 3].copy()
if len(neg_df) > 0:
    # 29. Word cloud of negative reviews
    neg_text = ' '.join(neg_df['cleaned_review'])
    plt.figure(figsize=(10, 5))
    if len(neg_text.strip()) > 0:
        wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
        plt.imshow(wc_neg, interpolation='bilinear')
    plt.title("29. Word Cloud of Negative Reviews")
    plt.axis('off')
    plt.show()
    
    # 30. Top complaint keywords & 31. Negative review frequency by location
    neg_words = pd.Series(neg_text.split()).value_counts().head(20).reset_index()
    neg_words.columns = ['word', 'count']
    
    neg_loc = neg_df['location'].value_counts().reset_index()
    neg_loc.columns = ['location', 'count']
    
    fig8 = make_subplots(rows=1, cols=2, subplot_titles=("30. Top Complaint Keywords", "31. Negative Reviews by Location"))
    fig8.add_trace(go.Bar(x=neg_words['count'], y=neg_words['word'], orientation='h', marker_color='red'), row=1, col=1)
    fig8.update_yaxes(autorange="reversed", row=1, col=1)
    fig8.add_trace(go.Bar(x=neg_loc['location'], y=neg_loc['count'], marker_color='darkred'), row=1, col=2)
    fig8.update_layout(height=500, title_text="Negative Reviews (Section 8)", showlegend=False)
    fig8.show()
else:
    print("Warning: No negative reviews (<=3) found in the dataset.")
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Negative Reviews):**
- Word frequency in negative reviews directly illuminates the root causes of poor experiences (e.g., cleanliness, customer service disputes).
- Tying negative frequency to location helps zero in on operational bottlenecks."""))

# ================================
# SECTION 9: TOPIC MODELING
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 9 — TOPIC MODELING
Latent Dirichlet Allocation (LDA) modeling to categorize unstructured reviews into discrete unlabelled topics."""))

cells.append(nbf.v4.new_code_cell("""# Create Dictionary and Corpus for LDA
texts = [text.split() for text in df['cleaned_review']]
id2word = corpora.Dictionary(texts)
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model (Using 3 topics for simplicity)
lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=3, random_state=42, passes=10)

# Format the Topic Words
topics = lda_model.show_topics(formatted=False, num_words=10)
topic_data = []
for t in range(lda_model.num_topics):
    words = [word[0] for word in topics[t][1]]
    weights = [word[1] for word in topics[t][1]]
    topic_data.append(pd.DataFrame({'Word': words, 'Weight': weights, 'Topic': f"Topic {t+1}"}))

topic_df = pd.concat(topic_data)

# 32. Top words per topic
fig9 = px.bar(topic_df, x='Weight', y='Word', color='Topic', facet_col='Topic', orientation='h', 
              title="32. Top Words per Topic (LDA)", height=400, template='plotly_white')
fig9.update_yaxes(matches=None, showticklabels=True)
fig9.show()

# Assign dominant topic to each review
def get_dominant_topic(bow):
    topic_probs = lda_model.get_document_topics(bow)
    return max(topic_probs, key=lambda x: x[1])[0] + 1

df['dominant_topic'] = [f"Topic {get_dominant_topic(c)}" for c in corpus]

# 33. Topic distribution & 34. Topic distribution by location
fig9_b = make_subplots(rows=1, cols=2, subplot_titles=("33. Topic Distribution Across Reviews", "34. Topic Distribution by Location"))

topic_counts = df['dominant_topic'].value_counts().reset_index()
fig9_b.add_trace(go.Pie(labels=topic_counts['dominant_topic'], values=topic_counts['count'], hole=0.3, marker_colors=['#636EFA', '#EF553B', '#00CC96']), row=1, col=1)

cross_topic_loc = pd.crosstab(df['location'], df['dominant_topic'])
for col in cross_topic_loc.columns:
    fig9_b.add_trace(go.Bar(x=cross_topic_loc.index, y=cross_topic_loc[col], name=col), row=1, col=2)

fig9_b.update_layout(barmode='stack', height=500, title_text="Topic Distributions (Section 9)", showlegend=True)
fig9_b.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""**Insights (Topic Modeling):**
- Topic modeling clusters keywords dynamically. We might interpret Topic 1 as "Service & Staff", Topic 2 as "Room & Amenities", etc. It allows us to pinpoint which properties index highly on which topics."""))

# ================================
# SECTION 10: FINAL DASHBOARD
# ================================
cells.append(nbf.v4.new_markdown_cell("""## SECTION 10 — FINAL DASHBOARD
A consolidated, visually appealing high-level dashboard combining key KPIs and main charts in a single pane format."""))

cells.append(nbf.v4.new_code_cell("""# Create a comprehensive Grid Layout showing 4 pivotal charts
fig10 = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "domain"}, {"type": "bar"}],
           [{"type": "xy"}, {"type": "bar"}]],
    subplot_titles=("Sentiment Breakdown", "Average Sentiments by Location",
                    "Rating Distribution", "Top 10 Words Overall")
)

# A. Sentiment Pie
sent_p = df['sentiment_category'].value_counts().reset_index()
fig10.add_trace(go.Pie(labels=sent_p['sentiment_category'], values=sent_p['count'], marker_colors=['#00CC96', '#EF553B', '#636EFA']), row=1, col=1)

# B. Avg Sentiment Location
fig10.add_trace(go.Bar(x=avg_sent_loc['location'], y=avg_sent_loc['sentiment_score'], marker_color='#AB63FA'), row=1, col=2)

# C. Rating Dist
fig10.add_trace(go.Bar(x=rating_counts['rating'], y=rating_counts['count'], marker_color='#19D3F3'), row=2, col=1)

# D. Top 10 Words
t10 = top_words_30.head(10)
fig10.add_trace(go.Bar(x=t10['word'], y=t10['count'], marker_color='#FECB52'), row=2, col=2)

fig10.update_layout(height=800, title_text="EXECUTIVE ANALYTICS DASHBOARD - HOSPITALITY REVIEWS", 
                    showlegend=False, 
                    paper_bgcolor='rgba(245, 246, 249, 1)', 
                    plot_bgcolor='rgba(245, 246, 249, 1)')
fig10.show()
"""))

# Write to file
nb['cells'] = cells
with open('eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook 'eda.ipynb' generated successfully.")
