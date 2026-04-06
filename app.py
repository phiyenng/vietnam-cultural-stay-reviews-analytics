import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# NLP Packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# Topic Modeling
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel

# Page Settings - "Wide" mode makes dashboards pop
st.set_page_config(page_title="Hotel Reviews Analytics", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR BEAUTY ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Ensure NLTK data is downloaded silently
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

@st.cache_data(show_spinner=False)
def load_and_preprocess():
    """Loads dataset and performs standard NLP preprocessing tasks"""
    df = pd.read_excel('raw_data.xlsx')
    df = df.dropna(subset=['review']).reset_index(drop=True)
    
    # Ensure reviews are English
    def is_english(text):
        try:
            return detect(str(text)) == 'en'
        except:
            return False
            
    df['is_en'] = df['review'].apply(is_english)
    df = df[df['is_en']].reset_index(drop=True)
    df.drop(columns=['is_en'], inplace=True)
    
    # Feature Eng
    df['review_len'] = df['review'].astype(str).apply(len)
    df['word_count'] = df['review'].astype(str).apply(lambda x: len(x.split()))
    
    # NLP Preprocessing
    lemmatizer = WordNetLemmatizer()
    
    # Load custom English stopwords from file
    try:
        with open('stopwords-en.txt', 'r', encoding='utf-8') as f:
            custom_stops = set([line.strip() for line in f if line.strip()])
    except:
        custom_stops = set(stopwords.words('english'))
        
    custom_stops.update(['hotel', 'room', 'stay', 'would', 'could', 'us', 'get', 'one'])
    
    def clean_text(text):
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ''.join([i for i in text if not i.isdigit()])
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in custom_stops]
        return ' '.join(tokens)
        
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Sentiment Calculation via VADER
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['review'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda s: 'Positive' if s >= 0.05 else ('Negative' if s <= -0.05 else 'Neutral')
    )
    
    return df

with st.spinner("Loading Analytics Engine & Model Data..."):
    raw_data = load_and_preprocess()

# Sidebar Setup
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Filter the dataset dynamically across the entire application:")

platforms = st.sidebar.multiselect("Select Platforms", options=raw_data['platform'].unique(), default=raw_data['platform'].unique())
locations = st.sidebar.multiselect("Select Locations", options=raw_data['location'].unique(), default=raw_data['location'].unique())


# Filter Data Based on User Input
if not platforms or not locations:
    st.warning("⚠️ Please select at least one platform and location to display data.")
    st.stop()

df = raw_data[(raw_data['platform'].isin(platforms)) & (raw_data['location'].isin(locations))].copy()

# App Header
st.title("Hospitality Analytics")
# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews (Filtered)", f"{len(df):,}")
col2.metric("Average Rating", f"{df['rating'].mean():.2f} ⭐")
col3.metric("Avg Sentiment Score", f"{df['sentiment_score'].mean():.2f}")
col4.metric("Complaint Load (<4 Stars)", f"{len(df[df['rating'] <= 3])}")

st.divider()

# Core Tabs Construction
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Demographics", 
    "Textual NLP Insights", 
    "Sentiment Analysis", 
    "Complaint Radar", 
    "Topic Modeling"
])

# Utility Function for N-Grams
def get_top_ngrams(corpus, n=None, ngram_range=(1,1)):
    try:
        vec = CountVectorizer(ngram_range=ngram_range).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return pd.DataFrame(words_freq[:n], columns=['Phrase', 'Count'])
    except ValueError:
        return pd.DataFrame(columns=['Phrase', 'Count'])

# =============== TAB 1: OVERVIEW ===============
with tab1:
    st.header("Demographics & Base Statistics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        plat_count = df['platform'].value_counts().reset_index()
        fig_plat = px.pie(plat_count, values='count', names='platform', hole=0.4, 
                          title="Reviews Volume by Platform")
        st.plotly_chart(fig_plat, width='stretch')
        
    with col_b:
        loc_count = df['location'].value_counts().reset_index()
        fig_loc = px.bar(loc_count, x='location', y='count', title="Reviews Volume by Location", 
                         color='location', text_auto=True)
        fig_loc.update_traces(textposition="outside")
        st.plotly_chart(fig_loc, width='stretch')
        
    col_c, col_d = st.columns(2)
    with col_c:
        rat_count = df['rating'].value_counts().reset_index().sort_values('rating')
        fig_rat = px.bar(rat_count, x='rating', y='count', title="Overall Rating Distribution", 
                         color='rating', color_continuous_scale="greens")
        st.plotly_chart(fig_rat, width='stretch')
        
    with col_d:
        heat = pd.crosstab(df['platform'], df['location'])
        fig_heat = px.imshow(heat, text_auto=True, aspect="auto", title="Platform vs Location Matrix", 
                           color_continuous_scale="BuPu")
        st.plotly_chart(fig_heat, width='stretch')

# =============== TAB 2: TEXTUAL NLP ===============
with tab2:
    st.header("Keyword & N-Gram Insights")
    all_words = ' '.join(df['cleaned_review'])
    
    if all_words.strip():
        st.subheader("Global Word Cloud")
        fig, ax = plt.subplots(figsize=(15, 4))
        wc = WordCloud(width=1200, height=300, background_color='white', 
                       colormap='ocean', max_words=150).generate(all_words)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        c1, c2 = st.columns(2)
        with c1:
            top_words = get_top_ngrams(df['cleaned_review'], 20, (1,1))
            fig_w = px.bar(top_words, x='Count', y='Phrase', orientation='h', 
                           title="Top 20 Single Words", color='Count', color_continuous_scale="Blues")
            fig_w.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_w, width='stretch')
            
        with c2:
            top_bigrams = get_top_ngrams(df['cleaned_review'], 20, (2,2))
            fig_b = px.bar(top_bigrams, x='Count', y='Phrase', orientation='h', 
                           title="Top 20 Bigrams (Two-Word Phrases)", color='Count', color_continuous_scale="Teal")
            fig_b.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_b, width='stretch')
    else:
        st.info("Not enough text data to generate NLP diagrams.")

# =============== TAB 3: SENTIMENT ===============
with tab3:
    st.header("Sentiment & Emotional Polarity")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_sd = px.histogram(df, x='sentiment_score', nbins=30, 
                              title="Sentiment Score Distribution (-1 to +1)", color_discrete_sequence=['#9467bd'])
        st.plotly_chart(fig_sd, width='stretch')
        
    with col_b:
        sent_cat = df['sentiment_category'].value_counts().reset_index()
        fig_sc = px.pie(sent_cat, values='count', names='sentiment_category', title="Sentiment Proportions", 
                        hole=0.3, color='sentiment_category', 
                        color_discrete_map={'Positive':'#00CC96', 'Negative':'#EF553B', 'Neutral':'#636EFA'})
        st.plotly_chart(fig_sc, width='stretch')
        
    col_c, col_d = st.columns(2)
    with col_c:
        fig_svr = px.box(df, x='rating', y='sentiment_score', 
                         title="Sentiment Score Range by Rating", color='rating')
        st.plotly_chart(fig_svr, width='stretch')
        
    with col_d:
        avg_sent = df.groupby('location')['sentiment_score'].mean().reset_index()
        fig_asl = px.bar(avg_sent, x='location', y='sentiment_score', 
                         title="Avg Sentiment by Location", color='sentiment_score', color_continuous_scale="RdYlGn")
        st.plotly_chart(fig_asl, width='stretch')

# =============== TAB 4: COMPLAINT RADAR ===============
with tab4:
    st.header("Negative Reviews Focus (Ratings 3 or Below)")
    neg_df = df[df['rating'] <= 3]
    
    if len(neg_df) > 0:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Total Flagged Reviews", len(neg_df))
            bad_locs = neg_df['location'].value_counts().reset_index()
            fig_bl = px.pie(bad_locs, values='count', names='location', hole=0.5, 
                            title="Complaints by Location", color_discrete_sequence=px.colors.sequential.Reds_r)
            st.plotly_chart(fig_bl, width='stretch')
            
        with c2:
            top_neg_words = get_top_ngrams(neg_df['cleaned_review'], 15, (1,1))
            fig_nw = px.bar(top_neg_words, x='Count', y='Phrase', orientation='h', 
                            title="Top 15 Complaint Keywords", color_discrete_sequence=['#EF553B'])
            fig_nw.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_nw, width='stretch')
            
        st.subheader("Top Complaint Themes (Bigrams)")
        top_neg_bigrams = get_top_ngrams(neg_df['cleaned_review'], 10, (2,2))
        fig_nb = px.bar(top_neg_bigrams, x='Count', y='Phrase', orientation='h', 
                        color_discrete_sequence=['#ab0000'])
        fig_nb.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_nb, width='stretch')
    else:
        st.success("Hooray! No negative reviews found in the selected filters. 🎉")

# =============== TAB 5: TOPIC MODELING ===============
with tab5:
    st.header("Unsupervised Topic Extraction (LDA Algorithms)")
    st.write("Dynamic execution: This groups your *currently filtered* text into 3 hidden latent topics on-the-fly.")
    
    if len(df) > 10:
        with st.spinner("Extracting hidden topics using Gensim LDA..."):
            texts = [str(text).split() for text in df['cleaned_review']]
            id2word = corpora.Dictionary(texts)
            corpus = [id2word.doc2bow(text) for text in texts]
            
            # Simple fast LDA
            lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=3, random_state=42, passes=4)
            topics = lda_model.show_topics(formatted=False, num_words=10)
            
            # Visualize top words per topic
            t_cols = st.columns(3)
            for t in range(3):
                with t_cols[t]:
                    st.subheader(f"Topic {t+1}")
                    words = [word[0] for word in topics[t][1]]
                    weights = [word[1] for word in topics[t][1]]
                    tdf = pd.DataFrame({'Word': words, 'Weight': weights})
                    
                    fig_ld = px.bar(tdf, x='Weight', y='Word', orientation='h', 
                                    color='Weight', color_continuous_scale="Sunset")
                    fig_ld.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=350, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_ld, width='stretch')
            
            # Assign dominant topic
            def get_dominant_topic(bow):
                if not bow: return 1
                topic_probs = lda_model.get_document_topics(bow)
                return max(topic_probs, key=lambda x: x[1])[0] + 1

            df['dominant_topic'] = [f"Topic {get_dominant_topic(c)}" for c in corpus]
            
            c_a, c_b = st.columns(2)
            with c_a:
                topic_dist = df['dominant_topic'].value_counts().reset_index()
                fig_td = px.pie(topic_dist, values='count', names='dominant_topic', 
                                title="Topic Weight Across All Reviews", hole=0.3)
                st.plotly_chart(fig_td, width='stretch')
            with c_b:
                topic_loc = pd.crosstab(df['location'], df['dominant_topic']).reset_index()
                topic_loc_melt = topic_loc.melt(id_vars='location', var_name='Topic', value_name='Count')
                fig_tl = px.bar(topic_loc_melt, x='location', y='Count', color='Topic', 
                                title="Topic Presence within Locations", barmode='stack')
                st.plotly_chart(fig_tl, width='stretch')
    else:
        st.warning("Not enough text data to train a reliable topic model. Please adjust your filters to include more reviews.")
