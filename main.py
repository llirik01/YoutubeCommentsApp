import streamlit as st
from youtubeAPI import *
from sentiment_analysis import *
from frequency_analysis import *
import time

# API key for accessing YouTube data
api_key = 'AIzaSyA40tswEwzrSr5HrxvescQLz06S9pQXtfo'

# Setting up the Streamlit app configuration
st.set_page_config(page_title='YouTubeCommentsApp', layout='wide')

st.logo('logo.png', size="large")

# Initializing session state variables if they are not set
if "page" not in st.session_state:
    st.session_state["page"] = "home"  # Default page

if "primary_df" not in st.session_state:
    st.session_state["primary_df"] = None  # Stores raw comments data

if "df_clean_sentiment" not in st.session_state:
    st.session_state["df_clean_sentiment"] = None  # Stores cleaned data for sentiment analysis

if "df_sentiment" not in st.session_state:
    st.session_state["df_sentiment"] = None  # Stores sentiment analysis results


# Function to switch between different pages
def set_page(page_name):
    st.session_state["page"] = page_name


# Sidebar with navigation buttons
with st.sidebar:
    if st.button("Homepage", use_container_width=True):
        set_page("home")

    st.divider()

    # Input field for YouTube video link
    link = st.text_input("", placeholder="Paste your link here", label_visibility="hidden")
    submit_button = st.button("Pass the link", use_container_width=True)

    if submit_button and link:
        # Clearing previous data before fetching new comments
        st.session_state.primary_df = None
        st.session_state.df_clean_sentiment = None
        st.session_state.df_sentiment = None

        with st.spinner("Fetching comments..."):
            time.sleep(3)
            primary_df = get_comments(api_key=api_key, video_url=link)  # Fetching comments

        if primary_df is None:
            st.error("Please insert a link before submitting.", icon='❗️')
        else:
            with st.spinner("Cleaning data for sentiment analysis..."):
                time.sleep(3)
                df_clean_sentiment = clean_df_for_sentiment(primary_df)  # Cleaning data

            with st.spinner("Performing sentiment analysis..."):
                df_sentiment = make_sentiment_analysis(df_clean_sentiment)  # Performing sentiment analysis

            # Storing results in session state
            st.session_state["primary_df"] = primary_df
            st.session_state['df_clean_sentiment'] = df_clean_sentiment
            st.session_state["df_sentiment"] = df_sentiment
            st.success(f'Analysis complete! {len(primary_df)} comments found', icon="✅")

    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.header("App`s functionality", divider=True)

    # Buttons for different functionalities
    if st.button("Top-100 most liked comments", use_container_width=True):
        set_page("function_1")
    if st.button("Top-100 most positive comments", use_container_width=True):
        set_page("function_2")
    if st.button("Top-100 most negative comments", use_container_width=True):
        set_page("function_3")
    if st.button("Show all comments", use_container_width=True):
        set_page("function_4")
    if st.button("Sentiment visualization", use_container_width=True):
        set_page("function_5")
    if st.button("Frequent words visualization", use_container_width=True):
        set_page("function_6")

    st.divider()
    st.markdown("<h5 style='text-align: center; color: black; font-weight: normal;'>Created by Kyryl Shum 🧑‍💻</h5>",
                unsafe_allow_html=True)

# Main screen logic based on the selected page
if st.session_state["page"] == "home":
    st.markdown("<h1 style='font-family: san-serif; text-align: center; color: black; \
     font-weight: 650;'>📺 YouTube Comment Analysis</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: san-serif; text-align: left; color: #261324; \
         font-weight: 550;'>🎯 Purpose of the Application</h2>", unsafe_allow_html=True)

    st.markdown("<p style='font-family: san-serif; text-align: justify; color: #261324; \
             font-weight: normal; font-size: 20px'>This application is designed for the automatic analysis of \
             comments under YouTube videos. It allows users to easily retrieve, process, and examine comments, \
             identifying key trends, emotional tone, and frequently used words. The tool is useful for bloggers, \
             marketers, analysts, and anyone who wants to gain deeper insights into audience opinions.</p>",
                unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: san-serif; text-align: left; color: black; \
             font-weight: 550;'>📊 Sentiment Analysis</h2>", unsafe_allow_html=True)

    st.markdown("<p style='font-family: san-serif; text-align: justify; color: black; \
                 font-weight: normal; font-size: 20px'>Sentiment analysis is used to automatically determine \
                 the emotional tone of comments. Each comment can be classified as positive, negative, or neutral. \
                 It is important to note that this classification does not necessarily reflect the audience's \
                 attitude toward the video creator but rather indicates general public sentiment regarding the \
                 discussed topic. For example, negative comments may express emotions such as sadness, regret, \
                 sympathy, or dissatisfaction. Similarly, positive comments may convey joy, support, hope, or \
                 admiration.</p>", unsafe_allow_html=True)

    st.markdown("<p style='font-family: san-serif; text-align: justify; color: black; \
                     font-weight: normal; font-size: 20px'>It is also important to consider that classification \
                     models may not always accurately recognize sarcastic or ironic statements. As a result, \
                     such messages may be mistakenly categorized into a different sentiment group, which should \
                     be considered when interpreting the analysis results.</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: san-serif; text-align: left; color: black; \
                    font-weight: 550;'>🔍 Frequency Analysis</h2>", unsafe_allow_html=True)

    st.markdown("<p style='font-family: san-serif; text-align: justify; color: black; \
                         font-weight: normal; font-size: 20px'>Frequency analysis helps identify the most \
                         commonly used words in comments. This allows users to understand what topics or events \
                         are being most discussed by the audience. Through this analysis, users can quickly \
                         highlight key trends in comments and gain deeper insights into audience reactions or \
                         the topics being discussed in the video.</p>", unsafe_allow_html=True)

    st.markdown("<h2 style='font-family: san-serif; text-align: left; color: black; \
                        font-weight: 550;'>🔤 Supported Languages</h2>", unsafe_allow_html=True)

    st.markdown("<p style='font-family: san-serif; text-align: justify; color: black; \
                         font-weight: normal; font-size: 20px'>The application can analyze comments in 23 languages:  \
                         English (en), Ukrainian (uk), French (fr), German (de), Italian (it), Japanese (ja), \
                         Korean (ko), Polish (pl), Portuguese (pt), Russian (ru), Spanish (es), Swedish (sv), \
                         Romanian (ro), Dutch (nl), Croatian (hr), Greek (el), Slovenian (sl), Norwegian Bokmål (nb), \
                         Macedonian (mk), Lithuanian (lt), Finnish (fi), Danish (da), Catalan (ca).",
                unsafe_allow_html=True)

    st.markdown("<p style='font-family: san-serif; text-align: left; text-align: justify; color: black; \
                        font-weight: 550; font-size: 24px'> 📢 Let's start analyzing! Paste the video link and \
                        get a detailed comment analysis.</p>", unsafe_allow_html=True)


# Page for displaying top 100 most liked comments
elif st.session_state["page"] == "function_1":
    st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
             font-weight: 550;'>Top-100 most liked comments</h2>", unsafe_allow_html=True)

    # Check if data is available
    if "df_sentiment" not in st.session_state or st.session_state["df_sentiment"] is None:
        st.warning("No data available. Please provide a link and fetch comments first.", icon='⚠️')
    else:
        # Get top liked comments and display them in a table
        toplikes_df = top_liked_comments(st.session_state["df_sentiment"])
        st.dataframe(data=toplikes_df, use_container_width=True)

# Page for displaying the top 100 most positive comments
elif st.session_state["page"] == "function_2":
    st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
                 font-weight: 550;'>Top-100 most positive comments</h2>", unsafe_allow_html=True)

    st.info('Only comments with a sentiment score above 0.85 are displayed.', icon='ℹ️')

    # Check if data is available
    if "df_sentiment" not in st.session_state or st.session_state["df_sentiment"] is None:
        st.warning("No data available. Please provide a link and fetch comments first.", icon='⚠️')
    else:
        # Get top positive comments and display them in a table
        topPos_df = show_positive_comments(st.session_state["df_sentiment"])
        st.dataframe(data=topPos_df, use_container_width=True)

# Page for displaying the top 100 most negative comments
elif st.session_state["page"] == "function_3":
    st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
                     font-weight: 550;'>Top-100 most negative comments</h2>", unsafe_allow_html=True)
    st.info('Only comments with a sentiment score above 0.85 are displayed.', icon='ℹ️')

    # Check if data is available
    if "df_sentiment" not in st.session_state or st.session_state["df_sentiment"] is None:
        st.warning("No data available. Please provide a link and fetch comments first.", icon='⚠️')
    else:
        # Get top negative comments and display them in a table
        topNeg_df = show_negative_comments(st.session_state["df_sentiment"])
        st.dataframe(data=topNeg_df, use_container_width=True)

# Page for displaying all comments
elif st.session_state["page"] == "function_4":
    st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
                         font-weight: 550;'>Show all comments</h2>", unsafe_allow_html=True)

    # Check if data is available
    if "df_sentiment" not in st.session_state or st.session_state["df_sentiment"] is None:
        st.warning("No data available. Please provide a link and fetch comments first.", icon='⚠️')
    else:
        # Display all comments with selected columns
        st.dataframe(data=st.session_state["df_sentiment"][['author', 'comment', 'date', 'likes']],
                     use_container_width=True)

# Page for sentiment distribution visualization
elif st.session_state["page"] == "function_5":
    st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
                             font-weight: 550;'>Sentiment Distribution Visualization</h2>", unsafe_allow_html=True)

    st.write("Select sentiments to display:")

    # Available sentiment options
    sentiment_options = ["negative", "neutral", "positive"]

    # Display checkboxes for selecting sentiment filters
    cols = st.columns(len(sentiment_options))
    selected_sentiments = []

    for col, sentiment in zip(cols, sentiment_options):
        with col:
            if st.checkbox(sentiment, value=True):
                selected_sentiments.append(sentiment)

    # Check if data is available
    if "df_sentiment" not in st.session_state or st.session_state["df_sentiment"] is None:
        st.error("No data available for visualization. Please provide a link and fetch comments first.", icon='❗️')
    elif not selected_sentiments:
        st.warning("Select at least one sentiment to display.", icon='⚠️')
    else:
        # Filter data based on selected sentiments
        filtered_df = st.session_state["df_sentiment"][
            st.session_state["df_sentiment"]["dominant_sentiment"].isin(selected_sentiments)
        ]

        if filtered_df.empty:
            st.warning("No data available for the selected sentiments.", icon='⚠️')
        else:
            # Display sentiment distribution chart
            with st.spinner("Analyzing sentiment distribution..."):
                time.sleep(2)
                sentiments_count = count_classes(filtered_df)
                sentiments_disribution_graph = plot_sentiment_analysis(sentiments_count)
                st.plotly_chart(sentiments_disribution_graph, use_container_width=True, theme=None)

            st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
                                         font-weight: 550;'>ANOVA Test and Comment Length Chart</h2>",
                        unsafe_allow_html=True)

            with st.spinner("Processing ANOVA test..."):
                anova_test = perform_anova_test(st.session_state["df_sentiment"])
                st.info(anova_test, icon='ℹ️')

            with st.spinner("Generating comment length chart..."):
                time.sleep(2)
                comments_length_graph = plot_comments_length_analysis(filtered_df)
                st.plotly_chart(comments_length_graph, use_container_width=True, theme=None)

# Page for frequent words visualization
elif st.session_state["page"] == "function_6":
    st.markdown("<h2 style='font-family: san-serif; text-align: center; color: #261324; \
                                             font-weight: 550;'>Frequent Words Visualization</h2>",
                unsafe_allow_html=True)

    # Check if data is available
    if "df_clean_sentiment" not in st.session_state or st.session_state["df_clean_sentiment"] is None:
        st.error("No data available for visualization. Please provide a link and fetch comments first.", icon='❗️')
    else:
        df_fr_an = create_df_for_fa(st.session_state["df_clean_sentiment"])
        frequent_words = frequent_words(df_fr_an)  # get frequent words

        # Dropdown to select chart type
        chart_type = st.selectbox(
            "Select chart type:",
            ["Word Cloud", "Noun Frequencies", "Verb Frequencies", "Adjective Frequencies"]
        )

        # Display the selected chart type
        if chart_type == "Word Cloud":
            with st.spinner("Generating word cloud..."):
                time.sleep(3)
                wordcloud = generate_wordcloud(df_fr_an)
            if wordcloud:
                st.pyplot(wordcloud, use_container_width=True)
            else:
                st.warning("I`m sorry. Word cloud could not be generated.", icon='⚠️')

        elif chart_type == "Noun Frequencies":
            with st.spinner("Generating noun frequency chart..."):
                time.sleep(3)
                nouns_plot = plot_word_frequencies(frequent_words[0])  # nouns
            if nouns_plot:
                st.plotly_chart(nouns_plot, use_container_width=True, theme=None)
            else:
                st.warning("I`m sorry. No noun frequencies available.", icon='⚠️')

        elif chart_type == "Verb Frequencies":
            with st.spinner("Generating verb frequency chart..."):
                time.sleep(3)
                verbs_plot = plot_word_frequencies(frequent_words[1])  # verbs
            if verbs_plot:
                st.plotly_chart(verbs_plot, use_container_width=True, theme=None)
            else:
                st.warning("I`m sorry. No verb frequencies available.", icon='⚠️')

        elif chart_type == "Adjective Frequencies":
            with st.spinner("Generating adjective frequency chart..."):
                time.sleep(3)
                adjectives_plot = plot_word_frequencies(frequent_words[2])  # adjectives
            if adjectives_plot:
                st.plotly_chart(adjectives_plot, use_container_width=True, theme=None)
            else:
                st.warning("I`m sorry. No adjective frequencies available.", icon='⚠️')
