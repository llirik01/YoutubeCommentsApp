import pandas as pd
import re
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.stats import f_oneway
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def clean_df_for_sentiment(df):
    """
    Clean the DataFrame containing comments, remove unwanted elements like links, mentions, hashtags
    and detect the language of each comment.

    :param df: DataFrame - Input DataFrame with a 'comment' column containing text data
    :return: DataFrame - DataFrame with added 'clean_comment' and 'language' columns
    """
    def clean_text(text):
        """
        Clean the given text by removing HTML tags, URLs, user mentions, and hashtags.

        :param text: str - Input text to clean
        :return: str - Cleaned text
        """
        if not isinstance(text, str) or text.strip() == "":
            return ""

        text = text.lower()  # Convert to lowercase
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'https?://\S+', '', text).strip()  # Remove links, leaving only text
        text = re.sub(r'@\w+', '', text)  # Remove user mentions
        text = re.sub(r'#\w+', '', text)  # Remove hashtags
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

        return text

    def is_only_link(text):
        """
        Check if the comment consists only of a link.

        :param text: str - Input text to check
        :return: bool - True if the text is only a URL, False otherwise
        """
        return bool(re.fullmatch(r'https?://\S+', text.strip()))

    def detect_language(text):
        """
        Detect the language of the input text.

        :param text: str - Input text to detect language
        :return: str - Detected language code (e.g., 'en', 'zh', 'es') or 'unknown' if detection fails
        """
        try:
            lang = detect(text)
            return "zh" if lang in ["zh-cn", "zh-tw"] else lang
        except:
            return "unknown"

    # Remove comments that contain only links
    df = df[~df["comment"].astype(str).apply(is_only_link)]
    # Clean the comments
    df["clean_comment"] = df["comment"].astype(str).apply(clean_text)
    # Detect the language
    df["language"] = df["clean_comment"].apply(detect_language)

    return df


def make_sentiment_analysis(df):
    """
    Perform sentiment analysis on the comments in the DataFrame and calculate the dominant sentiment for each comment.

    :param df: pandas.DataFrame - Input DataFrame with 'clean_comment' column containing text data
    :return: pandas.DataFrame - DataFrame with added columns for sentiment scores ('negative', 'neutral', 'positive'),
             'dominant_sentiment' and 'word_count'
    """
    # Load the tokenizer and model for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-roberta-base-sentiment-multilingual")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/xlm-roberta-base-sentiment-multilingual")

    def analyze_sentiment(text):
        """
        Analyze the sentiment of a single text using a pre-trained model.

        :param text: str - The input text to analyze
        :return: dict - A dictionary containing the sentiment scores for negative, neutral and positive
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1).tolist()[0]

        return {"negative": scores[0], "neutral": scores[1], "positive": scores[2]}

    # Apply sentiment analysis to all comments
    df[["negative", "neutral", "positive"]] = df["comment"].apply(lambda x: pd.Series(analyze_sentiment(str(x))))
    # Determine the dominant sentiment for each comment
    dominant_sentiment = df[['negative', 'neutral', 'positive']].idxmax(axis=1)
    df['dominant_sentiment'] = dominant_sentiment
    # Calculate the word count for each comment
    df['word_count'] = df['comment'].astype(str).apply(lambda x: max(1, len(re.findall(r'\b\w+\b', x))))

    return df


def top_liked_comments(df, n=100):
    """
    Retrieve the top N most liked comments from the DataFrame.

    :param df: DataFrame - Input DataFrame with sentiment analysis performed
    :param n: int - The number of top liked comments to retrieve (default is 100)
    :return: DataFrame - A DataFrame with the top N liked comments, including 'author', 'comment', 'likes'
             and 'dominant_sentiment'
    """
    top_df = df.sort_values(by="likes", ascending=False).reset_index(drop=True).head(n)

    return top_df[['author', 'comment', 'likes', 'dominant_sentiment']]


def show_positive_comments(df, thresh=0.85, n=100):
    """
    Retrieve the top N comments with the highest positive sentiment scores.

    :param df: DataFrame - Input DataFrame with sentiment analysis performed
    :param thresh: float - The threshold for the positive sentiment score (default is 0.85)
    :param n: int - The number of top positive comments to retrieve (default is 100)
    :return: DataFrame - A DataFrame with the top N positive comments, including 'author', 'comment' and 'positive'
    """
    emotional_comments_pos = df[df['positive'] > thresh]
    emotional_comments_pos = emotional_comments_pos[['author', 'comment', 'positive']].sort_values(by=['positive'],
                                                                                                   ascending=False)

    return emotional_comments_pos.reset_index(drop=True).head(n)


def show_negative_comments(df, thresh=0.85, n=100):
    """
    Retrieve the top N comments with the highest negative sentiment scores.

    :param df: DataFrame - Input DataFrame with sentiment analysis performed
    :param thresh: float - The threshold for the positive sentiment score (default is 0.85)
    :param n: int - The number of top positive comments to retrieve (default is 100)
    :return: DataFrame - A DataFrame with the top N positive comments, including 'author', 'comment' and 'negative'
    """
    emotional_comments_neg = df[df['negative'] > thresh]
    emotional_comments_neg = emotional_comments_neg[['author', 'comment', 'negative']].sort_values(by=['negative'],
                                                                                                   ascending=False)

    return emotional_comments_neg.reset_index(drop=True).head(n)


def count_classes(df):
    """
    Count the frequency of each dominant sentiment class in the DataFrame.

    :param df: DataFrame - Input DataFrame containing the 'dominant_sentiment' column
    :return: Series - A Series containing the count of each sentiment class
    """

    return df['dominant_sentiment'].value_counts()


def perform_anova_test(df):
    """
    Perform an ANOVA test to check the significance of differences in comment length across sentiment classes.

    :param df: DataFrame - Input DataFrame containing the columns 'word_count' (length of the comments)
                                       and 'dominant_sentiment' (sentiment class of the comment)
    :return: str - Conclusion based on the ANOVA test result, indicating whether there is a significant difference
                    in comment lengths between sentiment classes.
    """
    # Split the data by sentiment classes
    neg = df[df['dominant_sentiment'] == 'negative']['word_count']
    neu = df[df['dominant_sentiment'] == 'neutral']['word_count']
    pos = df[df['dominant_sentiment'] == 'positive']['word_count']

    # Perform the ANOVA test
    stat, p_value = f_oneway(neg, neu, pos)

    if p_value < 0.05:
        conclusion = "ANOVA is a statistical test that helps determine whether comment length varies depending \
        on sentiment. The results showed that there is a correlation between text length and its emotional tone."
    else:
        conclusion = "ANOVA is a statistical test that helps determine whether comment length varies depending \
        on sentiment. The analysis did not reveal a significant correlation between comment sentiment and its length."

    return conclusion


def plot_sentiment_analysis(sentiments):
    """
    Create a pie chart and a horizontal bar chart for sentiment analysis visualization.

    :param sentiments: Series - Sentiment distribution where the index represents sentiment categories
                                       (negative, neutral, positive), and values represent their respective counts.
    :return: Plotly figure - A figure containing both pie and bar charts for sentiment visualization.
    """
    if not isinstance(sentiments, pd.Series):
        raise TypeError("Expected a pandas Series containing sentiment data.")

    # Define colors with transparency (0.6 - semi-transparent)
    color_map = {
        'negative': 'rgba(255, 0, 0, 0.6)',   # Red
        'neutral': 'rgba(128, 128, 128, 0.6)',  # Gray
        'positive': 'rgba(0, 128, 0, 0.6)'     # Green
    }
    colors = [color_map.get(sentiment, 'rgba(128, 128, 128, 0.6)') for sentiment in sentiments.index]

    # Calculate percentages for the pie chart
    total = sentiments.sum()
    sentiments_percentage = (sentiments / total * 100).round(1)

    # Create a combined figure
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        column_widths=[0.3, 0.7]  # Make the bar chart wider
    )

    # Pie chart
    fig.add_trace(go.Pie(
        labels=sentiments_percentage.index,
        values=sentiments_percentage.values,
        name="Sentiments",
        hole=0.2,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        hoverinfo='none'
    ), row=1, col=1)

    # Bar chart
    fig.add_trace(go.Bar(
        y=sentiments.index,
        x=sentiments.values,
        name="Sentiments",
        orientation='h',
        marker=dict(color=colors),
        text=sentiments.values,
        textposition='outside',
        hoverinfo='none'
    ), row=1, col=2)

    # Global layout for both charts
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        font=dict(family="Arial, sans-serif", size=16, color="black"),  # Font style
        showlegend=False,
    )

    return fig


def plot_comments_length_analysis(df):
    """
    Creates a subplot with two visualizations:
    1. A horizontal boxplot to analyze comment length distribution by sentiment.
    2. A filled line (area) chart showing the distribution of comment lengths for each sentiment category.

    :param df: DataFrame - A DataFrame containing 'word_count' (comment length)
                                  and 'dominant_sentiment' (sentiment category).
    :return: Plotly figure - A figure containing both visualizations.
    """

    # Define colors with transparency (0.6 - semi-transparent)
    color_map = {
        "negative": "rgba(255, 0, 0, 0.6)",   # Red
        "neutral": "rgba(128, 128, 128, 0.6)",  # Gray
        "positive": "rgba(0, 128, 0, 0.6)"     # Green
    }

    # Create a subplot with two rows (stacked visualizations)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],  # Proportional heights
        shared_xaxes=True,  # Shared X-axis for both visualizations
        vertical_spacing=0.05
    )

    # Horizontal boxplot
    boxplot = px.box(
        df, y="dominant_sentiment", x="word_count",
        color="dominant_sentiment",
        color_discrete_map=color_map,
        labels={"dominant_sentiment": "Sentiment", "word_count": "Comment Length (words)"},
        orientation="h"
    )

    # Remove hover details and add traces to the subplot
    for trace in boxplot["data"]:
        trace.update(hoverinfo="skip", hovertemplate='', marker=dict(size=12))  # Disable hover
        fig.add_trace(trace, row=1, col=1)

    # Prepare data for the area chart
    length_counts = df.groupby(["dominant_sentiment", "word_count"]).size().reset_index(name="comment_count")

    # Area chart
    area_chart = px.area(
        length_counts, x="word_count", y="comment_count", color="dominant_sentiment",
        labels={"word_count": "Comment Length (words)", "comment_count": "Number of Comments"},
        color_discrete_map=color_map
    )

    # Remove hover details and add traces to the subplot
    for trace in area_chart["data"]:
        trace.update(hovertemplate="Comment Length (words): %{x}")
        fig.add_trace(trace, row=2, col=1)

    # Define X-axis range
    min_word_count = df[df["word_count"] > 0]["word_count"].min()  # Найменше значення > 0
    max_word_count = df["word_count"].max() + 5  # Трішки більше максимального значення

    # Update layout and styling
    fig.update_layout(
        template="plotly_white",
        height=700,
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        font=dict(family="Arial, sans-serif", size=16, color="black"),
        showlegend=False,  # Hide legend
        xaxis=dict(
            dtick=50,  # Step of 50
            range=[min_word_count, max_word_count]  # Exclude 0 for better readability
        ),
        xaxis2=dict(
            title="Comment Length (words)",  # X-axis label (for area chart)
            tickangle=-20,  # Rotate X-axis labels
            dtick=50,  # Step of 50
            range=[min_word_count, max_word_count]  # Exclude 0 for better readability
        ),
        yaxis2=dict(
            title="Number of Comments",
        )
    )

    return fig
