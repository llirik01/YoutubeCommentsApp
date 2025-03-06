import pandas as pd
import spacy
from collections import Counter
import string
import emoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px


# Mapping of language codes to corresponding spaCy models
nlp_models = {
    "en": spacy.load("models/en_core_web_sm"),
    "uk": spacy.load("models/uk_core_news_sm"),
    "fr": spacy.load("models/fr_core_news_sm"),
    "de": spacy.load("models/de_core_news_sm"),
    "it": spacy.load("models/it_core_news_sm"),
    "ja": spacy.load("models/ja_core_news_sm"),
    "ko": spacy.load("models/ko_core_news_sm"),
    "pl": spacy.load("models/pl_core_news_sm"),
    "pt": spacy.load("models/pt_core_news_sm"),
    "ru": spacy.load("models/ru_core_news_sm"),
    "es": spacy.load("models/es_core_news_sm"),
    "sv": spacy.load("models/sv_core_news_sm"),
    "ro": spacy.load("models/ro_core_news_sm"),
    "nl": spacy.load("models/nl_core_news_sm"),
    "hr": spacy.load("models/hr_core_news_sm"),
    "el": spacy.load("models/el_core_news_sm"),
    "sl": spacy.load("models/sl_core_news_sm"),
    "nb": spacy.load("models/nb_core_news_sm"),
    "mk": spacy.load("models/mk_core_news_sm"),
    "lt": spacy.load("models/lt_core_news_sm"),
    "fi": spacy.load("models/fi_core_news_sm"),
    "da": spacy.load("models/da_core_news_sm"),
    "ca": spacy.load("models/ca_core_news_sm"),
}


def create_df_for_fa(df):
    """
    Process a DataFrame by lemmatizing text, removing punctuation, and cleaning emojis.

    :param df: DataFrame - Input DataFrame containing comments
    :return: DataFrame - Processed DataFrame with lemmatized comments
    """

    def lemmatize_text(text, lang):
        """
        Lemmatize text based on the specified language.

        :param text: str - Input text
        :param lang: str - Language code
        :return: str - Lemmatized text or original text if language is unsupported
        """
        if not isinstance(text, str) or text.strip() == "":
            return text

        nlp = nlp_models.get(lang)  # Get the corresponding model
        if not nlp:
            return text  # Return unchanged text if the language is unsupported

        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    # Remove rows where language is unknown
    df = df[df['language'] != 'unknown'].reset_index(drop=True)
    # Add a column with lemmatized comments
    df["comment_lemmatized"] = df.apply(lambda row: lemmatize_text(row["clean_comment"], row["language"]), axis=1)
    # Remove punctuation
    df["comment_lemmatized"] = df["comment_lemmatized"].astype(str).apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation)) if isinstance(x, str) else x)
    # Remove emojis
    df['comment_lemmatized'] = df['comment_lemmatized'].apply(lambda x: emoji.replace_emoji(x, ""))

    return df[['author', 'comment', 'language', 'dominant_sentiment', 'comment_lemmatized']]


def frequent_words(df, n=15):
    """
    Extract the most frequent words categorized as nouns, verbs and adjectives.

    :param df: DataFrame - Input DataFrame
    :param n: int - Number of top words to return
    :return: tuple - Top nouns, verbs and adjectives
    """

    def extract_from_df(dff, lang_part, text_column='comment_lemmatized', language_column='language'):
        """
        Extract words of a specific part of speech from text.

        :param dff: DataFrame - Input DataFrame
        :param lang_part: str - Part of speech (e.g., NOUN, VERB, ADJ)
        :param text_column: str - Column name containing text data (default: 'comment_lemmatized')
        :param language_column: str - Column name specifying the language of the text (default: 'language')
        :return: list - Extracted words
        """

        def extract_words(text, lang):
            """
            Function to extract words of a specific part of speech.
            """
            # Check if text is a valid string
            if pd.isna(text) or not isinstance(text, str):
                return []

            nlp = nlp_models.get(lang)  # Get the corresponding model
            if not nlp:
                return []  # Return an empty list if the language is unsupported

            doc = nlp(text)
            return [token.lemma_.lower() for token in doc if token.pos_ == lang_part]

        list_of_words = []
        for _, row in dff.iterrows():
            lang = row.get(language_column, "en")  # Default to English
            words = extract_words(row[text_column], lang)
            list_of_words.extend(words)

        return list_of_words

    def get_top_phrases(words):
        """
        Retrieve the most common words from a given list.

        :param words: list - List of words
        :return: dict - Dictionary containing the most frequent words
        """
        word_counts = Counter(words)

        return {"top_words": word_counts.most_common(n)}

    all_nouns = extract_from_df(df, lang_part='NOUN')
    all_verbs = extract_from_df(df, lang_part='VERB')
    all_adj = extract_from_df(df, lang_part='ADJ')

    top_nouns = get_top_phrases(all_nouns)
    top_verbs = get_top_phrases(all_verbs)
    top_adj = get_top_phrases(all_adj)

    return top_nouns["top_words"], top_verbs["top_words"], top_adj["top_words"]


def generate_wordcloud(df):
    """
    Function to create a word cloud based on lemmatized comments.

    :param df: DataFrame with columns "comment_lemmatized" and "language"
    :return: Matplotlib figure
    """
    def extract_keywords(text, lang):
        # Check if text is a valid string
        if not isinstance(text, str) or text.strip() == "":
            return ""

        nlp = nlp_models.get(lang)  # Get the corresponding model
        if not nlp:
            return ""

        doc = nlp(text)  # Process the text with the model
        return " ".join([token.lemma_ for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ"}])

    df["keywords"] = df.apply(lambda row: extract_keywords(row["comment_lemmatized"], row["language"]), axis=1)
    text = " ".join(df["keywords"].dropna())

    wordcloud = WordCloud(width=1200, height=600, background_color="white", colormap="viridis").generate(text)

    # Create a figure object for Streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


def plot_word_frequencies(word_data):
    """
    Plots a stylish bar chart of word frequencies.

    :param word_data: list of tuples (word, count)
    :return: Plotly figure
    """
    # Input data validation
    if not isinstance(word_data, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in word_data):
        raise ValueError("Вхідні дані повинні бути списком кортежів (слово, кількість)")

    if not all(isinstance(count, (int, float)) for _, count in word_data):
        raise ValueError("Другий елемент кожного кортежу повинен бути числом (int або float)")

    # Unpacking tuples into two lists
    words, counts = zip(*word_data) if word_data else ([], [])

    # Create a bar chart with custom coloring
    fig = px.bar(
        x=words, y=counts,
        labels={"y": "Frequency"},
        text=counts,  # Display frequency values on top of each bar
        color=counts,  # Gradient color based on the value
        color_continuous_scale="peach",
    )

    # Adding styles
    fig.update_traces(
        textfont_size=16,
        textposition="outside",
        marker_line_color='black',
        marker_line_width=2,
        hovertemplate='',  # Remove hover effect
        hoverinfo="skip"
    )

    # Update axis style
    fig.update_layout(
        height=600,
        xaxis_title=None,
        xaxis_tickangle=-30,  # Rotate word labels
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background for the plot
        font=dict(family="Arial, sans-serif", size=18, color="black"),  # Font style
        coloraxis_showscale=False  # Disable color gradient scale
    )

    return fig
