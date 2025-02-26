import pandas as pd
import subprocess
import spacy
from collections import Counter
# from itertools import islice
import string
import emoji

# Список моделей, які потрібно завантажити
models = [
    "en_core_web_sm", "uk_core_news_sm", "fr_core_news_sm", "de_core_news_sm",
    "it_core_news_sm", "ja_core_news_sm", "ko_core_news_sm", "pl_core_news_sm",
    "pt_core_news_sm", "ru_core_news_sm", "es_core_news_sm", "sv_core_news_sm",
    "ro_core_news_sm", "nl_core_news_sm", "hr_core_news_sm", "el_core_news_sm",
    "sl_core_news_sm", "nb_core_news_sm", "mk_core_news_sm", "lt_core_news_sm",
    "fi_core_news_sm", "da_core_news_sm", "ca_core_news_sm"
]

# Завантажуємо моделі перед використанням
for lang, model in models.items():
    try:
        spacy.load(model)
    except OSError:
        print(f"Модель {model} не знайдена. Завантажуємо...")
        subprocess.run(["python", "-m", "spacy", "download", model], check=True)
        print(f"Модель {model} успішно встановлена!")

# Тепер можна використовувати моделі
nlp_models = {lang: spacy.load(lang) for lang in models}

def create_df_for_fa(df):
    # Функція лематизації з підтримкою багатьох мов
    def lemmatize_text(text, lang):
        if not isinstance(text, str) or text.strip() == "":
            return text

        nlp = nlp_models.get(lang)  # Отримуємо відповідну модель
        if not nlp:
            return text  # Якщо мова не підтримується, повертаємо без змін

        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    # Видалення рядків
    df = df[df['language'] != 'unknown'].reset_index(drop=True)
    # Додаємо колонку з лематизованими коментарями
    df["comment_lemmatized"] = df.apply(lambda row: lemmatize_text(row["clean_comment"], row["language"]), axis=1)
    # Видалення розділових знакі
    df["comment_lemmatized"] = df["comment_lemmatized"].astype(str).apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation)) if isinstance(x, str) else x)
    # Видалення емоджі
    df['comment_lemmatized'] = df['comment_lemmatized'].apply(lambda x: emoji.replace_emoji(x, ""))

    return df[['author', 'comment', 'language', 'dominant_sentiment', 'comment_lemmatized']]


def frequent_words(df, n=15):

    def extract_from_df(dff, lang_part, text_column='comment_lemmatized', language_column='language'):

        # Функція для отримання іменників (або іншої частини мови)
        def extract_words(text, lang):
            if pd.isna(text) or not isinstance(text, str):
                return []

            nlp = nlp_models.get(lang)  # Отримуємо відповідну модель
            if not nlp:
                return []  # Якщо мова не підтримується, повертаємо пустий список

            doc = nlp(text)
            return [token.lemma_.lower() for token in doc if token.pos_ == lang_part]

        list_of_words = []
        for _, row in dff.iterrows():
            lang = row.get(language_column, "en")  # За замовчуванням англійська
            words = extract_words(row[text_column], lang)
            list_of_words.extend(words)

        return list_of_words

    # Функція для отримання популярних слів, біграм, триграм
    def get_top_phrases(words):
        # bigrams = list(zip(words, islice(words, 1, None)))
        # trigrams = list(zip(words, islice(words, 1, None), islice(words, 2, None)))

        word_counts = Counter(words)
        # bigram_counts = Counter(bigrams)
        # trigram_counts = Counter(trigrams)

        return {
            "top_words": word_counts.most_common(n),
            # "top_bigrams": bigram_counts.most_common(n),
            # "top_trigrams": trigram_counts.most_common(n)
    }

    all_nouns = extract_from_df(df, lang_part='NOUN')
    all_verbs = extract_from_df(df, lang_part='VERB')
    all_adj = extract_from_df(df, lang_part='ADJ')

    # Отримуємо популярні слова
    top_nouns = get_top_phrases(all_nouns)
    top_verbs = get_top_phrases(all_verbs)
    top_adj = get_top_phrases(all_adj)

    return top_nouns["top_words"], top_verbs["top_words"], top_adj["top_words"]


import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px


def generate_wordcloud(df, nlp_models=nlp_models):
    """
    Функція для створення хмари слів на основі лематизованих коментарів.

    :param df: DataFrame з колонками "comment_lemmatized" і "language"
    :param nlp_models: Словник NLP-моделей для різних мов
    :return: Matplotlib figure
    """
    def extract_keywords(text, lang):
        if not isinstance(text, str) or text.strip() == "":
            return ""

        nlp = nlp_models.get(lang)
        if not nlp:
            return ""

        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ"}])

    df["keywords"] = df.apply(lambda row: extract_keywords(row["comment_lemmatized"], row["language"]), axis=1)
    text = " ".join(df["keywords"].dropna())

    wordcloud = WordCloud(width=1200, height=600, background_color="white", colormap="viridis").generate(text)

    # Створення об'єкта фігури для Streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


def plot_word_frequencies(word_data):
    """
    Будує стильний бар-чарт частотності слів.

    :param word_data: список кортежів (слово, кількість)
    """
    # Перевірка вхідних даних
    if not isinstance(word_data, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in word_data):
        raise ValueError("Вхідні дані повинні бути списком кортежів (слово, кількість)")

    if not all(isinstance(count, (int, float)) for _, count in word_data):
        raise ValueError("Другий елемент кожного кортежу повинен бути числом (int або float)")

    # Розпаковка кортежів у два списки
    words, counts = zip(*word_data) if word_data else ([], [])

    # Створюємо бар-чарт із кастомним кольором
    fig = px.bar(
        x=words, y=counts,
        labels={"y": "Frequency"},
        text=counts,  # Додаємо значення частоти на кожен бар
        color=counts,  # Градієнтний колір залежно від значення
        color_continuous_scale="peach",  # Кольорова схема
    )

    # Додаємо стилі
    fig.update_traces(
        textfont_size=16,  # Розмір підписів на барах
        textposition="outside",  # Розташування підписів над стовпцями
        marker_line_color='black',  # Колір контуру барів
        marker_line_width=2,  # Товщина контуру
        hovertemplate='',  # Прибирає hover
        hoverinfo="skip"
    )

    # Оновлюємо стиль осей
    fig.update_layout(
        height=600,
        xaxis_title=None,
        xaxis_tickangle=-45,  # Поворот підписів слів
        plot_bgcolor="rgba(0,0,0,0)",  # Прозорий фон графіка
        # paper_bgcolor="#f4f4f4",  # Фон всієї області графіка
        font=dict(family="Arial, sans-serif", size=18, color="black"),  # Стиль тексту
        coloraxis_showscale=False  # Відключаємо шкалу градієнту кольорів
    )

    # fig.show()
    return fig