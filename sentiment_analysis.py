import pandas as pd
import re
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.stats import f_oneway


def clean_df_for_sentiment(df):
    # Функція очищення тексту
    def clean_text(text):
        if not isinstance(text, str) or text.strip() == "":
            return ""

        text = text.lower()  # Усе до нижнього регістру
        text = re.sub(r'<.*?>', '', text)  # Видаляємо HTML-теги
        text = re.sub(r'https?://\S+', '', text).strip()  # Видаляємо посилання, залишаючи текст
        text = re.sub(r'@\w+', '', text)  # Видаляємо згадки користувачів
        text = re.sub(r'#\w+', '', text)  # Видаляємо хештеги
        text = re.sub(r'\s+', ' ', text).strip()  # Прибираємо зайві пробіли

        return text

    # Функція перевірки, чи складається коментар лише з посилання
    def is_only_link(text):
        return bool(re.fullmatch(r'https?://\S+', text.strip()))

    # Функція для визначення мови
    def detect_language(text):
        try:
            lang = detect(text)
            return "zh" if lang in ["zh-cn", "zh-tw"] else lang
        except:
            return "unknown"

    # Видаляємо коментарі, що містять лише посилання
    df = df[~df["comment"].astype(str).apply(is_only_link)]
    # Очищуємо коментарі
    df["clean_comment"] = df["comment"].astype(str).apply(clean_text)
    # Визначаємо мову
    df["language"] = df["clean_comment"].apply(detect_language)

    return df


def make_sentiment_analysis(df):
    # Завантажуємо модель і токенайзер
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-roberta-base-sentiment-multilingual")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/xlm-roberta-base-sentiment-multilingual")

    # Функція для аналізу сентименту
    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=1).tolist()[0]

        return {"negative": scores[0], "neutral": scores[1], "positive": scores[2]}

    # Аналізуємо всі коментарі
    df[["negative", "neutral", "positive"]] = df["comment"].apply(lambda x: pd.Series(analyze_sentiment(str(x))))
    # Визначаємо основний настрій кожного коментаря
    dominant_sentiment = df[['negative', 'neutral', 'positive']].idxmax(axis=1)
    df['dominant_sentiment'] = dominant_sentiment
    # Визначаємо довжину коментаря за кількістю слів
    df['word_count'] = df['comment'].astype(str).apply(lambda x: max(1, len(re.findall(r'\b\w+\b', x))))

    return df


def top_liked_comments(df, n=100):
    top_df = df.sort_values(by="likes", ascending=False).reset_index(drop=True).head(n)

    return top_df[['author', 'comment', 'likes', 'dominant_sentiment']]


def show_positive_comments(df, thresh=0.85, n=100):
    emotional_comments_pos = df[df['positive'] > thresh]
    emotional_comments_pos = emotional_comments_pos[['author', 'comment', 'positive']].sort_values(by=['positive'],
                                                                                                   ascending=False)

    return emotional_comments_pos.reset_index(drop=True).head(n)


def show_negative_comments(df, thresh=0.85, n=100):
    emotional_comments_neg = df[df['negative'] > thresh]
    emotional_comments_neg = emotional_comments_neg[['author', 'comment', 'negative']].sort_values(by=['negative'],
                                                                                                   ascending=False)

    return emotional_comments_neg.reset_index(drop=True).head(n)


def count_classes(df):

    return df['dominant_sentiment'].value_counts()

def perform_anova_test(df):
    """
    Виконує ANOVA-тест для перевірки значущості різниці в довжині коментарів між класами настроїв.

    :param df: pandas DataFrame з колонками 'word_count' і 'dominant_sentiment'.
    """

    # Розділення даних за класами настрою
    neg = df[df['dominant_sentiment'] == 'negative']['word_count']
    neu = df[df['dominant_sentiment'] == 'neutral']['word_count']
    pos = df[df['dominant_sentiment'] == 'positive']['word_count']

    # Виконання ANOVA-тесту
    stat, p_value = f_oneway(neg, neu, pos)

    if p_value < 0.05:
        conclusion = "ANOVA is a statistical test that helps determine whether comment length varies depending \
        on sentiment. The results showed that there is a correlation between text length and its emotional tone."
    else:
        conclusion = "ANOVA is a statistical test that helps determine whether comment length varies depending \
        on sentiment. The analysis did not reveal a significant correlation between comment sentiment and its length."

    return conclusion


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_sentiment_analysis(sentiments):
    """
    Побудова пай-чарту та горизонтального бар-чарту для аналізу настроїв.

    :param sentiments: pandas Series з настроями, де індекси - категорії (negative, neutral, positive),
                       а значення - їх кількість.
    """
    if not isinstance(sentiments, pd.Series):
        raise TypeError("Очікується pandas Series з настроями.")

     # Визначення кольорів з прозорістю (0.6 - напівпрозорість)
    color_map = {
        'negative': 'rgba(255, 0, 0, 0.6)',   # Червоний
        'neutral': 'rgba(128, 128, 128, 0.6)',  # Сірий
        'positive': 'rgba(0, 128, 0, 0.6)'     # Зелений
    }
    colors = [color_map.get(sentiment, 'rgba(128, 128, 128, 0.6)') for sentiment in sentiments.index]

    # Обчислюємо відсотки для пай-чарту
    total = sentiments.sum()
    sentiments_percentage = (sentiments / total * 100).round(1)

    # Створення загального графіка
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        column_widths=[0.3, 0.7]  # Робимо бар-чарт ширшим
    )

    # Пай-чарт
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

    # Бар-чарт
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

    # Глобальна легенда для обох графіків
    fig.update_layout(
        # title_text="Sentiment Analysis",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",  # Прозорий фон графіка
        font=dict(family="Arial, sans-serif", size=16, color="black"),  # Стиль тексту
        showlegend=False,
    )

    return fig



def plot_comments_length_analysis(df):
    """
    Створює subplot із двома графіками:
    1. Горизонтальний boxplot для аналізу довжини коментарів залежно від настрою.
    2. Заповнений лінійний графік (area chart) розподілу довжини коментарів для кожного класу настроїв.

    :param df: pandas DataFrame з колонками 'word_count' і 'dominant_sentiment'.
    """

    # Визначення кольорів із прозорістю (0.6 - напівпрозорість)
    color_map = {
        "negative": "rgba(255, 0, 0, 0.6)",   # Червоний
        "neutral": "rgba(128, 128, 128, 0.6)", # Сірий
        "positive": "rgba(0, 128, 0, 0.6)"     # Зелений
    }

    # Створення subplot із двома рядами (графіки один під одним)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],  # Пропорції висоти
        shared_xaxes=True,  # Спільна вісь X для обох графіків
        vertical_spacing=0.05
    )

    # Горизонтальний boxplot
    boxplot = px.box(
        df, y="dominant_sentiment", x="word_count",
        color="dominant_sentiment",
        color_discrete_map=color_map,
        labels={"dominant_sentiment": "Sentiment", "word_count": "Comment Length (words)"},
        orientation="h"
    )

    # Видаляємо hover та додаємо сліди в subplot
    for trace in boxplot["data"]:
        trace.update(hoverinfo="skip", hovertemplate='', marker=dict(size=12))  # Прибираємо hover
        fig.add_trace(trace, row=1, col=1)

    # Підготовка даних для area chart
    length_counts = df.groupby(["dominant_sentiment", "word_count"]).size().reset_index(name="comment_count")

    # Заповнений графік (area chart)
    area_chart = px.area(
        length_counts, x="word_count", y="comment_count", color="dominant_sentiment",
        labels={"word_count": "Comment Length (words)", "comment_count": "Number of Comments"},
        color_discrete_map=color_map
    )

    # Видаляємо hover та додаємо сліди в subplot
    for trace in area_chart["data"]:
        trace.update(hovertemplate="Comment Length (words): %{x}")
        fig.add_trace(trace, row=2, col=1)

    # Визначення меж осі X
    min_word_count = df[df["word_count"] > 0]["word_count"].min()  # Найменше значення > 0
    max_word_count = df["word_count"].max() + 5  # Трішки більше максимального значення

    # Оновлення стилю графіка
    fig.update_layout(
        template="plotly_white",
        height=700,
        plot_bgcolor="rgba(0,0,0,0)",  # Прозорий фон графіка
        font=dict(family="Arial, sans-serif", size=16, color="black"),  # Стиль тексту
        showlegend=False,  # Прибираємо легенду
        xaxis=dict(
            dtick=50,  # Крок 20
            range=[min_word_count, max_word_count]  # Початок не з 0
        ),
        xaxis2=dict(
            title="Comment Length (words)",  # Назва X осі (для area chart)
            tickangle=-20,  # Поворот підписів осі X
            dtick=50,  # Крок 20
            range=[min_word_count, max_word_count]  # Початок не з 0
        ),
        yaxis2=dict(
            title="Number of Comments",
        )
    )

    return fig