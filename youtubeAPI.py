import pandas as pd
from googleapiclient.discovery import build
import re


def get_comments(api_key, video_url):

    # Виділення відео id з посилання
    def extract_video_id(url):
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        return match.group(1) if match else None

    video_id = extract_video_id(video_url)

    if not video_id:
        print("Некоректне посилання. Будь ласка, введіть коректне посилання на відео YouTube.")
        return None  # Повертаємо None, щоб уникнути помилки

    # Ініціалізація YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)

    def get_all_comments(vid_id):
        comments = []
        next_page_token = None

        while True:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=vid_id,
                textFormat="plainText",
                maxResults=100,  # Максимально доступна кількість коментарів за запит
                pageToken=next_page_token  # Завантажуємо наступну сторінку
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": snippet["authorDisplayName"],
                    "comment": snippet["textDisplay"],
                    "date": snippet["publishedAt"],
                    "likes": snippet["likeCount"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # Вихід, якщо більше немає сторінок

        return comments

    # Отримання всіх коментарів
    comments_data = get_all_comments(video_id)

    try:
        comments_data = get_all_comments(video_id)
        return pd.DataFrame(comments_data)
    except Exception as e:
        print(f"Помилка при отриманні коментарів: {str(e)}")
        return None

    # return pd.DataFrame(comments_data)
