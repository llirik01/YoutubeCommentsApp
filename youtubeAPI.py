import pandas as pd
from googleapiclient.discovery import build
import re


def get_comments(api_key, video_url):
    """
    Fetch comments from a YouTube video.

    :param api_key: str - YouTube Data API key
    :param video_url: str - URL of the YouTube video
    :return: DataFrame or None - Pandas DataFrame containing comments or None if an error occurs
    """

    def extract_video_id(url):
        """
        Extracts the video ID from a YouTube URL.

        :param url: str - YouTube video URL
        :return: str or None - Video ID if found, otherwise None
        """
        pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
        match = re.search(pattern, url)
        return match.group(1) if match else None

    video_id = extract_video_id(video_url)

    if not video_id:
        print("Invalid URL. Please, enter a valid YouTube video URL.")
        return None  # Return None to avoid errors

    # Initialize YouTube API
    youtube = build("youtube", "v3", developerKey=api_key)

    def get_all_comments(vid_id):
        """
        Retrieve all comments from a YouTube video.

        :param vid_id: str - YouTube video ID
        :return: list - List of dictionaries containing comment details
        """
        comments = []
        next_page_token = None

        while True:
            # Request to fetch comments
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=vid_id,
                textFormat="plainText",
                maxResults=100,  # Maximum available comments per request
                pageToken=next_page_token  # Load the next page
            )
            response = request.execute()

            # Extract relevant comment details
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": snippet["authorDisplayName"],
                    "comment": snippet["textDisplay"],
                    "date": snippet["publishedAt"],
                    "likes": snippet["likeCount"]
                })

            # Check for next page token
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # Exit loop if no more pages

        return comments

    # Fetch all comments
    try:
        comments_data = get_all_comments(video_id)
        return pd.DataFrame(comments_data)
    except Exception as e:
        print(f"Error retrieving comments: {str(e)}")
        return None

