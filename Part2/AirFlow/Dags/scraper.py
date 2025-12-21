import os
import pandas as pd
from airflow import Dataset
from googleapiclient.discovery import build


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DATASET_PATH = Dataset(f"{BASE_DIR}/data/video_data.csv")
COMMENTS_DATASET_PATH = Dataset(f"{BASE_DIR}/data/comment_data.csv")

PP_VIDEO_DATASET_PATH = Dataset(f"{BASE_DIR}/data/pp_video_data.csv")
PP_COMMENTS_DATASET_PATH = Dataset(f"{BASE_DIR}/data/pp_comment_data.csv")


# =========================
# YouTube API config
# =========================
API_KEY = "AIzaSyAxs7uzwcGOqJftvR_p9gqZ2nT6KmwtKj4"
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

CHANNEL_IDS = [
    "UCoOae5nYA7VqaXzerajD0lg",
    # other channels
]

PLAYLIST_ID = "UUoOae5nYA7VqaXzerajD0lg"

youtube = build(
    API_SERVICE_NAME,
    API_VERSION,
    developerKey=API_KEY
)


# =========================
# Helper functions
# =========================
def get_video_ids(youtube, playlist_id):
    """Retrieve all video IDs from a playlist"""
    video_ids = []

    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults=50,
    )
    response = request.execute()

    while True:
        for item in response["items"]:
            video_ids.append(item["contentDetails"]["videoId"])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token,
        )
        response = request.execute()

    return video_ids


def get_video_details(youtube, video_ids):
    """Retrieve metadata for each video"""
    all_videos = []

    stats_to_keep = {
        "snippet": ["channelTitle", "title", "description", "tags", "publishedAt"],
        "statistics": ["viewCount", "likeCount", "favouriteCount", "commentCount"],
        "contentDetails": ["duration", "definition", "caption"],
    }

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids[i:i + 50]),
        )
        response = request.execute()

        for video in response["items"]:
            video_info = {"video_id": video["id"]}

            for section, fields in stats_to_keep.items():
                for field in fields:
                    video_info[field] = video.get(section, {}).get(field)

            all_videos.append(video_info)

    return pd.DataFrame(all_videos)


def get_comments_in_videos(youtube, video_ids):
    """Retrieve top comments for each video"""
    all_comments = []

    for video_id in video_ids:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
            )
            response = request.execute()

            comments = [
                item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                for item in response["items"][:10]
            ]

            all_comments.append({
                "video_id": video_id,
                "comments": comments,
            })

        except Exception:
            print(f"Comments disabled or unavailable for video {video_id}")

    return pd.DataFrame(all_comments)


# =========================
# Main scraper
# =========================
def scrap_main():
    print("Getting video IDs...")
    video_ids = get_video_ids(youtube, PLAYLIST_ID)
    print(f"{len(video_ids)} video IDs retrieved")

    print("Getting video details...")
    video_df = get_video_details(youtube, video_ids)

    print("Getting comments...")
    comments_df = get_comments_in_videos(youtube, video_ids)

    video_df.to_csv(VIDEO_DATASET_PATH, index=False)
    comments_df.to_csv(COMMENTS_DATASET_PATH, index=False)

    print("Video Data Sample:")
    print(video_df.head())

    print("Comments Data Sample:")
    print(comments_df.head())


# =========================
# Local execution
# =========================
if __name__ == "__main__":
    scrap_main()
