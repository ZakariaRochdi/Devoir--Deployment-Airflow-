import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from airflow import Dataset
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

# =========================
# NLTK resources
# =========================
nltk.download("stopwords")
nltk.download("punkt")

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DATASET_PATH = Dataset(f"{BASE_DIR}/data/video_data.csv")
COMMENTS_DATASET_PATH = Dataset(f"{BASE_DIR}/data/comment_data.csv")
PP_VIDEO_DATASET_PATH = Dataset(f"{BASE_DIR}/data/pp_video_data.csv")
PP_COMMENTS_DATASET_PATH = Dataset(f"{BASE_DIR}/data/pp_comment_data.csv")

PLOTS_PATH = f"{BASE_DIR}/plots"


# =========================
# Utils
# =========================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_thousands(ax):
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x/1000):,}K")
    )


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, filename))
    plt.clf()


# =========================
# Main plotting function
# =========================
def plots_main():
    print("Saving plots...")
    ensure_dir(PLOTS_PATH)

    video_df = pd.read_csv(PP_VIDEO_DATASET_PATH)

    # -------- Top 9 videos (descending) --------
    ax = sns.barplot(
        data=video_df.sort_values("viewCount", ascending=False).head(9),
        x="title",
        y="viewCount",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    format_thousands(ax)
    save_plot("top_9_videos_descending.png")

    # -------- Top 9 videos (ascending) --------
    ax = sns.barplot(
        data=video_df.sort_values("viewCount", ascending=True).head(9),
        x="title",
        y="viewCount",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    format_thousands(ax)
    save_plot("top_9_videos_ascending.png")

    # -------- Violin plot --------
    sns.violinplot(data=video_df, x="channelTitle", y="viewCount")
    save_plot("violin_plot_channel_vs_viewCount.png")

    # -------- Scatter plots --------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=video_df, x="commentCount", y="viewCount", ax=axes[0])
    sns.scatterplot(data=video_df, x="likeCount", y="viewCount", ax=axes[1])
    save_plot("scatter_plots_comment_like_vs_viewCount.png")

    # -------- Histogram --------
    sns.histplot(data=video_df, x="durationSecs", bins=30)
    save_plot("histogram_durationSecs.png")

    # -------- WordCloud --------
    stop_words = set(stopwords.words("english"))

    video_df["title_no_stopwords"] = video_df["title"].astype(str).apply(
        lambda title: [w for w in title.split() if w.lower() not in stop_words]
    )

    words = " ".join(
        word for title in video_df["title_no_stopwords"] for word in title
    )

    wordcloud = WordCloud(
        width=2000,
        height=1000,
        background_color="black",
        colormap="viridis",
        random_state=1,
        collocations=False,
    ).generate(words)

    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud)
    plt.axis("off")
    save_plot("wordcloud_video_titles.png")

    # -------- Day of week distribution --------
    weekdays = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    day_df = video_df["pushblishDayName"].value_counts().reindex(weekdays)
    day_df.reset_index().plot.bar(x="index", y="pushblishDayName", rot=0)
    save_plot("day_of_week_distribution.png")

    print("Saved plots ✔")


# =========================
# Run
# =========================
if __name__ == "__main__":
    plots_main()
