from airflow import Dataset
import os
from transformers import pipeline
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


VIDEO_DATASET_PATH = Dataset(
    f"{os.path.dirname(os.path.abspath(__file__))}/data/video_data.csv"
)
COMMENTS_DATASET_PATH = Dataset(
    f"{os.path.dirname(os.path.abspath(__file__))}/data/comment_data.csv"
)
PP_VIDEO_DATASET_PATH = Dataset(
    f"{os.path.dirname(os.path.abspath(__file__))}/data/pp_video_data.csv"
)
PP_COMMENTS_DATASET_PATH = Dataset(
    f"{os.path.dirname(os.path.abspath(__file__))}/data/pp_comment_data.csv"
)


ANALYZER = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def calculate_video_sentiment(comments_list):
    if not comments_list or len(comments_list) == 0:
        return 0.5

    cleaned_comments = [str(c)[:512] for c in comments_list]

    try:
        results = ANALYZER(cleaned_comments)

        positive_count = sum(1 for res in results if res["label"] == "POSITIVE")
        sentiment_ratio = positive_count / len(comments_list)
        return sentiment_ratio
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.5


def modeling_main():
    print("Began Modeling")

    try:
        video_path = PP_VIDEO_DATASET_PATH.uri
        comment_path = PP_COMMENTS_DATASET_PATH.uri
    except AttributeError:

        video_path = "data/pp_video_data.csv"
        comment_path = "data/pp_comment_data.csv"

    video_df = pd.read_csv(video_path)
    comments_df = pd.read_csv(comment_path)

    comments_df["comments"] = comments_df["comments"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    comments_df["sentiment_score"] = comments_df["comments"].apply(
        calculate_video_sentiment
    )

    merged_df = pd.merge(
        video_df,
        comments_df[["video_id", "sentiment_score"]],
        on="video_id",
        how="inner",
    )

    merged_df = merged_df.dropna(subset=["likeCount", "viewCount", "durationSecs"])

    X_numeric = merged_df[
        ["viewCount", "durationSecs", "tagCount", "sentiment_score"]
    ].reset_index(drop=True)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    day_encoded = encoder.fit_transform(merged_df[["pushblishDayName"]])
    day_encoded_df = pd.DataFrame(
        day_encoded, columns=encoder.get_feature_names_out(["pushblishDayName"])
    )

    X = pd.concat([X_numeric, day_encoded_df], axis=1)
    y = merged_df["likeCount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Da forst")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("-" * 30)
    print("Model Performance Results:")
    print(f"MSE: {mse:.2f}")
    print(f"R^2 Score: {r2:.4f}")
    print("-" * 30)

    importances = model.feature_importances_
    feature_names = X.columns
    print("Feature Importances:")
    for name, imp in sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    ):
        print(f"{name}: {imp:.4f}")

    print("Done Modeling")


if __name__ == "__main__":
    modeling_main()
