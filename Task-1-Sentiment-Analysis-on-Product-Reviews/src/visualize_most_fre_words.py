from wordcloud import WordCloud
import matplotlib.pyplot as plt


def visualize_most_frequent_words(df):
    positive_reviews = df[df["sentiment"] == "positive"]["review"].str.cat(sep=" ")
    negative_reviews = df[df["sentiment"] == "negative"]["review"].str.cat(sep=" ")

    wordcloud_pos = WordCloud(width=800, height=400, background_color="white").generate(
        positive_reviews
    )
    wordcloud_neg = WordCloud(
        width=800, height=400, background_color="black", colormap="Reds"
    ).generate(negative_reviews)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_pos, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Positive Words")

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_neg, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Negative Words")

    plt.tight_layout()
    plt.show()
