from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.metrics import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.utils import load_all_events, POSSIBLE_CATEGORIES


def fit_kmeans_to_event_embeddings(embeddings: np.array) -> KMeans:
    k_means = KMeans(n_clusters=len(POSSIBLE_CATEGORIES))
    k_means.fit(embeddings)
    return k_means


def get_category_to_label_counts(
    categories: List[str], labels: List[int]
) -> Dict[str, Dict[int, int]]:
    category_to_label_counts: Dict[str, Dict[int, int]] = {}
    for category in POSSIBLE_CATEGORIES:
        category_to_label_counts[category] = defaultdict(int)

    for category, label in zip(categories, labels):
        category_to_label_counts[category][label] += 1

    return category_to_label_counts


def get_label_to_category_counts(
    categories: List[str], labels: List[int]
) -> Dict[int, Dict[str, int]]:
    label_to_category_counts: Dict[int, Dict[str, int]] = {}
    for label in range(len(POSSIBLE_CATEGORIES)):
        label_to_category_counts[label] = defaultdict(int)

    for category, label in zip(categories, labels):
        label_to_category_counts[label][category] += 1

    return label_to_category_counts


def decompose_embeddings(
    embedding_matrix: np.array,
    pca_step_embedding: int = 25,
    tsne_final_embedding: int = 2,
) -> np.array:
    pca_reduced_matrix = PCA(n_components=pca_step_embedding).fit_transform(
        embedding_matrix
    )
    return TSNE(n_components=tsne_final_embedding).fit_transform(pca_reduced_matrix)


def generate_visualization_2d(
    decomposed_embeddings: np.array,
    labels: List[int],
    categories: List[str],
    type_: str  # 'label' or 'category'
) -> None:
    df = pd.DataFrame(decomposed_embeddings, columns=["TSNE-1", "TSNE-2"])
    df["label"] = labels
    df["category"] = categories
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="TSNE-1",
        y="TSNE-2",
        hue=type_,
        data=df,
        palette=sns.color_palette("hls", len(POSSIBLE_CATEGORIES)),
        alpha=0.3,
    )
    plt.show()


def get_metrics(categories: List[str], labels: List[int]):
    all_labels_result: Dict[int, float] = {}
    label_to_category_count = get_label_to_category_counts(categories, labels)
    for label in label_to_category_count:
        category_count = label_to_category_count[label]
        all_labels_result[label] = entropy(category_count)
    return all_labels_result


if __name__ == "__main__":
    events = load_all_events()
    embeddings_as_np = np.array([event.embedding for event in events])
    k_means = fit_kmeans_to_event_embeddings(embeddings_as_np)
    event_with_labels = [(evt, label) for evt, label in zip(events, k_means.labels_)]
    decomposed_embeddings = decompose_embeddings(embeddings_as_np)

    categories = [event.category for event, _ in event_with_labels]
    labels = [label for _, label in event_with_labels]

    generate_visualization_2d(
        decomposed_embeddings=decomposed_embeddings,
        labels=labels,
        categories=categories,
        type_="label"
    )
    generate_visualization_2d(
        decomposed_embeddings=decomposed_embeddings,
        labels=labels,
        categories=categories,
        type_="category"
    )
    print(get_metrics(categories, labels))
