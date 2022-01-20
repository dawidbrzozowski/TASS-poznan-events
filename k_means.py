from collections import defaultdict
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    labels = [[label] for label in labels]
    categories = [[POSSIBLE_CATEGORIES.index(category)] for category in categories]
    concat = np.concatenate((decomposed_embeddings, labels, categories), axis=1)
    df = pd.DataFrame(concat, columns=["TSNE-1", "TSNE-2", "label", "category"])
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="TSNE-1",
        y="TSNE-2",
        hue=type_,
        data=df,
        palette=sns.color_palette("hls", len(POSSIBLE_CATEGORIES)),
        legend="full",
        alpha=0.3,
    )
    plt.show()


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