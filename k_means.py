from collections import defaultdict
from typing import Tuple, List, Dict, DefaultDict

from sklearn.cluster import KMeans

from src.utils import load_all_events, POSSIBLE_CATEGORIES, Event
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def fit_kmeans_to_data() -> List[Tuple[Event, int]]:
    events = load_all_events()
    events = [event for i, event in enumerate(events) if i % 2 == 0]
    embeddings = [event.embedding for event in events]
    embeddings_as_np = np.array(embeddings)
    k_means = KMeans(n_clusters=len(POSSIBLE_CATEGORIES))
    k_means.fit(embeddings_as_np)
    labels = k_means.labels_
    return [
        (evt, label)
        for evt, label in zip(events, labels)
    ], embeddings_as_np


def decompose_embeddings(embedding_matrix: np.array) -> np.array:
    pca = PCA(n_components=25)
    tsne = TSNE(n_components=2)
    reduced_matrix = pca.fit_transform(embedding_matrix)
    ready_for_visualization = tsne.fit_transform(reduced_matrix)
    return ready_for_visualization


if __name__ == '__main__':
    data, embs = fit_kmeans_to_data()
    decomposed_embeddings = decompose_embeddings(embs)
    labels = [[label] for _, label in data]
    categories = [[POSSIBLE_CATEGORIES.index(event.category)] for event, _ in data]
    labels_as_np = np.array(labels)
    categories_as_np = np.array(categories)
    concat = np.concatenate(
        (decomposed_embeddings, labels_as_np, categories), axis=1)
    df = pd.DataFrame(concat, columns=["pca1", "pca2", "label", "category"])
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="label",
        data=df,
        palette=sns.color_palette("hls", len(POSSIBLE_CATEGORIES)),
        legend="full",
        alpha=0.3
    )
    plt.show()
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="category",
        data=df,
        palette=sns.color_palette("hls", len(POSSIBLE_CATEGORIES)),
        legend="full",
        alpha=0.3
    )
    plt.show()

