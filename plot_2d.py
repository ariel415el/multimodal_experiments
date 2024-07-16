import os
from matplotlib import pyplot as plt

from dimensionality_recuction import get_TSNE_embeddings, get_pcas
from utils import get_dataset, get_clip_features


def plot_pca_embeddings(text_embeddings, image_embeddings, title, outputs_dir):
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(111)
    for name, embeddings, color, marker in [("text", text_embeddings, 'blue', '^'),
                                            ("image", image_embeddings, 'red', 'o')]:
        ax.scatter(embeddings[:, 0], embeddings[:, 1],
                   color=color, label=name, marker=marker, s=5, alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(outputs_dir, f"{title}.png"))
    plt.clf()


def plot_per_class_embeddings(text_embeddings, image_embeddings, all_labels, class_names, title, outputs_dir):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    for label in range(10):
        idx = all_labels == label
        ax.scatter(text_embeddings[idx, 0], text_embeddings[idx, 1],
                   color=colors[label], label=class_names[label], s=2, alpha=0.5)
        ax.scatter(image_embeddings[idx, 0], image_embeddings[idx, 1],
                   color=colors[label], s=2, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.savefig(os.path.join(outputs_dir, f"{title}.png"))


def plot_2d(model, dataset_name, pca_mode, outputs_dir, device):
    dataset, label_map = get_dataset(dataset_name, model.preprocess)

    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    text_embeddings, image_embeddings = get_pcas(text_features, image_features, pca_mode=pca_mode)
    # text_embeddings, image_embeddings = get_TSNE_embeddings(text_features, image_features)

    title = f"{dataset_name}_{pca_mode}"
    if dataset_name == 'Flickr8k':
        plot_pca_embeddings(text_embeddings, image_embeddings, title, outputs_dir)
    else:
        plot_per_class_embeddings(text_embeddings, image_embeddings, labels, dataset.classes, title, outputs_dir)