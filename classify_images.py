import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import clip
from tqdm import tqdm

from dimensionality_recuction import pca
from utils import get_dataset, get_clip_features


def classify_stl(model, n_images=100, outputs_dir='outputs', device=torch.device('cpu')):
    dataset, label_map = get_dataset("STL10", model.preprocess)
    text_descriptions = [f"This is a photo of a {label}" for label in dataset.classes]
    text_tokens = clip.tokenize(text_descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    dataloader = DataLoader(dataset, batch_size=n_images, shuffle=True)
    images, labels = next(iter(dataloader))

    with torch.no_grad():
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        image_features = model.encode_image(images.to(device)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
    accuracy = (text_probs.argmax(-1).cpu() == labels).float().mean()

    print(f"Accuracy: {(accuracy*100).int()}%")

    plt.figure(figsize=(16, 16))
    plt.title(f"Accuracy: {(accuracy*100).int()}%")
    for i in range(8):
        plt.subplot(4, 4, 2 * i + 1)

        img = images[i]
        img = img - img.min()
        img = img / img.max()
        img = img.permute(1,2,0).cpu().numpy()

        plt.imshow(img)
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [dataset.classes[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(os.path.join(outputs_dir, "classify_STL.png"))
    plt.clf()


def classify_pcs(model, outputs_dir='outputs', device=torch.device('cpu')):
    dataset_name = "STL10"
    dataset, label_map = get_dataset(dataset_name, model.preprocess)
    _, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    text_descriptions = [f"This is a photo of a {label}" for label in dataset.classes]
    text_tokens = clip.tokenize(text_descriptions).cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    PCs, eigv, mean = pca(np.concatenate((text_features, image_features)))
    text_embeddings = np.dot(text_features - mean, PCs)
    image_embeddings = np.dot(image_features - mean, PCs)

    n = text_embeddings.shape[1]
    dropped_top_pcs = np.arange(0, n)
    accuracies_lower = []
    accuracies_top = []
    for i in tqdm(dropped_top_pcs):
        # text_probs = np.(100.0 * image_embeddings[:, i:] @ text_embeddings[:, i:].T).softmax(dim=-1)
        text_probs = image_embeddings[:, i:] @ text_embeddings[:, i:].T
        acc = (text_probs.argmax(-1) == labels).mean()
        accuracies_lower.append(acc)

        text_probs = image_embeddings[:, :i] @ text_embeddings[:, :i].T
        acc = (text_probs.argmax(-1) == labels).mean()
        accuracies_top.append(acc)

    print(f"accuracies_lower: {accuracies_lower}")
    print(f"accuracies_top: {accuracies_top}")
    plt.plot(n - dropped_top_pcs, accuracies_lower, label="low x pcs", color='r')
    plt.plot(dropped_top_pcs, accuracies_top, label="top x pcs", color='b')
    plt.legend()
    plt.xlabel("# PCs")
    plt.ylabel("Aligmnent")
    plt.savefig(os.path.join(outputs_dir, "classify_pcs.png"))
    plt.clf()