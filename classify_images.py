import os

import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import clip
from tqdm import tqdm

from dimensionality_recuction import pca, get_pcas
from plot_2d import plot_per_class_embeddings
from text_synthesis.gpt import templates
from utils import get_dataset, get_clip_features


class ClipClassifier:
    def __init__(self, model, reference_texts, device):
        self.device = device
        text_tokens = clip.tokenize(reference_texts).to(self.device)
        with torch.no_grad():
            reference_features = model.encode_text(text_tokens).float()
            reference_features /= reference_features.norm(dim=-1, keepdim=True)

        self.reference_features = reference_features
    
    def predict(self, features, return_probs=False, **kwargs):
        logits = (features.to(self.device) @ self.reference_features.T).softmax(-1)
        if return_probs:
            return logits
        return logits.argmax(1)

def classify_bath(model, dataset, classifier, batch_size, return_probs=False):
    images, labels = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=True)))
    with torch.no_grad():
        image_features = model.encode_image(images.to(device)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    text_probs = 100.0 * classifier.predict(image_features, return_probs=True)
    if return_probs:
        return text_probs
    else:
        accuracy = (text_probs.argmax(-1).cpu() == labels).float().mean()
        return accuracy


def plot_classification(model, dataset, n_images=100, outputs_dir='outputs', device=torch.device('cpu')):
    classifier = ClipClassifier(model, [f"This is a photo of a {label}" for label in dataset.classes], device)

    images, labels = next(iter(DataLoader(dataset, batch_size=n_images, shuffle=True)))

    with torch.no_grad():
        image_features = model.encode_image(images.to(device)).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

    text_probs = classify_bath(model, dataset, classifier, n_images, return_probs=True)
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


def compare_prompts(n_images):
    print(np.unique(templates).tolist())
    for template in templates[:15]:
        classifier = ClipClassifier(model, [template.replace('{object}', label) for label in dataset.classes], device)

        images, labels = next(iter(DataLoader(dataset, batch_size=n_images, shuffle=True)))

        with torch.no_grad():
            image_features = model.encode_image(images.to(device)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)

        text_probs = classifier.predict(image_features).cpu()
        accuracy = (text_probs == labels).float().mean()

        print(f"{template}: Accuracy: {(accuracy*100).int()}%")


def plot_classifier(classifier, image_features, all_labels, PCs, mean, title):
    new_reference_embs = (classifier.reference_features.cpu().numpy() - mean) @ PCs
    image_embeddings = (image_features - mean) @ PCs
    cmap = plt.get_cmap('tab10')
    labels = range(len(classifier.reference_features))
    colors = [cmap(i) for i in labels]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    for label in labels:
        idx = all_labels == label
        ax.scatter(new_reference_embs[label, 0], new_reference_embs[label, 1],
                   color=colors[label], label=dataset.classes[label], s=10, alpha=0.5, marker='x')
        ax.scatter(image_embeddings[idx, 0], image_embeddings[idx, 1],
                   color=colors[label], s=2, alpha=0.5, marker='o')

    # plot linear separator
    a,b = new_reference_embs[0, :2]
    c,d = new_reference_embs[1, :2]
    y = lambda x: (b+d)/2 + (x-(a+c)/2)*(a-c) / (d-b)
    xs =np.array([image_embeddings[:, 0].min(), new_reference_embs[:, 0].max()])
    ylim  = plt.ylim()
    xlim  = plt.xlim()
    plt.plot(xs, y(xs), linestyle='--', color='black', label='classifier')
    plt.ylim(ylim)
    plt.xlim(xlim)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, f"classifier_vis.png"))
    plt.show()
    plt.clf()


def test_reference_shift():
    pc_shift = 0.15
    dataset_name = 'STL10'
    dataset = get_dataset(dataset_name, model.preprocess, data_root, restrict_to_classes=['cat', 'dog'])
    label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"
    text_features, image_features, all_labels = get_clip_features(model, dataset, label_map, device,
                                                                  os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, _, mean = pca(np.concatenate((text_features, image_features)))
    # PCs = PCs[:, :2]

    classifier = ClipClassifier(model, dataset.classes, device)
    classifier.reference_features[1, :] += torch.from_numpy(PCs[:, 0] * pc_shift).to(device)

    logits = (image_features - mean) @ (classifier.reference_features.cpu().numpy() - mean).T
    acc = (logits.argmax(-1) == all_labels).mean()
    print(f"Accuracy: {(acc*100):.1f}%")

    plot_classifier(classifier, image_features, all_labels, PCs, mean, f"shift pc0: ({pc_shift}) Acc: {acc*100:.1f}%")


def classify_pcs(model, outputs_dir='outputs', device=torch.device('cpu')):
    dataset_name = "STL10"
    dataset, label_map = get_dataset(dataset_name, model.preprocess)
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))

    PCs, eigv, mean = pca(np.concatenate((text_features, image_features)))
    text_embeddings = np.dot(text_features - mean, PCs)
    image_embeddings = np.dot(image_features - mean, PCs)
    classifier = ClipClassifier(model, dataset)
    label_embeddings = np.dot(classifier.labe_features.cpu().numpy() - mean, PCs)

    n = text_embeddings.shape[1]
    dropped_top_pcs = np.arange(0, n)
    accuracies_lower = []
    accuracies_top = []
    for i in tqdm(dropped_top_pcs):
        # text_probs = np.(100.0 * image_embeddings[:, i:] @ text_embeddings[:, i:].T).softmax(dim=-1)
        text_probs = image_embeddings[:, i:] @ label_embeddings[:, i:].T
        acc = (text_probs.argmax(-1) == labels).mean()
        accuracies_lower.append(acc)

        text_probs = image_embeddings[:, :i] @ label_embeddings[:, :i].T
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


if __name__ == '__main__':
    cache_dir = '/mnt/storage_ssd/big_files'
    data_root = '/mnt/storage_ssd/datasets'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_dataset = 'laion2b_s34b_b79k'
    model_name = 'ViT-B-32'

    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_dataset})")
    os.makedirs(outputs_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset,
                                                                 device=device, cache_dir=cache_dir)
    model.preprocess = preprocess
    model.eval()

    dataset = get_dataset('STL10', model.preprocess, data_root, restrict_to_classes=None)

    # plot_classification(model, dataset, n_images=100, outputs_dir=outputs_dir, device=device)

    # compare_prompts(n_images=1000)

    test_reference_shift()

    # classify_pcs(model, outputs_dir, device, outputs_dir=outputs_dir, device=device)