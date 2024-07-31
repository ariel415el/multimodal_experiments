import os
from collections import defaultdict

import clip
import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from dimensionality_recuction import pca
from text_synthesis.gpt import templates

import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'ViT-B-32'
    pretrained_datset = 'laion2b_s34b_b79k'
    classes = ['cat', 'dog', 'truck', 'car']

    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")
    os.makedirs(outputs_dir, exist_ok=True)

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=device)
    model.preprocess = preprocess

    features = defaultdict(list)

    for template in tqdm(templates):
        for cls in classes:
            with torch.no_grad():
                text_tokens = clip.tokenize(template.replace('{object}', cls)).to(device)
                text_feature = model.encode_text(text_tokens).float()
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                features[cls].append(text_feature.cpu().numpy()[0])

    PCs, _, mean = pca(np.concatenate(list(features.values())))

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(classes))]
    fig = plt.figure(1, figsize=(8, 4))
    ax = fig.add_subplot(111)
    for i, cls in enumerate(classes):
        text_embeddings = (np.array(features[cls]) - mean) @ PCs
        plt.scatter(text_embeddings[:, 0], text_embeddings[:, 1], s=5, alpha=0.5, color=colors[i], label=cls)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("text_variation.png")
    plt.show()
    plt.clf()