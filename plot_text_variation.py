import os
from collections import defaultdict

import clip
import numpy as np
import open_clip
import torch
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm

from dimensionality_recuction import pca
from text_synthesis.gpt import templates

import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'ViT-B-32'
    pretrained_datset = 'laion2b_s34b_b79k'
    classes = ['car', 'dog']

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

    # PCs, _, mean = pca(np.concatenate(list(features.values())))
    PCs, mean = get_multimodal_pcs()


    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(classes))]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)

    embs = {k: (np.array(v) - mean) @ PCs for k, v in features.items()}
    for i, (cls, emb) in enumerate(embs.items()):
        print(f"{cls}: {emb.std(0)[:3]}")
        plt.scatter(emb[:, 0], emb[:, 1], s=15, alpha=0.5, color=colors[i], label=cls)

    for i in range(len(templates)):
        x0 = embs[classes[0]][i, 0]
        y0 = embs[classes[0]][i, 1]
        x1 = embs[classes[1]][i, 0]
        y1 = embs[classes[1]][i, 1]
        # plt.arrow(x0, y0, x1 - x0, y1 - y0, fc='k', ec='k', head_width=0.005)
        plt.plot([x0, x1], [y0, y1], c='k', linewidth=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("text_variation.png")
    plt.show()
    plt.clf()