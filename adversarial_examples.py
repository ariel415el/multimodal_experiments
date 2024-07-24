import os
import open_clip
import torch
import numpy as np
from matplotlib import pyplot as plt

from dimensionality_recuction import pca
from utils import get_dataset, get_clip_features
from classify_images import ClipClassifier

import clip

from torch import optim
from PIL import Image
import cv2
import torchvision


def optimize_image(model, img, target_label, classifier, norm_coeff=0.01, lr=0.1, n_steps=100, pc_slice=None, verbose=False):
    y_goal = torch.tensor([target_label]).cuda()
    img = img.clone().cuda().unsqueeze(0)
    residue = torch.randn_like(img) * 1e-4
    residue.requires_grad = True
    optimizer = optim.Adam([residue], lr=lr)
    best_img = None
    best_norm = np.inf
    for i in range(n_steps):
        features = model.encode_image(img + residue).float()
        features = features / features.norm(dim=-1, keepdim=True)
        if i == 0:
            org_pred = classifier.predict(features)

        logits = classifier.predict(features, return_probs=True)
        # logits = classifier.predict(features, return_probs=True, slice=slice(1, 2))

        residue_norm = torch.norm(residue.view(len(residue), -1), dim=1).mean()

        loss = torch.nn.functional.nll_loss(logits, y_goal) + norm_coeff * residue_norm
        loss.backward()
        # residue.grad = residue.grad.sign()
        optimizer.step()
        residue.grad.zero_()
        
        # with torch.no_grad():
        #     residue = torch.clamp(residue, -0.1, 0.1)
        #     # residue = residue / residue.norm() * 5
        # residue.requires_grad = True
        with torch.no_grad():
            features = model.encode_image(img + residue).float()
            features = features / features.norm(dim=-1, keepdim=True)
            cur_pred = classifier.predict(features)
        if residue_norm.item() < best_norm and cur_pred != org_pred:
            best_img = (img + residue).detach()
            best_norm = residue_norm.item()


        if verbose or i  == n_steps - 1:
            print(f"iter: {i}, loss: {loss}, best norm: {best_norm}, norm: {residue_norm}, pred {org_pred}->{cur_pred}")

    return best_img[0], best_norm


def test_average_adversarial_margin(model, dataset, classifier, n_images=10):
    all_inputs = []
    all_advs = []
    all_norms = []
    for i in range(n_images):
        input, gt_label = dataset[i]
        target_label = np.random.choice([x for x in range(len(dataset.classes)) if x != gt_label])
        adv, adv_residue_norm = optimize_image(model, input, target_label, classifier, norm_coeff=0.01, lr=0.02, n_steps=200, pc_slice=None, verbose=False)
        # assert adv is not None
        if adv is not None:
            all_norms.append(adv_residue_norm)
            all_inputs.append(input)
            all_advs.append(adv)
    return all_inputs, all_advs, all_norms


def plot(model, PCs, mean, label_features, image_features, dataset_labels, class_names, inputs, advs, save_path):

    cmap = plt.get_cmap('tab10')
    n = len(inputs) + 2
    colors = [cmap(i / n) for i in range(n)]
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111)
    pc_1 = 0
    pc_2 = 1

    # plot data and labels
    label_pcs = (label_features - mean) @ PCs
    image_pcs = (image_features - mean) @ PCs
    for label in np.unique(dataset_labels):
        ax.scatter(label_pcs[label, pc_1], label_pcs[label, pc_2],
                   color=colors[-label - 1], label=class_names[label], s=5, alpha=0.5)
        idx = dataset_labels == label
        ax.scatter(image_pcs[idx, pc_1], image_pcs[idx, pc_2],
                   color=colors[-label - 1], s=5, alpha=0.5)
    
    for name, arr, mar in [('inputs', inputs, 'x'),
                           ('advs', advs, 'o')]:
        for i, img in enumerate(arr):
            with torch.no_grad():
                f = model.encode_image(img.cuda().unsqueeze(0)).float().cpu()
                f /= f.norm(dim=-1, keepdim=True)
            pcs = (f - mean) @ PCs
            ax.scatter(pcs[:, pc_1], pcs[:, pc_2],
                    color=colors[i], s=60, alpha=0.75, marker=mar, label=name)


    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.clf()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = 'ViT-B-32'
    # pretrained_datset = 'laion2b_s34b_b79k'
    model_name = 'RN50'
    pretrained_datset = 'openai'

    dataset_name = "STL10"
    outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")

    # Load_model
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_datset, device=device, cache_dir='/cs/labs/yweiss/ariel1/big_files')
    model.preprocess = preprocess

    dataset, label_map = get_dataset(dataset_name, model.preprocess, data_root='/cs/labs/yweiss/ariel1/data', restrict_to_classes=['cat', 'dog'])


    # Extract CLIP features
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                            os.path.join(outputs_dir, f"{dataset_name}_features"))

    # PCA
    PCs, _, mean = pca(np.concatenate((text_features, image_features)))
    PCs, mean = torch.from_numpy(PCs), torch.from_numpy(mean)


    # Shift text labels into class image means
    classifier = ClipClassifier(model, dataset)
    
    inputs, advs, all_norms = test_average_adversarial_margin(model, dataset, classifier, n_images=3)
    plot(model, PCs, mean, classifier.label_features.cpu(), image_features, labels, dataset.classes, inputs, advs, "With_gap.png")

    gap_mean, gap_std = np.mean(all_norms), np.std(all_norms)
    print(f"With gap: avg adversarial norm {gap_mean}+-{gap_std}")

    classifier.label_features[0] = image_features[labels==0].mean(0)
    classifier.label_features[1] = image_features[labels==1].mean(0)

    all_imgs, all_advs, all_norms = test_average_adversarial_margin(model, dataset, classifier, n_images=10)
    no_gap_mean, no_gap_std = np.mean(all_norms), np.std(all_norms)

    print(f"No gap: avg adversarial norm {no_gap_mean}+-{no_gap_std}")
    

main()
