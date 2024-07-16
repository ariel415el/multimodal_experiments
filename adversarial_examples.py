# %%
import os
import open_clip
import torch

import clip
from utils import get_clip_features, get_dataset
from dimensionality_recuction import pca
import numpy as np
import torchvision

class ClipClassifier:
    def __init__(self, model, dataset):
        text_tokens = clip.tokenize([f"This is a photo of a {label}" for label in dataset.classes]).cuda()
        with torch.no_grad():
            label_features = model.encode_text(text_tokens).float()
            label_features /= label_features.norm(dim=-1, keepdim=True)
            label_features = label_features
        
        self.labe_features = label_features
    
    def predict(self, x, return_probs=False):
       logits = x @ self.labe_features.T
       if return_probs:
          return logits
       return logits.argmax(1).item()


def optimize_latent(model, dataset, classifier):
    text_features, image_features, labels = get_clip_features(model, dataset, label_map, device,
                                                              os.path.join(outputs_dir, f"{dataset_name}_features"))
    
    PCs, _, mean = pca(np.concatenate((text_features, image_features)))
    image_embeddings = np.dot(image_features - mean, PCs)
    
    pc = 0
    idx = 0
    y_gt = labels[idx]
    y_goal = torch.tensor([1]).cuda()
    org_x = torch.from_numpy(image_embeddings[idx]).cuda()
    mask = torch.ones(len(org_x)).to(bool)
    mask[:10] = False
    x = org_x.clone()
    for i in range(10):
    #   with torch.no_grad():
    #     x[mask] = org_x[mask]
        x.requires_grad = True

        logits = classifier.predict(x.unsqueeze(0), return_probs=True)

        loss = torch.nn.functional.nll_loss(logits, y_goal)
        loss.backward()
        with torch.no_grad():
            # x.grad.zero_()
            x = x - x.grad.data*0.1
        print(f"iter: {i}, gt {y_gt.item()}, pred {logits.argmax(1).item()}, loss: {loss.item()}")

    print(f"Final budget: {torch.norm(org_x - x)}")

def optimize_image(model, dataset, classifier):
    img, y_gt = dataset[0]
    y_goal = torch.tensor([0]).cuda()
    torchvision.utils.save_image(img, "origin.png", normalize=True)
    img = img.clone().cuda().unsqueeze(0)
    for i in range(300):
        img.requires_grad = True
        features = model.encode_image(img).float()
        features = features / features.norm(dim=-1, keepdim=True)
        logits = classifier.predict(features, return_probs=True)
        loss = torch.nn.functional.nll_loss(logits, y_goal)
        loss.backward()
        # with torch.no_grad():
            # x.grad.zero_()
        grad = img.grad.data.clone()
        with torch.no_grad():
            img = img - grad* 1
        print(f"iter: {i}, gt {y_gt}, pred {logits.argmax(1).item()}, loss: {loss.item()}")

    torchvision.utils.save_image(img, "fiinal.png", normalize=True)




# @title settings
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'ViT-B-32'
pretrained_datset = 'laion2b_s34b_b79k'
dataset_name = "STL10"
outputs_dir = os.path.join("outputs", f"{model_name}({pretrained_datset})")

model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_datset, 
                                                             device=device, cache_dir='/cs/labs/yweiss/ariel1/big_files')
model.preprocess = preprocess
# %%
dataset, label_map = get_dataset(dataset_name, model.preprocess)
classifier = ClipClassifier(model, dataset)
# %%

# optimize_latent(model, dataset, classifier)

# %%
optimize_image(model, dataset, classifier)

# %%
