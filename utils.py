import os

import torch
import torchvision
import clip
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from dimensionality_recuction import *


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,images_dir, captions_file, transform=None):
        self.images_dir = images_dir
        self.data = defaultdict(list)
        for line in open(captions_file).readlines():
            line = line.strip().replace(" .", ".")
            i = line.find("#")
            img_name, text = line[:i], line[i:]
            if os.path.exists(os.path.join(self.images_dir, img_name)):
                text = text[2:].strip()
                self.data[img_name].append(text)
        self.data = list(self.data.items())
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, texts = self.data[idx]
        img = Image.open(os.path.join(self.images_dir, img_name))
        img = self.transform(img)
        return img, texts[0]


def extract_clip_embedding_from_dataset(model, dataset, label_map=None, device=torch.device('cpu')):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    all_text_features = []
    all_image_features = []
    all_labels = []
    pbar = tqdm(dataloader)
    pbar.set_description("Extracting features from data")
    for images, labels in pbar:
        if label_map is not None:
            text_descriptions = map(label_map, labels)
        else:
            text_descriptions = list(labels)
        text_tokens = clip.tokenize(text_descriptions).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            image_features = model.encode_image(images.to(device)).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        all_text_features += [text_features.cpu().numpy()]
        all_image_features += [image_features.cpu().numpy()]
        all_labels += [labels]

    all_text_features = np.concatenate(all_text_features)
    all_image_features = np.concatenate(all_image_features)
    all_labels = np.concatenate(all_labels)

    return all_text_features, all_image_features, all_labels


def get_clip_features(model, dataset, label_map, device, output_dir):
    if not os.path.exists(os.path.join(output_dir, 'text_features.npy')):
        os.makedirs(output_dir, exist_ok=True)
        text_features, image_features, labels = extract_clip_embedding_from_dataset(model, dataset, label_map, device)
        np.save(os.path.join(output_dir, 'text_features.npy'), text_features)
        np.save(os.path.join(output_dir, 'image_features.npy'), image_features)
        np.save(os.path.join(output_dir, 'labels.npy'), labels)
    else:
        text_features = np.load(os.path.join(output_dir, 'text_features.npy'))
        image_features = np.load(os.path.join(output_dir, 'image_features.npy'))
        labels = np.load(os.path.join(output_dir, 'labels.npy'))
    return text_features, image_features, labels


def get_dataset(dataset_name, preprocess, data_root='/cs/labs/yweiss/ariel1/data'):
    # data_root = '/mnt/storage_ssd/datasets'
    if dataset_name == 'Flickr8k':
        dataset = FlickrDataset(os.path.join(data_root, "Flickr8k/images"),
                                captions_file=os.path.join(data_root, 'Flickr8k/Flickr8k_text/Flickr8k.token.txt'),
                                transform=preprocess)
        label_map = None
    else:
        dataset = torchvision.datasets.STL10(os.path.join(data_root, "STL10"), transform=preprocess, download=True)
        label_map = lambda x: f"This is a photo of a {dataset.classes[x]}"

    return dataset, label_map
