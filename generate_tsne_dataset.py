import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.manifold import TSNE
from tqdm import tqdm

from data.loader import get_dataset, get_loader
from models.mmmaml_convnet import MMAMLConvNet3

def plot_tsne(model, dataloader, device, dataset_name):
    model.eval()
    features, labels, domains = [], [], []

    print(f"Extracting features for {dataset_name}...")
    with torch.no_grad():
        for i, (images, targets, groups) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            feats = model.get_features(images)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
            domains.append(groups.numpy())
            if i >= 50: break

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    domains = np.concatenate(domains)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=0, perplexity=30 if dataset_name != 'rimagenet' else 50)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=labels, palette="tab10" if dataset_name != 'rimagenet' else "hls", s=20, legend=None)
    plt.title(f"DDML {dataset_name.upper()} (By Class)")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=domains, palette="viridis", s=20, alpha=0.6, legend=None)
    plt.title(f"DDML {dataset_name.upper()} (By Domain)")

    plt.tight_layout()
    plt.savefig(f"tsne_{dataset_name}_results.png", dpi=300)
    print(f"Saved to tsne_{dataset_name}_results.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'affnist', 'rimagenet'])
    parser.add_argument('--ckpt', type=str, required=True, help='Path to best_weights.pkl')
    args_cmd = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mapping Konfigurasi Dataset
    config = {
        'mnist':     {'shape': (1, 28, 28), 'classes': 10,  'net': 'convnet',  'ctx': False},
        'affnist':   {'shape': (1, 40, 40), 'classes': 10,  'net': 'resnet50', 'ctx': False},
        'rimagenet': {'shape': (3, 64, 64), 'classes': 200, 'net': 'resnet50', 'ctx': False}
    }
    
    cfg = config[args_cmd.dataset]

    class DataArgs:
        dataset = args_cmd.dataset
        data_dir = "./data"
        sampling_type = 'regular'
        batch_size = 128
        num_workers = 4
        drop_last = False

    print(f"Loading {args_cmd.dataset}...")
    _, _, test_dataset = get_dataset(DataArgs())
    dataloader = get_loader(test_dataset, sampling_type='regular', batch_size=128, shuffle=True, args=DataArgs())

    model = MMAMLConvNet3(
        num_channels=cfg['shape'][0],
        prediction_net=cfg['net'],
        num_classes=cfg['classes'],
        use_context=cfg['ctx'],
        context_num=1,
        support_size=50
    ).to(device)

    checkpoint = torch.load(args_cmd.ckpt, map_location=device)
    state_dict = checkpoint[0] if isinstance(checkpoint, tuple) else checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    plot_tsne(model, dataloader, device, args_cmd.dataset)

if __name__ == "__main__":
    main()
