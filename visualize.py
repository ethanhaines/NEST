import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
import argparse

def load_embeddings(embeddings_path):
    embeddings = np.load(embeddings_path)
    metadata_path = Path(embeddings_path).with_suffix(".json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    return embeddings, metadata

def create_tsne_plot(embeddings_path, output_path=None, perplexity=30, n_iter=1000):
    embeddings, metadata = load_embeddings(embeddings_path)
    
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Running t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    species_list = sorted(set(m["species"] for m in metadata))
    colors = plt.cm.tab10(np.linspace(0, 1, len(species_list)))
    species_to_color = {sp: colors[i] for i, sp in enumerate(species_list)}
    
    plt.figure(figsize=(12, 10))
    
    for species in species_list:
        indices = [i for i, m in enumerate(metadata) if m["species"] == species]
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]
        plt.scatter(x, y, c=[species_to_color[species]], label=species, alpha=0.6, s=50)
    
    plt.legend(loc='best', fontsize=10)
    plt.title(f"t-SNE of DINOv3 Pollen Grain Embeddings ({len(embeddings)} grains)", fontsize=14)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create t-SNE visualization of pollen grain embeddings")
    parser.add_argument("--embeddings", "-e", default="embeddings/grain_embeddings.npy", help="Path to embeddings")
    parser.add_argument("--output", "-o", default="tsne_plot.png", help="Output image path")
    parser.add_argument("--perplexity", "-p", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--iterations", "-n", type=int, default=1000, help="t-SNE iterations")
    
    args = parser.parse_args()
    create_tsne_plot(args.embeddings, args.output, args.perplexity, args.iterations)

if __name__ == "__main__":
    main()
