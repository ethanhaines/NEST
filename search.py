import numpy as np
import faiss
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from embed import load_model, extract_embedding

load_dotenv()

IMAGES_DIR = Path("cropped_grains")

def build_index(embeddings_path):
    embeddings = np.load(embeddings_path).astype("float32")
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    metadata_path = Path(embeddings_path).with_suffix(".json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    return index, metadata

def get_species_list():
    species = [d.name for d in IMAGES_DIR.iterdir() if d.is_dir()]
    return sorted(species)

def get_images_for_species(species_name):
    species_dir = IMAGES_DIR / species_name
    return sorted([f for f in species_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

def query_image(image_path, index, metadata, processor, model, k=10):
    embedding = extract_embedding(image_path, processor, model)
    embedding = embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(embedding)
    
    similarities, indices = index.search(embedding, k + 1)
    
    query_path_str = str(image_path)
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if metadata[idx]["path"] == query_path_str:
            continue
        results.append({
            "similarity": float(sim),
            "species": metadata[idx]["species"],
            "filename": metadata[idx]["filename"],
            "path": metadata[idx]["path"]
        })
    
    return results[:k]

def open_image(path):
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])

def print_results(results, query_path):
    print(f"\nTop {len(results)} matches for tile {query_path.name}:")
    print("-" * 60)
    
    species_scores = {}
    for r in results:
        if r["species"] not in species_scores:
            species_scores[r["species"]] = []
        species_scores[r["species"]].append(r["similarity"])
    
    for i, r in enumerate(results, 1):
        tile_id = r['filename'].split('_tile_')[-1].replace('.jpg', '')
        print(f"{i:2}. {r['species']:25} tile {tile_id:>3} (similarity: {r['similarity']:.3f})")
    
    print("\n" + "-" * 60)
    print("Species summary:")
    for species, scores in sorted(species_scores.items(), key=lambda x: -max(x[1])):
        print(f"  {species}: {len(scores)} matches, best={max(scores):.3f}")
    
    while True:
        view_input = input("\nOpen an image? (y/n): ").strip().lower()
        if view_input != 'y':
            break
        
        img_input = input(f"Enter result number (1-{len(results)}): ").strip()
        try:
            img_idx = int(img_input)
            if img_idx < 1 or img_idx > len(results):
                print(f"Invalid number (must be 1-{len(results)})")
                continue
        except ValueError:
            print("Invalid input")
            continue
        
        open_image(results[img_idx - 1]["path"])
        print(f"Opened: {results[img_idx - 1]['filename']}")

def interactive_mode(index, metadata, processor, model, k):
    while True:
        species_list = get_species_list()
        print(f"\n{len(species_list)} species available:")
        for i, sp in enumerate(species_list):
            count = len(get_images_for_species(sp))
            print(f"  {i}: {sp} ({count} grains)")
        
        species_input = input("\nSelect species number (or 'q' to quit): ").strip()
        if species_input.lower() == 'q':
            break
        
        try:
            species_idx = int(species_input)
            if species_idx < 0 or species_idx >= len(species_list):
                print("Invalid species number")
                continue
        except ValueError:
            print("Invalid input")
            continue
        
        selected_species = species_list[species_idx]
        images = get_images_for_species(selected_species)
        print(f"\n{selected_species}: {len(images)} grains (0-{len(images)-1})")
        
        img_input = input("Select grain number: ").strip()
        try:
            img_idx = int(img_input)
            if img_idx < 0 or img_idx >= len(images):
                print(f"Invalid number (must be 0-{len(images)-1})")
                continue
        except ValueError:
            print("Invalid input")
            continue
        
        query_path = images[img_idx]
        results = query_image(query_path, index, metadata, processor, model, k)
        print_results(results, query_path)

def main():
    parser = argparse.ArgumentParser(description="Search for similar pollen grains")
    parser.add_argument("--index", "-i", default="embeddings/grain_embeddings.npy", help="Path to embeddings")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--model", "-m", default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="DINOv3 model name")
    
    args = parser.parse_args()
    
    processor, model = load_model(args.model)
    index, metadata = build_index(args.index)
    print(f"Built index with {len(metadata)} embeddings")
    
    interactive_mode(index, metadata, processor, model, args.k)

if __name__ == "__main__":
    main()
