import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import argparse
import json
import os
from dotenv import load_dotenv

load_dotenv()

def load_model(model_name="facebook/dinov3-vitb16-pretrain-lvd1689m"):
    token = os.getenv("hugging_face_token")
    processor = AutoImageProcessor.from_pretrained(model_name, token=token)
    model = AutoModel.from_pretrained(model_name, token=token)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} on {device}")
    return processor, model

def augment_image(image):
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    r90 = image.rotate(90, expand=True)
    r180 = image.rotate(180)
    r270 = image.rotate(270, expand=True)
    return [image, flipped, r90, r180, r270]

def extract_single(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_avg = patch_tokens.mean(dim=1)
        embedding = (cls_token + patch_avg) / 2.0
    return embedding.cpu().numpy().squeeze()

def extract_embedding(image_path, processor, model):
    image = Image.open(image_path).convert("L").convert("RGB")
    augmented = augment_image(image)
    embeddings = [extract_single(img, processor, model) for img in augmented]
    embedding = np.mean(embeddings, axis=0)
    return embedding

def process_directory(input_dir, output_path, processor, model):
    input_dir = Path(input_dir)
    embeddings = []
    metadata = []
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for i, img_path in enumerate(image_files):
        try:
            embedding = extract_embedding(img_path, processor, model)
            embeddings.append(embedding)
            
            relative_path = img_path.relative_to(input_dir)
            species = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
            metadata.append({
                "path": str(img_path),
                "species": species,
                "filename": img_path.name
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_files)}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, embeddings)
    
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(embeddings)} embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings, metadata

def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 embeddings from pollen images")
    parser.add_argument("input_dir", help="Directory containing images (can have subdirectories by species)")
    parser.add_argument("--output", "-o", default="embeddings/extant_embeddings.npy", help="Output path for embeddings")
    parser.add_argument("--model", "-m", default="facebook/dinov3-vitb16-pretrain-lvd1689m", help="DINOv3 model name")
    
    args = parser.parse_args()
    
    processor, model = load_model(args.model)
    process_directory(args.input_dir, args.output, processor, model)

if __name__ == "__main__":
    main()
