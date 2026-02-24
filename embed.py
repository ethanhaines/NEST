import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import argparse
import json
import os
import re
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


def l2_normalize(x, axis=-1, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, eps, None)


def extract_single(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_avg = patch_tokens.mean(dim=1)
        embedding = (cls_token + patch_avg) / 2.0
    return embedding.cpu().numpy().squeeze()


def extract_batch(images, processor, model):
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        cls_token = outputs.last_hidden_state[:, 0, :]
        patch_avg = patch_tokens.mean(dim=1)
        embeddings = (cls_token + patch_avg) / 2.0
    return embeddings.cpu().numpy()


def extract_embedding(image_path, processor, model):
    image = Image.open(image_path).convert("L").convert("RGB")
    augmented = augment_image(image)
    embeddings = extract_batch(augmented, processor, model)
    embeddings = l2_normalize(embeddings, axis=1)
    embedding = np.mean(embeddings, axis=0)
    embedding = l2_normalize(embedding, axis=0)
    return embedding.astype(np.float32)


def parse_species_folder(folder_name):
    # Split labels like "betula_populifolia (256x256)" into species + crop size metadata.
    match = re.match(r"^(.*?)\s*\((\d+x\d+)\)\s*$", folder_name)
    if not match:
        return folder_name, None
    species = match.group(1).strip()
    crop_size = match.group(2)
    return species, crop_size


def parse_metadata_from_relative_path(relative_path):
    parts = relative_path.parts
    if len(parts) >= 3:
        # New layout: cropped_grains/<species>/<crop_size>/<image>
        species = parts[0]
        crop_candidate = parts[1]
        if re.fullmatch(r"\d+x\d+", crop_candidate):
            return species, crop_candidate, "/".join(parts[:2])

    # Legacy layout: cropped_grains/<species (256x256)>/<image>
    source_folder = parts[0] if len(parts) > 1 else "unknown"
    species, crop_size = parse_species_folder(source_folder)
    return species, crop_size, source_folder


def process_directory(input_dir, output_path, processor, model):
    input_dir = Path(input_dir)
    embeddings = []
    metadata = []

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    image_files = sorted(
        (f for f in input_dir.rglob("*") if f.suffix.lower() in image_extensions),
        key=lambda p: str(p).lower(),
    )

    print(f"Found {len(image_files)} images in {input_dir}")

    for i, img_path in enumerate(image_files):
        try:
            embedding = extract_embedding(img_path, processor, model)
            embeddings.append(embedding)

            relative_path = img_path.relative_to(input_dir)
            species, crop_size, source_folder = parse_metadata_from_relative_path(relative_path)
            metadata.append(
                {
                    "path": str(img_path),
                    "species": species,
                    "filename": img_path.name,
                    "source_folder": source_folder,
                    "crop_size": crop_size,
                }
            )

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
