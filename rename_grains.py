from pathlib import Path
import re
import uuid

grains_dir = Path("cropped_grains")

def base_species_name(folder_name):
    # Keep filenames species-based even when folders include crop size suffixes like "(384x384)".
    return re.sub(r"\s*\(\d+x\d+\)\s*$", "", folder_name).strip()

for species_dir in sorted(grains_dir.iterdir()):
    if not species_dir.is_dir() or species_dir.name.startswith("."):
        continue
    
    images = sorted(species_dir.glob("*.jpg"))

    species_stem = base_species_name(species_dir.name)
    temp_tag = uuid.uuid4().hex[:8]
    temp_paths = []

    # Two-pass rename avoids collisions when filling index gaps.
    for idx, img_path in enumerate(images):
        temp_path = species_dir / f".__renaming_{temp_tag}_{idx:04d}.jpg"
        img_path.rename(temp_path)
        temp_paths.append(temp_path)

    for new_idx, temp_path in enumerate(temp_paths):
        new_name = f"{species_stem}_grain_{new_idx:04d}.jpg"
        new_path = species_dir / new_name
        temp_path.rename(new_path)
    
    print(f"{species_dir.name}: {len(images)} grains renamed")
