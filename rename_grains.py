from pathlib import Path
import re
import uuid

grains_dir = Path("cropped_grains")

def base_species_name(folder_name):
    # Support both legacy "species (384x384)" and new nested "species/<384x384>" layouts.
    return re.sub(r"\s*\(\d+x\d+\)\s*$", "", folder_name).strip()

def rename_images_in_dir(images_dir, species_stem):
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        return 0

    temp_tag = uuid.uuid4().hex[:8]
    temp_paths = []

    # Two-pass rename avoids collisions when filling index gaps.
    for idx, img_path in enumerate(images):
        temp_path = images_dir / f".__renaming_{temp_tag}_{idx:04d}.jpg"
        img_path.rename(temp_path)
        temp_paths.append(temp_path)

    for new_idx, temp_path in enumerate(temp_paths):
        new_name = f"{species_stem}_grain_{new_idx:04d}.jpg"
        new_path = images_dir / new_name
        temp_path.rename(new_path)

    return len(images)

for species_dir in sorted(grains_dir.iterdir()):
    if not species_dir.is_dir() or species_dir.name.startswith("."):
        continue

    species_stem = base_species_name(species_dir.name)

    # New layout: cropped_grains/species/256x256/*.jpg
    crop_dirs = [d for d in sorted(species_dir.iterdir()) if d.is_dir()]
    renamed_any = False
    for crop_dir in crop_dirs:
        count = rename_images_in_dir(crop_dir, species_stem)
        if count:
            print(f"{species_dir.name}/{crop_dir.name}: {count} grains renamed")
            renamed_any = True

    # Legacy fallback: cropped_grains/species (256x256)/*.jpg
    if not renamed_any:
        count = rename_images_in_dir(species_dir, species_stem)
        print(f"{species_dir.name}: {count} grains renamed")
