from pathlib import Path

grains_dir = Path("cropped_grains")

for species_dir in sorted(grains_dir.iterdir()):
    if not species_dir.is_dir():
        continue
    
    images = sorted(species_dir.glob("*.jpg"))
    
    for new_idx, img_path in enumerate(images):
        new_name = f"{species_dir.name}_grain_{new_idx:04d}.jpg"
        new_path = species_dir / new_name
        if img_path.name != new_name:
            img_path.rename(new_path)
    
    print(f"{species_dir.name}: {len(images)} grains renamed")
