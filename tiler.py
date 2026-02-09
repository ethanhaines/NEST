import cv2
import os
import re
import argparse

def extract_species(filename):
    patterns = [
        r"REAL[_ ](?:Betulaceae[_ ])?(\w+)[_ ](\w+)",
        r"Juglandaceae[_ ](\w+)[_ ](\w+)",
        r"(\w+)[_ ](\w+)[_ ]-",
        r"1000x[_ ]#\d+[_ ](\w+)[_ ](\w+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return f"{match.group(1).lower()}_{match.group(2).lower()}"
    
    return None

def tile_slide(img_path, output_dir, tile_size=1024, overlap=0.2):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        return 0
    
    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap))
    count = 0
    basename = os.path.splitext(os.path.basename(img_path))[0]
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = img[y:y_end, x:x_end]
            
            if tile.shape[0] < tile_size // 2 or tile.shape[1] < tile_size // 2:
                continue
            
            tile_name = f"{basename}_tile_{count:03d}.jpg"
            cv2.imwrite(os.path.join(output_dir, tile_name), tile)
            count += 1
    
    return count

def get_processed_basenames(output_dir):
    processed = set()
    output_path = os.path.join(output_dir, ".processed")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            processed = set(line.strip() for line in f)
    return processed

def mark_as_processed(output_dir, basename):
    output_path = os.path.join(output_dir, ".processed")
    with open(output_path, "a") as f:
        f.write(basename + "\n")

def main():
    parser = argparse.ArgumentParser(description="Tile pollen slide images")
    parser.add_argument("--input", "-i", default="../../imges", help="Input directory with slide images")
    parser.add_argument("--output", "-o", default="tiles", help="Output directory for tiles")
    parser.add_argument("--tile-size", "-t", type=int, default=1024, help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap fraction between tiles")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-tiling of already processed images")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    processed = get_processed_basenames(args.output) if not args.force else set()
    
    for filename in os.listdir(args.input):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
            continue
        
        basename = os.path.splitext(filename)[0]
        if basename in processed:
            print(f"Skipping {filename} (already processed)")
            continue
        
        species = extract_species(filename)
        if not species:
            print(f"Warning: Could not extract species from {filename}, skipping")
            continue
        
        species_dir = os.path.join(args.output, species)
        os.makedirs(species_dir, exist_ok=True)
        
        img_path = os.path.join(args.input, filename)
        count = tile_slide(img_path, species_dir, args.tile_size, args.overlap)
        print(f"Tiled {filename} -> {count} tiles in {species}/")
        
        mark_as_processed(args.output, basename)

if __name__ == "__main__":
    main()