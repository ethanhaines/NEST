import cv2
import numpy as np
from pathlib import Path
import argparse

TILES_DIR = Path("tiles")
OUTPUT_DIR = Path("cropped_grains")
CROP_SIZE = 256

clicks = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

def crop_grain(image, cx, cy, crop_size):
    h, w = image.shape[:2]
    half = crop_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = image[y1:y2, x1:x2]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))

    return crop

def get_species_list():
    species = [d.name for d in TILES_DIR.iterdir() if d.is_dir()]
    return sorted(species)

def get_tiles_for_species(species_name):
    species_dir = TILES_DIR / species_name
    return sorted([f for f in species_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])

def interactive_crop(tiles_dir, output_dir, crop_size):
    global clicks
    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)

    species_list = get_species_list()
    print(f"\n{len(species_list)} species available:")
    for i, sp in enumerate(species_list):
        tile_count = len(get_tiles_for_species(sp))
        print(f"  {i}: {sp} ({tile_count} tiles)")

    species_input = input("\nSelect species number: ").strip()
    try:
        species_idx = int(species_input)
        selected_species = species_list[species_idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return

    tiles = get_tiles_for_species(selected_species)
    species_output = output_dir / (selected_species + " (" + str(crop_size) + "x" + str(crop_size) + ")")
    species_output.mkdir(parents=True, exist_ok=True)

    existing = sorted(species_output.glob("*.jpg"))
    grain_count = len(existing)

    start = int(input(f"\nStart at tile number (0-{len(tiles)-1}): ").strip())

    print(f"\nControls:")
    print(f"  Left click = mark grain center")
    print(f"  'u' = undo last crop")
    print(f"  'n' = next tile")
    print(f"  's' = skip 10 tiles")
    print(f"  'q' = quit")

    cv2.namedWindow("Tile", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tile", 800, 800)
    cv2.setMouseCallback("Tile", on_mouse)

    saved_this_tile = []
    i = start
    while i < len(tiles):
        tile_path = tiles[i]
        image = cv2.imread(str(tile_path))
        if image is None:
            i += 1
            continue

        clicks = []
        saved_this_tile = []
        display = image.copy()
        cv2.putText(display, f"Tile {i}/{len(tiles)-1} | Grains: {grain_count}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Tile", display)

        while True:
            key = cv2.waitKey(50) & 0xFF

            if len(clicks) > 0:
                cx, cy = clicks[-1]
                crop = crop_grain(image, cx, cy, crop_size)
                output_name = f"{selected_species}_grain_{grain_count:04d}.jpg"
                output_path = species_output / output_name
                cv2.imwrite(str(output_path), crop)
                saved_this_tile.append(output_path)
                grain_count += 1
                cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
                cv2.rectangle(display,
                              (cx - crop_size // 2, cy - crop_size // 2),
                              (cx + crop_size // 2, cy + crop_size // 2),
                              (0, 255, 0), 1)
                cv2.putText(display, f"Tile {i}/{len(tiles)-1} | Grains: {grain_count}",
                             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Tile", display)
                clicks = []

            if key == ord('u'):
                if saved_this_tile:
                    last = saved_this_tile.pop()
                    last.unlink()
                    grain_count -= 1
                    display = image.copy()
                    for sp in saved_this_tile:
                        name = sp.stem
                        pass
                    cv2.putText(display, f"Tile {i}/{len(tiles)-1} | Grains: {grain_count} (undone)",
                                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Tile", display)
            elif key == ord('n'):
                break
            elif key == ord('s'):
                i += 9
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print(f"\nTotal grains for {selected_species}: {grain_count}")
                return

        i += 1

    cv2.destroyAllWindows()
    print(f"\nTotal grains for {selected_species}: {grain_count}")

def main():
    parser = argparse.ArgumentParser(description="Manually crop pollen grains from tiles")
    parser.add_argument("--input", "-i", default="tiles", help="Input tiles directory")
    parser.add_argument("--output", "-o", default="cropped_grains", help="Output directory")
    parser.add_argument("--crop-size", "-c", type=int, default=256, help="Crop size in pixels")

    args = parser.parse_args()
    interactive_crop(args.input, args.output, args.crop_size)

if __name__ == "__main__":
    main()
