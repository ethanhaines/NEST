import cv2
from pathlib import Path
import argparse
import re
import json

SLIDES_DIR = Path("images")
OUTPUT_DIR = Path("cropped_grains")
CROP_SIZE = 256


def normalize_species_name(raw_name):
    normalized = re.sub(r"[^a-z0-9]+", "_", raw_name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def infer_species_name(image_path, images_root):
    relative = image_path.relative_to(images_root)

    if len(relative.parts) > 1 and relative.parts[0].lower() != "done":
        return normalize_species_name(relative.parts[0])

    filename = image_path.stem
    patterns = [
        r"REAL[_ ](?:Betulaceae[_ ])?(\w+)[_ ](\w+)",
        r"Juglandaceae[_ ](\w+)[_ ](\w+)",
        r"(\w+)[_ ](\w+)[_ ]-",
        r"\d+x[_ ]#\d+[_ ](\w+)[_ ](\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return normalize_species_name(f"{match.group(1)}_{match.group(2)}")

    return ""

class SlideViewer:
    def __init__(self, image_path, crop_size, output_dir, species_name):
        self.image_path = image_path
        self.crop_size = crop_size
        self.output_dir = output_dir
        self.species_name = species_name
        
        print(f"Loading {image_path}...")
        self.full_image = cv2.imread(str(image_path))
        if self.full_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.full_h, self.full_w = self.full_image.shape[:2]
        print(f"Image size: {self.full_w} x {self.full_h} pixels")
        
        self.window_size = 1200
        self.zoom_level = 1.0
        self.min_zoom = max(self.window_size / self.full_w, self.window_size / self.full_h)
        self.max_zoom = 10.0
        
        self.center_x = self.full_w // 2
        self.center_y = self.full_h // 2

        self.grain_count = 0
        self.next_index_by_size = {}

        markers_dir = output_dir / ".markers"
        markers_dir.mkdir(parents=True, exist_ok=True)
        marker_key = re.sub(r"[^a-zA-Z0-9._-]+", "__", str(image_path))
        self.marker_path = markers_dir / f"{marker_key}.json"

        self.dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_x = self.window_size // 2
        self.mouse_y = self.window_size // 2

        self.cropped_grains = []
        self.crop_history = []

        self.load_markers()

        print(f"\nControls:")
        print(f"  Left click = crop grain at cursor")
        print(f"  Two-finger click+drag (right click) or middle-click drag = pan image")
        print(f"  Arrow keys = pan image")
        print(f"  '+' or '=' = zoom in")
        print(f"  '-' = zoom out")
        print(f"  '[' / ']' = decrease/increase crop size")
        print(f"  'c' = set exact crop size")
        print(f"  'u' = undo last crop")
        print(f"  'r' = reset view")
        print(f"  'q' = quit")
        print(f"\nStarting crop size: {self.crop_size}")
        print(f"Loaded {self.grain_count} persisted markers")
        print(f"Saving crops into size-specific folders like: {self.species_name}/256x256")

    def get_species_output_dir(self, crop_size):
        species_output = self.output_dir / self.species_name / f"{crop_size}x{crop_size}"
        species_output.mkdir(parents=True, exist_ok=True)
        return species_output

    def get_next_index_for_size(self, crop_size):
        if crop_size not in self.next_index_by_size:
            species_output = self.get_species_output_dir(crop_size)
            self.next_index_by_size[crop_size] = len(list(species_output.glob("*.jpg")))
        return self.next_index_by_size[crop_size]

    def set_crop_size(self, new_size):
        if new_size < 32:
            print("Crop size must be at least 32")
            return
        if new_size % 2 != 0:
            new_size += 1
        self.crop_size = int(new_size)
        print(f"Crop size set to {self.crop_size}")

    def load_markers(self):
        if not self.marker_path.exists():
            return
        try:
            with open(self.marker_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            crops = payload.get("crops")
            if isinstance(crops, list):
                loaded = []
                for item in crops:
                    if not isinstance(item, dict):
                        continue
                    if "x" not in item or "y" not in item:
                        continue
                    crop_size = int(item.get("crop_size", payload.get("crop_size", self.crop_size)))
                    loaded.append({
                        "x": int(item["x"]),
                        "y": int(item["y"]),
                        "crop_size": crop_size,
                    })
                self.cropped_grains = loaded
            else:
                points = payload.get("points", [])
                legacy_size = int(payload.get("crop_size", self.crop_size))
                self.cropped_grains = [
                    {"x": int(point[0]), "y": int(point[1]), "crop_size": legacy_size}
                    for point in points
                    if isinstance(point, list) and len(point) == 2
                ]
            self.grain_count = len(self.cropped_grains)
        except Exception as exc:
            print(f"Warning: Could not load markers from {self.marker_path}: {exc}")

    def save_markers(self):
        payload = {
            "image_path": str(self.image_path),
            "crop_size": self.crop_size,
            "species": self.species_name,
            "points": [[item["x"], item["y"]] for item in self.cropped_grains],                        
            "crops": [
                {"x": item["x"], "y": item["y"], "crop_size": item["crop_size"]}
                for item in self.cropped_grains
            ],
        }
        with open(self.marker_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    
    def get_visible_region(self):
        view_w = int(self.window_size / self.zoom_level)
        view_h = int(self.window_size / self.zoom_level)
        
        x1 = max(0, self.center_x - view_w // 2)
        y1 = max(0, self.center_y - view_h // 2)
        x2 = min(self.full_w, x1 + view_w)
        y2 = min(self.full_h, y1 + view_h)
        
        if x2 - x1 < view_w:
            x1 = max(0, x2 - view_w)
        if y2 - y1 < view_h:
            y1 = max(0, y2 - view_h)
        
        return x1, y1, x2, y2
    
    def render_view(self):
        x1, y1, x2, y2 = self.get_visible_region()
        
        view = self.full_image[y1:y2, x1:x2].copy()

        display = cv2.resize(view, (self.window_size, self.window_size), interpolation=cv2.INTER_LINEAR)

        view_w = max(1, x2 - x1)
        view_h = max(1, y2 - y1)
        scale_x = self.window_size / view_w
        scale_y = self.window_size / view_h

                                                                                               
        for item in self.cropped_grains:
            grain_x = item["x"]
            grain_y = item["y"]
            crop_size = item.get("crop_size", self.crop_size)
            if x1 <= grain_x < x2 and y1 <= grain_y < y2:
                rel_x = grain_x - x1
                rel_y = grain_y - y1

                disp_x = int(round(rel_x * scale_x))
                disp_y = int(round(rel_y * scale_y))

                circle_r = max(3, int(round(15 * scale_x)))
                rect_half_w = max(1, int(round((crop_size / 2) * scale_x)))
                rect_half_h = max(1, int(round((crop_size / 2) * scale_y)))

                cv2.circle(display, (disp_x, disp_y), circle_r, (0, 255, 0), 2)
                cv2.rectangle(
                    display,
                    (disp_x - rect_half_w, disp_y - rect_half_h),
                    (disp_x + rect_half_w, disp_y + rect_half_h),
                    (0, 255, 0),
                    2,
                )

                                                                             
        ghost_x = int(max(0, min(self.window_size - 1, self.mouse_x)))
        ghost_y = int(max(0, min(self.window_size - 1, self.mouse_y)))
        ghost_half_w = max(1, int(round((self.crop_size / 2) * scale_x)))
        ghost_half_h = max(1, int(round((self.crop_size / 2) * scale_y)))
        cv2.rectangle(
            display,
            (ghost_x - ghost_half_w, ghost_y - ghost_half_h),
            (ghost_x + ghost_half_w, ghost_y + ghost_half_h),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(display, f"Zoom: {self.zoom_level:.2f}x | Grains: {self.grain_count} | Crop: {self.crop_size}px", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Position: ({self.center_x}, {self.center_y})",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display
    
    def screen_to_image_coords(self, screen_x, screen_y):
        x1, y1, x2, y2 = self.get_visible_region()
        
        view_w = x2 - x1
        view_h = y2 - y1
        
        rel_x = screen_x / self.window_size
        rel_y = screen_y / self.window_size
        
        img_x = int(x1 + rel_x * view_w)
        img_y = int(y1 + rel_y * view_h)
        
        return img_x, img_y

    def adjust_zoom(self, zoom_in):
        if zoom_in:
            self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
        else:
            self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)

    def pan_by(self, dx, dy):
        self.center_x += dx
        self.center_y += dy
        self.center_x = max(0, min(self.full_w, self.center_x))
        self.center_y = max(0, min(self.full_h, self.center_y))
    
    def crop_grain(self, center_x, center_y):
        crop_size = self.crop_size
        half = crop_size // 2
        
        x1 = center_x - half
        y1 = center_y - half
        x2 = center_x + half
        y2 = center_y + half
        
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > self.full_w:
            x1 -= (x2 - self.full_w)
            x2 = self.full_w
        if y2 > self.full_h:
            y1 -= (y2 - self.full_h)
            y2 = self.full_h
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.full_w, x2)
        y2 = min(self.full_h, y2)
        
        crop = self.full_image[y1:y2, x1:x2]
        
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size))

        species_output = self.get_species_output_dir(crop_size)
        grain_idx = self.get_next_index_for_size(crop_size)
        output_name = f"{self.species_name}_grain_{grain_idx:04d}.jpg"
        output_path = species_output / output_name
        cv2.imwrite(str(output_path), crop)

        crop_record = {"x": center_x, "y": center_y, "crop_size": crop_size}
        self.cropped_grains.append(crop_record)
        self.crop_history.append({"path": output_path, "crop": crop_record})
        self.next_index_by_size[crop_size] = grain_idx + 1
        self.save_markers()
        self.grain_count += 1
        print(f"Saved: {output_name} [{crop_size}x{crop_size}] (total markers: {self.grain_count})")

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self.screen_to_image_coords(x, y)
            self.crop_grain(img_x, img_y)

        elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y

        elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.dragging = False

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y

            move_scale = 1.0 / self.zoom_level
            self.pan_by(-int(dx * move_scale), -int(dy * move_scale))
            
            self.last_mouse_x = x
            self.last_mouse_y = y
    
    def run(self):
        key_left = 2424832
        key_up = 2490368
        key_right = 2555904
        key_down = 2621440

        cv2.namedWindow("Slide Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slide Viewer", self.window_size, self.window_size)
        cv2.setMouseCallback("Slide Viewer", self.on_mouse)
        
        while True:
            display = self.render_view()
            cv2.imshow("Slide Viewer", display)
            
            key = cv2.waitKeyEx(50)
            pan_step = max(10, int(600 / self.zoom_level))
            
            if key == ord('q'):
                break
            elif key == ord('u'):
                if self.crop_history:
                    last = self.crop_history.pop()
                    last_crop_path = last["path"]
                    last_crop = last["crop"]
                    if last_crop_path.exists():
                        last_crop_path.unlink()
                    self.grain_count -= 1
                    if self.cropped_grains:
                        self.cropped_grains.pop()
                    crop_size = int(last_crop.get("crop_size", self.crop_size))
                    if crop_size in self.next_index_by_size:
                        self.next_index_by_size[crop_size] = max(0, self.next_index_by_size[crop_size] - 1)
                    self.save_markers()
                    print(f"Undone last crop (total: {self.grain_count})")
                else:
                    print("Nothing to undo")
            elif key == ord('r'):
                self.zoom_level = 1.0
                self.center_x = self.full_w // 2
                self.center_y = self.full_h // 2
                print("View reset")
            elif key in (ord('+'), ord('=')):
                self.adjust_zoom(zoom_in=True)
            elif key in (ord('-'), ord('_')):
                self.adjust_zoom(zoom_in=False)
            elif key in (ord(']'), ord('}')):
                self.set_crop_size(self.crop_size + 32)
            elif key in (ord('['), ord('{')):
                self.set_crop_size(self.crop_size - 32)
            elif key == ord('c'):
                crop_size_input = input(f"Set crop size (current {self.crop_size}): ").strip()
                if crop_size_input:
                    try:
                        self.set_crop_size(int(crop_size_input))
                    except ValueError:
                        print("Invalid crop size")
            elif key == key_left:
                self.pan_by(-pan_step, 0)
            elif key == key_right:
                self.pan_by(pan_step, 0)
            elif key == key_up:
                self.pan_by(0, -pan_step)
            elif key == key_down:
                self.pan_by(0, pan_step)
        
        cv2.destroyAllWindows()
        print(f"\nTotal grains cropped: {self.grain_count}")

def get_slide_list():
    slides = []
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        slides.extend(
            path
            for path in SLIDES_DIR.rglob(f"*{ext}")
            if "done" not in [part.lower() for part in path.relative_to(SLIDES_DIR).parts]
        )
    return sorted(slides)

def main():
    parser = argparse.ArgumentParser(description="Crop pollen grains from full-resolution slide images")
    parser.add_argument("--slides-dir", "--images-dir", "-i", default="images", help="Directory containing full-resolution input images")
    parser.add_argument("--output", "-o", default="cropped_grains", help="Output directory")
    parser.add_argument("--crop-size", "-c", type=int, default=256, help="Crop size in pixels")
    
    args = parser.parse_args()
    
    global SLIDES_DIR, OUTPUT_DIR, CROP_SIZE
    SLIDES_DIR = Path(args.slides_dir)
    OUTPUT_DIR = Path(args.output)
    CROP_SIZE = args.crop_size
    
    slides = get_slide_list()
    
    if not slides:
        print(f"No slide images found in {SLIDES_DIR}")
        return
    
    print(f"\n{len(slides)} slide images available:")
    for i, slide in enumerate(slides):
        rel_path = slide.relative_to(SLIDES_DIR)
        inferred_species = infer_species_name(slide, SLIDES_DIR) or "unknown"
        print(f"  {i}: {rel_path}  [species: {inferred_species}]")
    
    slide_input = input("\nSelect slide number: ").strip()
    try:
        slide_idx = int(slide_input)
        selected_slide = slides[slide_idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    inferred_species = infer_species_name(selected_slide, SLIDES_DIR)
    species_prompt = f"Species name [{inferred_species}]: " if inferred_species else "Species name (e.g., betula_populifolia): "
    species_input = input(species_prompt).strip()
    species_name = normalize_species_name(species_input) if species_input else inferred_species
    if not species_name:
        print("Species name required")
        return
    
    crop_size_input = input(f"Crop size in pixels (default {CROP_SIZE}): ").strip()
    if crop_size_input:
        try:
            crop_size = int(crop_size_input)
        except ValueError:
            print(f"Invalid crop size, using default {CROP_SIZE}")
            crop_size = CROP_SIZE
    else:
        crop_size = CROP_SIZE
    
    viewer = SlideViewer(selected_slide, crop_size, OUTPUT_DIR, species_name)
    viewer.run()

if __name__ == "__main__":
    main()
