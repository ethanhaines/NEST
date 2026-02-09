import cv2
import numpy as np
from pathlib import Path
import argparse

SLIDES_DIR = Path("slides")
OUTPUT_DIR = Path("cropped_grains")
CROP_SIZE = 256

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
        self.species_output = output_dir / (species_name + f" ({crop_size}x{crop_size})")
        self.species_output.mkdir(parents=True, exist_ok=True)
        
        existing = sorted(self.species_output.glob("*.jpg"))
        self.grain_count = len(existing)
        
        self.dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.cropped_grains = []
        self.last_crop_path = None
        
        print(f"\nControls:")
        print(f"  Left click = crop grain at cursor")
        print(f"  Right drag = pan image")
        print(f"  Mouse wheel = zoom in/out")
        print(f"  'u' = undo last crop")
        print(f"  'r' = reset view")
        print(f"  'q' = quit")
        print(f"\nStarting with {self.grain_count} existing grains")
    
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
        
        for grain_x, grain_y in self.cropped_grains:
            if x1 <= grain_x < x2 and y1 <= grain_y < y2:
                rel_x = grain_x - x1
                rel_y = grain_y - y1
                
                scale = self.window_size / (x2 - x1)
                screen_x = int(rel_x * scale)
                screen_y = int(rel_y * scale)
                screen_radius = max(5, int(15 * scale))
                screen_half_crop = int((self.crop_size // 2) * scale)
                
                cv2.circle(view, (rel_x, rel_y), int(screen_radius / scale), (0, 255, 0), 2)
                cv2.rectangle(view,
                            (rel_x - self.crop_size // 2, rel_y - self.crop_size // 2),
                            (rel_x + self.crop_size // 2, rel_y + self.crop_size // 2),
                            (0, 255, 0), 1)
        
        display = cv2.resize(view, (self.window_size, self.window_size), interpolation=cv2.INTER_LINEAR)
        
        cv2.putText(display, f"Zoom: {self.zoom_level:.2f}x | Grains: {self.grain_count}", 
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
    
    def crop_grain(self, center_x, center_y):
        half = self.crop_size // 2
        
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
        
        if crop.shape[0] != self.crop_size or crop.shape[1] != self.crop_size:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        
        output_name = f"{self.species_name}_grain_{self.grain_count:04d}.jpg"
        output_path = self.species_output / output_name
        cv2.imwrite(str(output_path), crop)
        
        self.cropped_grains.append((center_x, center_y))
        self.last_crop_path = output_path
        self.grain_count += 1
        print(f"Saved: {output_name} (total: {self.grain_count})")
    
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_x, img_y = self.screen_to_image_coords(x, y)
            self.crop_grain(img_x, img_y)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
        
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            move_scale = 1.0 / self.zoom_level
            self.center_x -= int(dx * move_scale)
            self.center_y -= int(dy * move_scale)
            
            self.center_x = max(0, min(self.full_w, self.center_x))
            self.center_y = max(0, min(self.full_h, self.center_y))
            
            self.last_mouse_x = x
            self.last_mouse_y = y
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            img_x, img_y = self.screen_to_image_coords(x, y)
            
            old_zoom = self.zoom_level
            if flags > 0:
                self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
            else:
                self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)
            
            zoom_ratio = self.zoom_level / old_zoom
            self.center_x = int(img_x + (self.center_x - img_x) * zoom_ratio)
            self.center_y = int(img_y + (self.center_y - img_y) * zoom_ratio)
            
            self.center_x = max(0, min(self.full_w, self.center_x))
            self.center_y = max(0, min(self.full_h, self.center_y))
    
    def run(self):
        cv2.namedWindow("Slide Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Slide Viewer", self.window_size, self.window_size)
        cv2.setMouseCallback("Slide Viewer", self.on_mouse)
        
        while True:
            display = self.render_view()
            cv2.imshow("Slide Viewer", display)
            
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('u'):
                if self.last_crop_path and self.last_crop_path.exists():
                    self.last_crop_path.unlink()
                    self.grain_count -= 1
                    if self.cropped_grains:
                        self.cropped_grains.pop()
                    print(f"Undone last crop (total: {self.grain_count})")
                    self.last_crop_path = None
                else:
                    print("Nothing to undo")
            elif key == ord('r'):
                self.zoom_level = 1.0
                self.center_x = self.full_w // 2
                self.center_y = self.full_h // 2
                print("View reset")
        
        cv2.destroyAllWindows()
        print(f"\nTotal grains cropped: {self.grain_count}")

def get_slide_list():
    slides = []
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        slides.extend(SLIDES_DIR.rglob(f"*{ext}"))
    return sorted(slides)

def main():
    parser = argparse.ArgumentParser(description="Crop pollen grains from full-resolution slide images")
    parser.add_argument("--slides-dir", "-s", default="slides", help="Directory containing slide images")
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
        print(f"  {i}: {rel_path}")
    
    slide_input = input("\nSelect slide number: ").strip()
    try:
        slide_idx = int(slide_input)
        selected_slide = slides[slide_idx]
    except (ValueError, IndexError):
        print("Invalid selection")
        return
    
    species_name = input("Enter species name (e.g., betula_populifolia): ").strip()
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
