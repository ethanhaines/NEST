import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.widgets import Slider
from pathlib import Path
import argparse
import os
import subprocess
import sys

try:
    from mpl_toolkits.mplot3d import Axes3D              
except ImportError:
    Axes3D = None

try:
    import mplcursors
except ImportError:
    mplcursors = None


def load_embeddings(embeddings_path):
    embeddings = np.load(embeddings_path)
    metadata_path = Path(embeddings_path).with_suffix(".json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    return embeddings, metadata


def _build_species_color_map(metadata):
    species_list = sorted(set(m["species"] for m in metadata))
    cmap = plt.get_cmap("tab20", max(len(species_list), 1))
    species_to_color = {sp: cmap(i) for i, sp in enumerate(species_list)}
    return species_list, species_to_color


def _enable_hover_tooltips(scatter_groups, metadata):
    if mplcursors is None:
        print("Hover tooltips unavailable: install mplcursors (`pip install mplcursors`) to enable point hover labels.")
        return

    hover_mode = getattr(getattr(mplcursors, "HoverMode", None), "Transient", True)
    cursor = mplcursors.cursor([group["artist"] for group in scatter_groups], hover=hover_mode)
    artist_to_indices = {group["artist"]: group["indices"] for group in scatter_groups}

    @cursor.connect("add")
    def _on_add(sel):
        indices = artist_to_indices.get(sel.artist, [])
        if not indices:
            return
        point_idx = indices[sel.index]
        item = metadata[point_idx]
        crop_size = item.get("crop_size") or "unknown"
        sel.annotation.set_text(
            f"{item.get('filename', 'unknown')}\n"
            f"{item.get('species', 'unknown')} | crop={crop_size}"
        )
        sel.annotation.get_bbox_patch().set(alpha=0.9)


def _open_image(path):
    path = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        print(f"Could not open file: {path} ({exc})")


def _resolve_metadata_image_path(item):
    raw_path = item.get("path")
    if raw_path:
        p = Path(raw_path)
        if p.exists():
            return p

    filename = item.get("filename")
    species = item.get("species")
    crop_size = item.get("crop_size")
    source_folder = item.get("source_folder")
    if not filename:
        return None

    candidates = []

                                                                          
    if species and crop_size:
        candidates.append(Path("cropped_grains") / str(species) / str(crop_size) / str(filename))

                                                                                  
    if source_folder:
        candidates.append(Path("cropped_grains") / Path(str(source_folder)) / str(filename))

                                                                           
    if species and crop_size:
        candidates.append(Path("cropped_grains") / f"{species} ({crop_size})" / str(filename))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _enable_click_open(fig, scatter_groups, metadata):
    artist_to_indices = {group["artist"]: group["indices"] for group in scatter_groups}

    def _on_pick(event):
        artist = event.artist
        indices = artist_to_indices.get(artist)
        if not indices:
            return
        picked = getattr(event, "ind", None)
        if picked is None or len(picked) == 0:
            return
        point_idx = indices[int(picked[0])]
        item = metadata[point_idx]
        path = _resolve_metadata_image_path(item)
        if not path:
            print(f"Could not resolve image path for: {item.get('filename', 'unknown')}")
            return
        print(f"Opening: {item.get('filename', path)}")
        _open_image(path)

    fig.canvas.mpl_connect("pick_event", _on_pick)


def create_tsne_plot_2d(embeddings_path, output_path=None, perplexity=30, n_iter=1000):
    embeddings, metadata = load_embeddings(embeddings_path)

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Running 2D t-SNE (perplexity={perplexity}, n_iter={n_iter})...")

    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    species_list, species_to_color = _build_species_color_map(metadata)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter_groups = []

    for species in species_list:
        indices = [i for i, m in enumerate(metadata) if m["species"] == species]
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]
        artist = ax.scatter(x, y, c=[species_to_color[species]], label=species, alpha=0.6, s=50, picker=True)
        scatter_groups.append({"artist": artist, "indices": indices})

    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"t-SNE of DINOv3 Pollen Grain Embeddings ({len(embeddings)} grains)", fontsize=14)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        _enable_click_open(fig, scatter_groups, metadata)
        _enable_hover_tooltips(scatter_groups, metadata)
        print("Interactive controls: hover for labels, left-click a point to open its image file.")
        plt.show()


def create_tsne_plot_3d(embeddings_path, output_path=None, perplexity=30, n_iter=1000, elev=20):
    if Axes3D is None:
        raise RuntimeError("3D plotting is not available in this matplotlib installation")

    embeddings, metadata = load_embeddings(embeddings_path)

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Running 3D t-SNE (perplexity={perplexity}, n_iter={n_iter})...")

    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)

    species_list, species_to_color = _build_species_color_map(metadata)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.14 if output_path is None else 0.10)
    scatter_groups = []

    for species in species_list:
        indices = [i for i, m in enumerate(metadata) if m["species"] == species]
        pts = embeddings_3d[indices]
        artist = ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=[species_to_color[species]],
            label=species,
            alpha=0.65,
            s=28,
            depthshade=True,
            picker=True,
        )
        scatter_groups.append({"artist": artist, "indices": indices})

    ax.set_title(f"3D t-SNE of DINOv3 Pollen Grain Embeddings ({len(embeddings)} grains)", fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_zlabel("t-SNE dim 3")
    ax.view_init(elev=elev, azim=-60)
    ax.legend(loc="best", fontsize=9)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        return

    slider_ax = fig.add_axes([0.18, 0.04, 0.64, 0.03])
    az_slider = Slider(slider_ax, "Azimuth", -180, 180, valinit=-60, valstep=1)

    def _update_azimuth(val):
        ax.view_init(elev=elev, azim=az_slider.val)
        fig.canvas.draw_idle()

    az_slider.on_changed(_update_azimuth)

    def _on_key(event):
        if event.key in ("left", "a"):
            az_slider.set_val(az_slider.val - 5)
        elif event.key in ("right", "d"):
            az_slider.set_val(az_slider.val + 5)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    _enable_click_open(fig, scatter_groups, metadata)
    _enable_hover_tooltips(scatter_groups, metadata)
    print("3D viewer controls: use the Azimuth slider or Left/Right (A/D) keys for horizontal rotation; left-click points to open files.")
    plt.show()


def create_tsne_plot(embeddings_path, output_path=None, perplexity=30, n_iter=1000, is_3d=False, elev=20):
    if is_3d:
        return create_tsne_plot_3d(embeddings_path, output_path, perplexity, n_iter, elev=elev)
    return create_tsne_plot_2d(embeddings_path, output_path, perplexity, n_iter)


def main():
    parser = argparse.ArgumentParser(description="Create t-SNE visualization of pollen grain embeddings")
    parser.add_argument("--embeddings", "-e", default="embeddings/extant_embeddings.npy", help="Path to embeddings")
    parser.add_argument("--output", "-o", default="tsne_plot.png", help="Output image path")
    parser.add_argument("--perplexity", "-p", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--iterations", "-n", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--3d", dest="is_3d", action="store_true", help="Generate a 3D t-SNE plot (interactive azimuth slider when not saving)")
    parser.add_argument("--elevation", type=float, default=20, help="Fixed elevation angle for 3D view (horizontal rotation stays locked to this angle)")
    parser.add_argument("--show", action="store_true", help="Show interactively instead of saving to --output")
    parser.add_argument("--interactive", "-I", action="store_true", help="Alias for --show (backward compatibility)")

    args = parser.parse_args()
    output_path = None if (args.show or args.interactive) else args.output
    create_tsne_plot(
        args.embeddings,
        output_path,
        args.perplexity,
        args.iterations,
        is_3d=args.is_3d,
        elev=args.elevation,
    )


if __name__ == "__main__":
    main()
