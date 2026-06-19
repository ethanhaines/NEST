import argparse
import json
from pathlib import Path

import numpy as np


FOSSIL_DISPLAY_LABEL = "Fossil pollen"


def load_embeddings(embeddings_path):
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata_path = embeddings_path.with_suffix(".json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata length ({len(metadata)}) does not match embeddings length ({len(embeddings)})"
        )

    return embeddings, metadata


def l2_normalize(x, axis=-1, eps=1e-12):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, eps, None)


def is_fossil_item(item):
    searchable = " ".join(
        str(item.get(key, ""))
        for key in ("species", "source_folder", "path", "filename", "label", "query_group")
    ).lower()
    return "fossil" in searchable


def clean_path(value):
    return str(value or "").replace("\\", "/")


def metadata_payload(item, node_id):
    return {
        "node_id": int(node_id),
        "species": item.get("species"),
        "display_species": "fossil_pollen" if is_fossil_item(item) else item.get("species"),
        "display_label": FOSSIL_DISPLAY_LABEL if is_fossil_item(item) else prettify_species(item.get("species")),
        "filename": item.get("filename"),
        "crop_size": item.get("crop_size"),
        "source_folder": clean_path(item.get("source_folder")),
        "path": clean_path(item.get("path")),
    }


def prettify_species(value):
    return str(value or "").replace("_", " ").title()


def build_static_search_results(embeddings, metadata, k=10, include_fossil_targets=False):
    normalized = l2_normalize(embeddings, axis=1)

    fossil_indices = [i for i, item in enumerate(metadata) if is_fossil_item(item)]
    target_indices = [
        i
        for i, item in enumerate(metadata)
        if include_fossil_targets or not is_fossil_item(item)
    ]

    target_matrix = normalized[target_indices]
    target_indices_array = np.asarray(target_indices, dtype=np.int32)

    queries = []
    for query_index in fossil_indices:
        scores = target_matrix @ normalized[query_index]
        order = np.argsort(-scores)

        results = []
        for target_order_index in order:
            target_index = int(target_indices_array[target_order_index])
            if target_index == query_index:
                continue

            payload = metadata_payload(metadata[target_index], target_index)
            payload["similarity"] = float(scores[target_order_index])
            results.append(payload)

            if len(results) >= k:
                break

        species_summary = {}
        for result in results:
            species = result["species"] or "unknown"
            if species not in species_summary:
                species_summary[species] = {
                    "species": species,
                    "label": prettify_species(species),
                    "count": 0,
                    "best_similarity": result["similarity"],
                }
            species_summary[species]["count"] += 1
            species_summary[species]["best_similarity"] = max(
                species_summary[species]["best_similarity"],
                result["similarity"],
            )

        queries.append(
            {
                "query": metadata_payload(metadata[query_index], query_index),
                "results": results,
                "species_summary": sorted(
                    species_summary.values(),
                    key=lambda item: item["best_similarity"],
                    reverse=True,
                ),
            }
        )

    queries.sort(key=lambda item: item["query"]["node_id"])

    return {
        "format": "nest_static_fossil_search_v1",
        "search_method": "cosine_similarity_on_stored_dinov3_embeddings",
        "target_set": "all_non_fossil_embeddings" if not include_fossil_targets else "all_embeddings",
        "k": int(k),
        "counts": {
            "queries": len(queries),
            "targets": len(target_indices),
        },
        "queries": queries,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def update_manifest(out_dir, output_name):
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    generated_files = manifest.setdefault("generated_files", {})
    generated_files["search_results"] = output_name

    notes = manifest.setdefault("notes", [])
    note = "search_results.json contains static fossil-pollen nearest-neighbor leaderboards"
    if note not in notes:
        notes.append(note)

    write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(
        description="Export static fossil-pollen query leaderboards from stored NEST embeddings"
    )
    parser.add_argument("--embeddings", "-e", default="embeddings/grain_embeddings.npy", help="Path to embeddings .npy")
    parser.add_argument("--out", "-o", default="exports/hypercube/search_results.json", help="Output JSON path")
    parser.add_argument("--k", type=int, default=10, help="Number of matches per fossil query")
    parser.add_argument(
        "--include-fossil-targets",
        action="store_true",
        help="Allow fossil nodes to appear in result leaderboards",
    )
    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    output_path = Path(args.out)

    embeddings, metadata = load_embeddings(embeddings_path)
    results = build_static_search_results(
        embeddings,
        metadata,
        k=args.k,
        include_fossil_targets=args.include_fossil_targets,
    )
    results["source_embeddings"] = str(embeddings_path).replace("\\", "/")

    write_json(output_path, results)
    update_manifest(output_path.parent, output_path.name)

    print(f"Exported static search results to: {output_path}")
    print(f"  fossil queries: {results['counts']['queries']}")
    print(f"  target embeddings: {results['counts']['targets']}")
    print(f"  matches per query: {results['k']}")


if __name__ == "__main__":
    main()
