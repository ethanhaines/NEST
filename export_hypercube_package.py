import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import faiss
except ImportError:
    faiss = None


def load_embeddings(embeddings_path: Path):
    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata_path = embeddings_path.with_suffix(".json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return embeddings, metadata


def normalize_path_for_web(path_value):
    if not path_value:
        return None
    return str(path_value).replace("\\", "/")


def build_species_palette(species_names):
    cmap = plt.get_cmap("tab20", max(len(species_names), 1))
    palette = {}
    for i, species in enumerate(species_names):
        r, g, b, a = cmap(i)
        palette[species] = {
            "rgba": [float(r), float(g), float(b), float(a)],
            "hex": "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)),
        }
    return palette


def compute_tsne_3d(embeddings, perplexity, n_iter, seed):
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=seed,
    )
    coords = tsne.fit_transform(embeddings).astype(np.float32)
    return coords


def normalize_coords(coords):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-8)

                                                                          
    centered = ((coords - mins) / spans) * 2.0 - 1.0
    return centered.astype(np.float32), mins, maxs


def normalize_coords_isotropic(coords, center_mode="mean"):
\
\
\
\
\
       
    x = np.asarray(coords, dtype=np.float32)
    if len(x) == 0:
        zero = np.zeros((0, 3), dtype=np.float32)
        return zero, {
            "center_mode": center_mode,
            "center": [0.0, 0.0, 0.0],
            "radius_scale": 1.0,
            "radius_max_before": 0.0,
            "bounds_before": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            "bounds_after": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
        }

    if center_mode == "median":
        center = np.median(x, axis=0).astype(np.float32)
    else:
        center = x.mean(axis=0).astype(np.float32)

    centered = (x - center).astype(np.float32)
    radii = np.linalg.norm(centered, axis=1)
    radius_max = float(np.max(radii)) if len(radii) else 0.0
    radius_scale = max(radius_max, 1e-8)
    x_norm = (centered / float(radius_scale)).astype(np.float32)

    return x_norm, {
        "center_mode": str(center_mode),
        "center": [float(v) for v in center],
        "radius_scale": float(radius_scale),
        "radius_max_before": float(radius_max),
        "bounds_before": {
            "min": [float(v) for v in x.min(axis=0)],
            "max": [float(v) for v in x.max(axis=0)],
        },
        "bounds_after": {
            "min": [float(v) for v in x_norm.min(axis=0)],
            "max": [float(v) for v in x_norm.max(axis=0)],
        },
    }


def compute_cloud_display_coords(coords_raw, whiten_strength=0.6, tanh_gain=1.2, z_boost=1.4):
\
\
\
\
\
\
\
\
\
       
    x = coords_raw.astype(np.float32).copy()

    med = np.median(x, axis=0)
    q25 = np.quantile(x, 0.25, axis=0)
    q75 = np.quantile(x, 0.75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    x = (x - med) / iqr

    cov = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)
    x = (x @ eigvecs).astype(np.float32)

    if whiten_strength > 0:
        scales = np.power(np.maximum(eigvals, 1e-8), float(whiten_strength) / 2.0).astype(np.float32)
        x /= scales

                                                               
    x[:, 2] *= float(z_boost)

    x = np.tanh(x * float(tanh_gain)).astype(np.float32)
    x_norm, norm_info = normalize_coords_isotropic(x, center_mode="mean")

    return x_norm, {
        "method": "robust_scale + pca + partial_whiten + tanh_clip + isotropic_radius_norm",
        "whiten_strength": float(whiten_strength),
        "tanh_gain": float(tanh_gain),
        "z_boost": float(z_boost),
        "robust_center_median": [float(v) for v in med],
        "robust_scale_iqr": [float(v) for v in iqr],
        "pca_eigenvalues": [float(v) for v in eigvals],
        "normalization": norm_info,
        "normalized_bounds": norm_info["bounds_after"],
    }


def _canonical_species_key(species_name):
    if species_name is None:
        return ""
    s = str(species_name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s


def _matching_species_indices(metadata, species_name):
    target_key = _canonical_species_key(species_name)
    matching = [
        i
        for i, m in enumerate(metadata)
        if _canonical_species_key(m.get("species")) == target_key
    ]
    return target_key, matching


def apply_species_display_axis_stretch(
    coords_display,
    metadata,
    species_name,
    stretch_factor=1.0,
    axis_index=1,
):
\
\
\
\
\
\
       
    if not species_name or float(stretch_factor) == 1.0:
        return coords_display, {"enabled": False, "applied": False}

    target_key, matching = _matching_species_indices(metadata, species_name)

    if not matching:
        return coords_display, {
            "enabled": True,
            "applied": False,
            "requested_species": str(species_name),
            "requested_species_canonical": target_key,
            "matched_nodes": 0,
            "reason": "species_not_found",
        }

    if len(matching) < 3:
        return coords_display, {
            "enabled": True,
            "applied": False,
            "requested_species": str(species_name),
            "requested_species_canonical": target_key,
            "matched_nodes": int(len(matching)),
            "reason": "too_few_nodes",
        }

    axis = int(np.clip(int(axis_index), 0, 2))
    factor = float(stretch_factor)

    out = coords_display.astype(np.float32, copy=True)
    idx = np.asarray(matching, dtype=np.int32)
    x = out[idx].astype(np.float32, copy=False)
    center = x.mean(axis=0).astype(np.float32)
    x0 = (x - center).astype(np.float32)

    cov = np.cov(x0, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)

                                                                
    local = (x0 @ eigvecs).astype(np.float32)
    std_before = np.std(local, axis=0).astype(np.float32)
    local[:, axis] *= factor
    std_after = np.std(local, axis=0).astype(np.float32)
    x_stretched = (local @ eigvecs.T).astype(np.float32) + center
    out[idx] = x_stretched

                                                                             
    out_norm, norm_info = normalize_coords_isotropic(out, center_mode="mean")

    matched_species_names = sorted({str(metadata[i].get("species")) for i in matching})
    return out_norm, {
        "enabled": True,
        "applied": True,
        "requested_species": str(species_name),
        "requested_species_canonical": target_key,
        "matched_species": matched_species_names,
        "matched_nodes": int(len(matching)),
        "axis_index": axis,
        "stretch_factor": factor,
        "local_pca_eigenvalues": [float(v) for v in eigvals],
        "local_std_before": [float(v) for v in std_before],
        "local_std_after": [float(v) for v in std_after],
        "renormalization": norm_info,
    }


def apply_species_display_cloudify(
    coords_display,
    metadata,
    species_name,
    cloudify_strength=0.0,
):
\
\
\
\
\
\
       
    if not species_name or float(cloudify_strength) <= 0:
        return coords_display, {"enabled": False, "applied": False}

    strength = float(np.clip(float(cloudify_strength), 0.0, 1.0))
    target_key, matching = _matching_species_indices(metadata, species_name)

    if not matching:
        return coords_display, {
            "enabled": True,
            "applied": False,
            "requested_species": str(species_name),
            "requested_species_canonical": target_key,
            "matched_nodes": 0,
            "reason": "species_not_found",
        }

    if len(matching) < 3:
        return coords_display, {
            "enabled": True,
            "applied": False,
            "requested_species": str(species_name),
            "requested_species_canonical": target_key,
            "matched_nodes": int(len(matching)),
            "reason": "too_few_nodes",
        }

    out = coords_display.astype(np.float32, copy=True)
    idx = np.asarray(matching, dtype=np.int32)
    x = out[idx].astype(np.float32, copy=False)
    center = x.mean(axis=0).astype(np.float32)
    x0 = (x - center).astype(np.float32)

    cov = np.cov(x0, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)

    local = (x0 @ eigvecs).astype(np.float32)
    std_before = np.std(local, axis=0).astype(np.float32)
    std_before_safe = np.clip(std_before, 1e-8, None).astype(np.float32)

    isotropic_target = np.float32(np.mean(std_before_safe))
    std_target = (1.0 - strength) * std_before_safe + strength * isotropic_target
    scales = (std_target / std_before_safe).astype(np.float32)

    local_cloud = (local * scales).astype(np.float32)
    std_after = np.std(local_cloud, axis=0).astype(np.float32)
    x_cloud = (local_cloud @ eigvecs.T).astype(np.float32) + center
    out[idx] = x_cloud

    out_norm, norm_info = normalize_coords_isotropic(out, center_mode="mean")
    matched_species_names = sorted({str(metadata[i].get("species")) for i in matching})
    anisotropy_before = float(np.max(std_before_safe) / max(float(np.min(std_before_safe)), 1e-8))
    anisotropy_after = float(np.max(std_after) / max(float(np.min(std_after)), 1e-8))

    return out_norm, {
        "enabled": True,
        "applied": True,
        "requested_species": str(species_name),
        "requested_species_canonical": target_key,
        "matched_species": matched_species_names,
        "matched_nodes": int(len(matching)),
        "cloudify_strength": float(strength),
        "local_pca_eigenvalues": [float(v) for v in eigvals],
        "local_std_before": [float(v) for v in std_before],
        "local_std_after": [float(v) for v in std_after],
        "axis_scales_applied": [float(v) for v in scales],
        "anisotropy_ratio_before": anisotropy_before,
        "anisotropy_ratio_after": anisotropy_after,
        "renormalization": norm_info,
    }


def normalize_embeddings_for_cosine(embeddings):
    vecs = embeddings.astype(np.float32).copy()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= np.clip(norms, 1e-12, None)
    return vecs


def _parse_force_bridge_pairs(force_bridge_specs, num_nodes):
    parsed = []
    for spec in force_bridge_specs or []:
        raw = str(spec).strip()
        if not raw:
            continue
        for sep in (":", ",", "-"):
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep)]
                break
        else:
            raise ValueError(
                f"Invalid --force-bridge '{raw}'. Use 'src:dst' (example: 803:66)."
            )

        if len(parts) != 2:
            raise ValueError(
                f"Invalid --force-bridge '{raw}'. Use exactly two node ids like '803:66'."
            )

        a = int(parts[0])
        b = int(parts[1])
        if a == b:
            raise ValueError(f"Invalid --force-bridge '{raw}': source and target are identical.")
        if not (0 <= a < num_nodes and 0 <= b < num_nodes):
            raise ValueError(
                f"Invalid --force-bridge '{raw}': node ids must be in [0, {num_nodes - 1}]."
            )
        parsed.append(tuple(sorted((a, b))))

                               
    unique = []
    seen = set()
    for pair in parsed:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    return unique


def add_manual_bridge_edges(embeddings, base_edges, force_bridge_specs):
    if not force_bridge_specs:
        return base_edges, {"enabled": False, "added_edges": 0}

    num_nodes = len(embeddings)
    force_pairs = _parse_force_bridge_pairs(force_bridge_specs, num_nodes)
    if not force_pairs:
        return base_edges, {"enabled": False, "added_edges": 0}

    vecs = normalize_embeddings_for_cosine(embeddings)
    seen_pairs = {tuple(sorted((int(e["source"]), int(e["target"])))) for e in base_edges}
    edges = list(base_edges)
    added = 0
    skipped_existing = 0

    for idx, (u, v) in enumerate(force_pairs, start=1):
        if (u, v) in seen_pairs:
            skipped_existing += 1
            continue
        seen_pairs.add((u, v))
        sim = float(np.dot(vecs[u], vecs[v]))
        edges.append(
            {
                "source": int(u),
                "target": int(v),
                "similarity": sim,
                "rank_hint": 1,
                "edge_kind": "manual_bridge",
                "bridge_group": idx,
                "bridge_basis": "explicit_user_pair",
            }
        )
        added += 1

    stats = {
        "enabled": True,
        "requested_pairs": len(force_pairs),
        "added_edges": int(added),
        "skipped_existing": int(skipped_existing),
        "pairs": [[int(a), int(b)] for a, b in force_pairs],
    }
    return edges, stats


def build_knn_edges(embeddings, k, vecs=None):
    if len(embeddings) == 0:
        return []

    vecs = normalize_embeddings_for_cosine(embeddings) if vecs is None else vecs

    if faiss is not None:
        faiss.normalize_L2(vecs)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        sims, idxs = index.search(vecs, k + 1)
    else:
        sims = vecs @ vecs.T
        idxs = np.argsort(-sims, axis=1)[:, : k + 1]
        sims = np.take_along_axis(sims, idxs, axis=1)

    seen = set()
    edges = []
    for src in range(len(vecs)):
        rank = 0
        for sim, dst in zip(sims[src], idxs[src]):
            dst = int(dst)
            if dst == src:
                continue
            a, b = sorted((src, dst))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            rank += 1
            edges.append(
                {
                    "source": a,
                    "target": b,
                    "similarity": float(sim),
                    "rank_hint": rank,
                    "edge_kind": "knn",
                }
            )
    return edges


def find_connected_components(num_nodes, edges):
    if num_nodes == 0:
        return []

    adjacency = [[] for _ in range(num_nodes)]
    for edge in edges:
        a = int(edge["source"])
        b = int(edge["target"])
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen = np.zeros(num_nodes, dtype=bool)
    components = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adjacency[node]:
                if seen[nxt]:
                    continue
                seen[nxt] = True
                stack.append(nxt)
        components.append(comp)

    components.sort(key=len, reverse=True)
    return components


def _component_centroid(vecs, node_ids):
    center = vecs[np.asarray(node_ids, dtype=np.int32)].mean(axis=0)
    norm = np.linalg.norm(center)
    if norm > 1e-12:
        center = center / norm
    return center.astype(np.float32)


def _prim_component_mst(component_centers):
    n = len(component_centers)
    if n <= 1:
        return []

    used = [False] * n
    used[0] = True
    links = []

    for _ in range(n - 1):
        best = None
        for i in range(n):
            if not used[i]:
                continue
            for j in range(n):
                if used[j]:
                    continue
                sim = float(np.dot(component_centers[i], component_centers[j]))
                if best is None or sim > best[0]:
                    best = (sim, i, j)
        if best is None:
            break
        sim, i, j = best
        used[j] = True
        links.append((i, j, sim))

    return links


def _prim_component_mst_by_display_distance(component_centers):
    n = len(component_centers)
    if n <= 1:
        return []

    used = [False] * n
    used[0] = True
    links = []

    for _ in range(n - 1):
        best = None
        for i in range(n):
            if not used[i]:
                continue
            for j in range(n):
                if used[j]:
                    continue
                dist = float(np.linalg.norm(component_centers[i] - component_centers[j]))
                if best is None or dist < best[0]:
                    best = (dist, i, j)
        if best is None:
            break
        dist, i, j = best
        used[j] = True
        links.append((dist, i, j))

    return links


def _top_cross_component_pairs(vecs, comp_a, comp_b, limit, seen_pairs):
    a = np.asarray(comp_a, dtype=np.int32)
    b = np.asarray(comp_b, dtype=np.int32)
    sims = vecs[a] @ vecs[b].T

                                                                                  
    order = np.argsort(sims, axis=None)[::-1]
    picked = []
    for flat_idx in order:
        ia, ib = np.unravel_index(int(flat_idx), sims.shape)
        src = int(a[ia])
        dst = int(b[ib])
        u, v = sorted((src, dst))
        if (u, v) in seen_pairs:
            continue
        picked.append((u, v, float(sims[ia, ib])))
        if len(picked) >= limit:
            break
    return picked


def _top_boundary_facing_cross_component_pairs(
    vecs,
    coords_display,
    comp_a,
    comp_b,
    limit,
    seen_pairs,
    pool_size=24,
    source_usage_counts=None,
    target_usage_counts=None,
    source_reuse_penalty=0.06,
    target_reuse_penalty=0.03,
):
\
\
\
       
    a_all = np.asarray(comp_a, dtype=np.int32)
    b_all = np.asarray(comp_b, dtype=np.int32)
    if len(a_all) == 0 or len(b_all) == 0 or limit <= 0:
        return []

    a_xyz = coords_display[a_all].astype(np.float32, copy=False)
    b_xyz = coords_display[b_all].astype(np.float32, copy=False)
    center_a = a_xyz.mean(axis=0).astype(np.float32)
    center_b = b_xyz.mean(axis=0).astype(np.float32)

    ab = center_b - center_a
    ab_norm = float(np.linalg.norm(ab))
    if ab_norm > 1e-12:
        dir_ab = (ab / ab_norm).astype(np.float32)
        dir_ba = (-dir_ab).astype(np.float32)
    else:
        dir_ab = np.zeros(3, dtype=np.float32)
        dir_ba = np.zeros(3, dtype=np.float32)

    def _boundary_candidate_scores(points_xyz, own_center, other_center, facing_dir):
        deltas = points_xyz - own_center
        local_r = np.linalg.norm(deltas, axis=1)
        local_r_max = max(float(local_r.max()) if len(local_r) else 0.0, 1e-8)
        edge_norm = (local_r / local_r_max).astype(np.float32)

        if np.any(facing_dir):
            facing_proj = deltas @ facing_dir
            facing_pos = np.clip(facing_proj, 0.0, None)
            facing_max = max(float(facing_pos.max()) if len(facing_pos) else 0.0, 1e-8)
            facing_norm = (facing_pos / facing_max).astype(np.float32)
        else:
            facing_norm = np.zeros(len(points_xyz), dtype=np.float32)

        dist_to_other = np.linalg.norm(points_xyz - other_center, axis=1)
        if len(dist_to_other):
            d_min = float(dist_to_other.min())
            d_span = max(float(dist_to_other.max() - d_min), 1e-8)
            close_norm = (1.0 - ((dist_to_other - d_min) / d_span)).astype(np.float32)
        else:
            close_norm = np.zeros(0, dtype=np.float32)

        score = (0.5 * facing_norm + 0.3 * edge_norm + 0.2 * close_norm).astype(np.float32)
        return score, edge_norm

    a_scores, a_edge_norm = _boundary_candidate_scores(a_xyz, center_a, center_b, dir_ab)
    b_scores, b_edge_norm = _boundary_candidate_scores(b_xyz, center_b, center_a, dir_ba)

    a_pool_n = min(len(a_all), max(int(pool_size), int(limit) * 4))
    b_pool_n = min(len(b_all), max(int(pool_size), int(limit) * 4))
    a_order = np.argsort(-a_scores)[:a_pool_n]
    b_order = np.argsort(-b_scores)[:b_pool_n]
    a_pool = a_all[a_order]
    b_pool = b_all[b_order]

    a_pool_xyz = coords_display[a_pool].astype(np.float32, copy=False)
    b_pool_xyz = coords_display[b_pool].astype(np.float32, copy=False)
    sims = (vecs[a_pool] @ vecs[b_pool].T).astype(np.float32)
    pair_dists = np.linalg.norm(a_pool_xyz[:, None, :] - b_pool_xyz[None, :, :], axis=2).astype(np.float32)

    if pair_dists.size:
        d_min = float(pair_dists.min())
        d_span = max(float(pair_dists.max() - d_min), 1e-8)
        spatial_closeness = (1.0 - ((pair_dists - d_min) / d_span)).astype(np.float32)
    else:
        spatial_closeness = np.zeros_like(pair_dists, dtype=np.float32)

    a_pool_edge = a_edge_norm[a_order] if len(a_order) else np.zeros(0, dtype=np.float32)
    b_pool_edge = b_edge_norm[b_order] if len(b_order) else np.zeros(0, dtype=np.float32)

    src_usage = {} if source_usage_counts is None else source_usage_counts
    dst_usage = {} if target_usage_counts is None else target_usage_counts
    local_src_usage = {}
    local_dst_usage = {}
    blocked = set()
    picked = []

    while len(picked) < limit:
        best = None
        for ia, src in enumerate(a_pool):
            src = int(src)
            for ib, dst in enumerate(b_pool):
                dst = int(dst)
                u, v = sorted((src, dst))
                if (u, v) in seen_pairs or (u, v) in blocked:
                    continue

                cosine_sim = float(sims[ia, ib])
                boundary_bonus = 0.5 * float(a_pool_edge[ia]) + 0.5 * float(b_pool_edge[ib])
                score = (
                    0.62 * cosine_sim
                    + 0.28 * float(spatial_closeness[ia, ib])
                    + 0.10 * float(boundary_bonus)
                )
                score -= float(source_reuse_penalty) * float(src_usage.get(src, 0) + local_src_usage.get(src, 0))
                score -= float(target_reuse_penalty) * float(dst_usage.get(dst, 0) + local_dst_usage.get(dst, 0))

                if best is None or score > best[0]:
                    best = (
                        float(score),
                        src,
                        dst,
                        cosine_sim,
                        float(pair_dists[ia, ib]),
                        float(boundary_bonus),
                    )

        if best is None:
            break

        score, src, dst, cosine_sim, display_dist, boundary_bonus = best
        u, v = sorted((src, dst))
        blocked.add((u, v))
        picked.append((u, v, cosine_sim, display_dist, boundary_bonus, score))
        local_src_usage[src] = local_src_usage.get(src, 0) + 1
        local_dst_usage[dst] = local_dst_usage.get(dst, 0) + 1

    return picked


def _greedy_antipode_component_pairs(coords_display, components):
    if len(components) <= 1:
        return [], {"global_center": [0.0, 0.0, 0.0], "candidate_pairs": 0}

    global_center = coords_display.mean(axis=0).astype(np.float32)
    centers = []
    directions = []

    for comp in components:
        comp_arr = np.asarray(comp, dtype=np.int32)
        center = coords_display[comp_arr].mean(axis=0).astype(np.float32)
        vec = center - global_center
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            direction = (vec / norm).astype(np.float32)
        else:
            direction = np.zeros(3, dtype=np.float32)
        centers.append(center)
        directions.append(direction)

    candidates = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            dot = float(np.dot(directions[i], directions[j]))
            dist = float(np.linalg.norm(centers[i] - centers[j]))
                                                                                  
            candidates.append((dot, -dist, i, j))

    candidates.sort()
    used = set()
    pairs = []
    for dot, neg_dist, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append(
            {
                "component_a": i,
                "component_b": j,
                "centroid_dot": float(dot),
                "centroid_distance": float(-neg_dist),
            }
        )

    return pairs, {
        "global_center": [float(v) for v in global_center],
        "candidate_pairs": len(candidates),
    }


def _top_inward_antipode_pairs(
    vecs,
    coords_display,
    comp_a,
    comp_b,
    global_center,
    limit,
    inward_pool,
    seen_pairs,
):
    a_all = np.asarray(comp_a, dtype=np.int32)
    b_all = np.asarray(comp_b, dtype=np.int32)

    a_d = np.linalg.norm(coords_display[a_all] - global_center, axis=1)
    b_d = np.linalg.norm(coords_display[b_all] - global_center, axis=1)

    a_pool_n = min(len(a_all), max(1, int(inward_pool)))
    b_pool_n = min(len(b_all), max(1, int(inward_pool)))
    a_pool = a_all[np.argsort(a_d)[:a_pool_n]]
    b_pool = b_all[np.argsort(b_d)[:b_pool_n]]

    sims = vecs[a_pool] @ vecs[b_pool].T
    order = np.argsort(sims, axis=None)[::-1]

    picked = []
    for flat_idx in order:
        ia, ib = np.unravel_index(int(flat_idx), sims.shape)
        src = int(a_pool[ia])
        dst = int(b_pool[ib])
        u, v = sorted((src, dst))
        if (u, v) in seen_pairs:
            continue
        picked.append((u, v, float(sims[ia, ib])))
        if len(picked) >= limit:
            break

    return picked


def _top_inward_pairs_diversified(
    vecs,
    coords_display,
    comp_a,
    comp_b,
    global_center,
    limit,
    inward_pool,
    seen_pairs,
    source_usage_counts=None,
    target_usage_counts=None,
    source_reuse_penalty=0.08,
    target_reuse_penalty=0.03,
    source_edge_bias=0.0,
):
\
\
\
       
    a_all = np.asarray(comp_a, dtype=np.int32)
    b_all = np.asarray(comp_b, dtype=np.int32)

    a_d = np.linalg.norm(coords_display[a_all] - global_center, axis=1)
    b_d = np.linalg.norm(coords_display[b_all] - global_center, axis=1)

    a_pool_n = min(len(a_all), max(1, int(inward_pool)))
    b_pool_n = min(len(b_all), max(1, int(inward_pool)))
    a_inward_pool = a_all[np.argsort(a_d)[:a_pool_n]]
    b_pool = b_all[np.argsort(b_d)[:b_pool_n]]

                                                                                
                                                                              
                                   
    a_pool = a_inward_pool
    source_edge_norm = {}
    if float(source_edge_bias) > 0 and len(a_all) > 1:
        src_center = coords_display[a_all].mean(axis=0)
        a_local_r = np.linalg.norm(coords_display[a_all] - src_center, axis=1)
        local_max = float(np.max(a_local_r)) if len(a_local_r) else 0.0
        local_max = max(local_max, 1e-8)

        a_edge_pool = a_all[np.argsort(-a_local_r)[:a_pool_n]]

        merged = []
        seen_src = set()
        for src_id in np.concatenate([a_inward_pool, a_edge_pool]):
            src_id = int(src_id)
            if src_id in seen_src:
                continue
            seen_src.add(src_id)
            merged.append(src_id)
        a_pool = np.asarray(merged, dtype=np.int32)

                                                   
        local_r_by_id = {int(node_id): float(rad / local_max) for node_id, rad in zip(a_all, a_local_r)}
        source_edge_norm = {int(node_id): local_r_by_id.get(int(node_id), 0.0) for node_id in a_pool}

    sims = vecs[a_pool] @ vecs[b_pool].T

    src_usage = {} if source_usage_counts is None else source_usage_counts
    dst_usage = {} if target_usage_counts is None else target_usage_counts
    local_src_usage = {}
    local_dst_usage = {}

    picked = []
    blocked = set()

    while len(picked) < limit:
        best = None
        for ia, src in enumerate(a_pool):
            src = int(src)
            for ib, dst in enumerate(b_pool):
                dst = int(dst)
                u, v = sorted((src, dst))
                if (u, v) in seen_pairs or (u, v) in blocked:
                    continue

                score = float(sims[ia, ib])
                score -= float(source_reuse_penalty) * float(src_usage.get(src, 0) + local_src_usage.get(src, 0))
                score -= float(target_reuse_penalty) * float(dst_usage.get(dst, 0) + local_dst_usage.get(dst, 0))
                if float(source_edge_bias) > 0:
                    score += float(source_edge_bias) * float(source_edge_norm.get(src, 0.0))

                if best is None or score > best[0]:
                    best = (score, src, dst, float(sims[ia, ib]))

        if best is None:
            break

        _, src, dst, sim = best
        u, v = sorted((src, dst))
        picked.append((u, v, sim))
        blocked.add((u, v))
        local_src_usage[src] = local_src_usage.get(src, 0) + 1
        local_dst_usage[dst] = local_dst_usage.get(dst, 0) + 1

    return picked


def _group_node_indices_by_species(metadata):
    groups = {}
    for idx, m in enumerate(metadata):
        species = (m.get("species") or "unknown")
        groups.setdefault(species, []).append(idx)
    return groups


def _largest_remainder_counts(weights, total_count):
    if total_count <= 0:
        return [0] * len(weights)
    if not weights:
        return []

    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        base = np.zeros(len(weights), dtype=np.int32)
        base[: total_count % len(weights)] = 1
        full_cycles = total_count // len(weights)
        base += full_cycles
        return [int(x) for x in base]

    raw = weights / weight_sum * float(total_count)
    counts = np.floor(raw).astype(np.int32)
    remainder = int(total_count - counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return [int(x) for x in counts]


def add_species_ratio_bridge_edges(
    embeddings,
    metadata,
    coords_display,
    base_edges,
    species_bridge_ratio=0.0,
    inward_pool=12,
    antipode_bias=0.0,
    source_isolation_boost=0.0,
    source_edge_bias=0.0,
):
\
\
\
\
\
\
       
    if species_bridge_ratio <= 0:
        return base_edges, {"enabled": False, "added_edges": 0}

    species_groups = _group_node_indices_by_species(metadata)
    species_names = sorted(species_groups.keys())
    num_species = len(species_names)
    if num_species <= 1:
        return base_edges, {
            "enabled": True,
            "strategy": "species_size_ratio_all_to_all",
            "species_count": num_species,
            "added_edges": 0,
        }

    vecs = normalize_embeddings_for_cosine(embeddings)
    coords_display = coords_display.astype(np.float32, copy=False)
    global_center = coords_display.mean(axis=0).astype(np.float32)
    species_centers = {}
    species_dirs = {}
    species_radii = {}
    max_species_radius = 1e-8

    for name in species_names:
        idxs = np.asarray(species_groups[name], dtype=np.int32)
        center = coords_display[idxs].mean(axis=0).astype(np.float32)
        delta = center - global_center
        radius = float(np.linalg.norm(delta))
        if radius > 1e-12:
            direction = (delta / radius).astype(np.float32)
        else:
            direction = np.zeros(3, dtype=np.float32)
        species_centers[name] = center
        species_dirs[name] = direction
        species_radii[name] = radius
        max_species_radius = max(max_species_radius, radius)

                                                                
                                                                    
    targets_per_source = num_species - 1
    species_sizes = {name: len(species_groups[name]) for name in species_names}
    seen_pairs = {tuple(sorted((int(e["source"]), int(e["target"])))) for e in base_edges}
    edges = list(base_edges)
    added = 0
    attempted_species_pairs = 0
    source_budgets = {}
    source_node_usage_by_species = {name: {} for name in species_names}
    target_node_usage_by_species = {name: {} for name in species_names}

    for source_species in species_names:
        target_species = [name for name in species_names if name != source_species]
        if not target_species:
            continue

        isolation_norm = species_radii[source_species] / max_species_radius if max_species_radius > 0 else 0.0
        source_budget_multiplier = 1.0 + float(source_isolation_boost) * float(isolation_norm)
        per_source_budget = max(
            targets_per_source,
            int(round(num_species * float(species_bridge_ratio) * source_budget_multiplier)),
        )
        source_budgets[source_species] = int(per_source_budget)

                                                                                      
        pair_counts = [1] * len(target_species)
        remaining = per_source_budget - len(target_species)
        if remaining > 0:
            target_weights = []
            src_dir = species_dirs[source_species]
            for name in target_species:
                weight = float(species_sizes[name])
                if antipode_bias > 0:
                    dot = float(np.dot(src_dir, species_dirs[name]))
                    antipode_score = (1.0 - dot) * 0.5                                
                    weight *= 1.0 + float(antipode_bias) * antipode_score
                target_weights.append(weight)
            extra_counts = _largest_remainder_counts(
                target_weights,
                remaining,
            )
            pair_counts = [a + b for a, b in zip(pair_counts, extra_counts)]

        src_nodes = species_groups[source_species]
        for target_name, edge_count in zip(target_species, pair_counts):
            if edge_count <= 0:
                continue
            attempted_species_pairs += 1
            dst_nodes = species_groups[target_name]

                                                                             
                                                                               
            top_pairs = _top_inward_pairs_diversified(
                vecs,
                coords_display,
                src_nodes,
                dst_nodes,
                global_center=global_center,
                limit=int(edge_count),
                inward_pool=max(1, int(inward_pool)),
                seen_pairs=seen_pairs,
                source_usage_counts=source_node_usage_by_species[source_species],
                target_usage_counts=target_node_usage_by_species[target_name],
                source_edge_bias=source_edge_bias,
            )

            for rank, (u, v, sim) in enumerate(top_pairs, start=1):
                seen_pairs.add((u, v))
                source_node_usage_by_species[source_species][u] = (
                    source_node_usage_by_species[source_species].get(u, 0) + 1
                )
                target_node_usage_by_species[target_name][v] = (
                    target_node_usage_by_species[target_name].get(v, 0) + 1
                )
                edges.append(
                    {
                        "source": int(u),
                        "target": int(v),
                        "similarity": float(sim),
                        "rank_hint": rank,
                        "edge_kind": "species_ratio_bridge",
                        "bridge_basis": "species_quota_by_receiving_size + embedding_cosine",
                        "bridge_source_species": source_species,
                        "bridge_target_species": target_name,
                    }
                )
                added += 1

    stats = {
        "enabled": True,
        "strategy": "species_size_ratio_all_to_all",
        "species_count": int(num_species),
        "species_bridge_ratio": float(species_bridge_ratio),
        "source_budgets": source_budgets,
        "species_bridge_antipode_bias": float(antipode_bias),
        "species_bridge_source_isolation_boost": float(source_isolation_boost),
        "species_bridge_source_edge_bias": float(source_edge_bias),
        "inward_pool": int(inward_pool),
        "attempted_species_pairs": int(attempted_species_pairs),
        "added_edges": int(added),
        "bridge_source_diversification": {
            "strategy": "reuse_penalty",
            "source_reuse_penalty": 0.08,
            "target_reuse_penalty": 0.03,
        },
    }
    return edges, stats


def add_component_bridge_edges(embeddings, base_edges, bridges_per_link=1, coords_display=None):
\
\
\
\
\
\
\
\
       
    if bridges_per_link <= 0:
        return base_edges, {"enabled": False, "added_edges": 0, "components_before": None, "components_after": None}

    num_nodes = len(embeddings)
    components = find_connected_components(num_nodes, base_edges)
    if len(components) <= 1:
        return base_edges, {
            "enabled": True,
            "strategy": "component_mst_cosine_bridges",
            "bridges_per_link": int(bridges_per_link),
            "components_before": len(components),
            "components_after": len(components),
            "added_edges": 0,
            "mst_component_links": 0,
        }

    vecs = normalize_embeddings_for_cosine(embeddings)
    embedding_centers = [_component_centroid(vecs, comp) for comp in components]
    coords_display_valid = (
        coords_display is not None
        and len(coords_display) == num_nodes
    )
    display_centers = None
    if coords_display_valid:
        coords_display = np.asarray(coords_display, dtype=np.float32)
        display_centers = [
            coords_display[np.asarray(comp, dtype=np.int32)].mean(axis=0).astype(np.float32)
            for comp in components
        ]
        mst_links = _prim_component_mst_by_display_distance(display_centers)
        mst_link_mode = "display_centroid_distance"
    else:
        mst_links = _prim_component_mst(embedding_centers)
        mst_link_mode = "embedding_centroid_cosine"

    seen_pairs = {tuple(sorted((int(e["source"]), int(e["target"])))) for e in base_edges}
    edges = list(base_edges)
    added = 0
    node_bridge_usage = {}

    for link_index, link in enumerate(mst_links, start=1):
        if coords_display_valid:
            mst_metric, ci, cj = float(link[0]), int(link[1]), int(link[2])
        else:
            mst_metric, ci, cj = float(link[0]), int(link[1]), int(link[2])
        centroid_sim = float(np.dot(embedding_centers[ci], embedding_centers[cj]))

        top_pairs = []
        if coords_display_valid:
            top_pairs = _top_boundary_facing_cross_component_pairs(
                vecs,
                coords_display,
                components[ci],
                components[cj],
                limit=max(1, int(bridges_per_link)),
                seen_pairs=seen_pairs,
                pool_size=max(20, int(bridges_per_link) * 6),
                source_usage_counts=node_bridge_usage,
                target_usage_counts=node_bridge_usage,
            )

        if not top_pairs:
            fallback_pairs = _top_cross_component_pairs(
                vecs,
                components[ci],
                components[cj],
                limit=max(1, int(bridges_per_link)),
                seen_pairs=seen_pairs,
            )
            top_pairs = [
                (u, v, sim, None, None, None)
                for (u, v, sim) in fallback_pairs
            ]

        for rank, pair in enumerate(top_pairs, start=1):
            u, v, sim, display_dist, boundary_bonus, selection_score = pair
            seen_pairs.add((u, v))
            node_bridge_usage[int(u)] = node_bridge_usage.get(int(u), 0) + 1
            node_bridge_usage[int(v)] = node_bridge_usage.get(int(v), 0) + 1
            edge_payload = {
                "source": u,
                "target": v,
                "similarity": float(sim),
                "rank_hint": rank,
                "edge_kind": "component_bridge",
                "bridge_group": link_index,
                "bridge_basis": (
                    "component_display_mst + boundary_facing_display_proximity + embedding_cosine"
                    if coords_display_valid
                    else "embedding_cosine"
                ),
                "component_centroid_similarity": float(centroid_sim),
            }
            if coords_display_valid and display_centers is not None:
                edge_payload["component_centroid_display_distance"] = float(mst_metric)
            if display_dist is not None:
                edge_payload["display_distance"] = float(display_dist)
            if boundary_bonus is not None:
                edge_payload["boundary_bonus"] = float(boundary_bonus)
            if selection_score is not None:
                edge_payload["selection_score"] = float(selection_score)

            edges.append(edge_payload)
            added += 1

    components_after = find_connected_components(num_nodes, edges)
    stats = {
        "enabled": True,
        "strategy": "component_mst_boundary_proximity_bridges",
        "mst_link_mode": mst_link_mode,
        "bridges_per_link": int(bridges_per_link),
        "components_before": len(components),
        "components_after": len(components_after),
        "added_edges": int(added),
        "mst_component_links": len(mst_links),
    }
    return edges, stats


def add_spatial_antipode_bridge_edges(
    embeddings,
    coords_display,
    base_edges,
    bridges_per_pair=1,
    inward_pool=12,
):
\
\
\
\
\
\
\
       
    if bridges_per_pair <= 0:
        return base_edges, {"enabled": False, "added_edges": 0}

    num_nodes = len(embeddings)
    components = find_connected_components(num_nodes, base_edges)
    if len(components) <= 1:
        return base_edges, {
            "enabled": True,
            "strategy": "spatial_antipode_component_pairs",
            "components_before": len(components),
            "pairs_selected": 0,
            "bridges_per_pair": int(bridges_per_pair),
            "inward_pool": int(inward_pool),
            "added_edges": 0,
        }

    vecs = normalize_embeddings_for_cosine(embeddings)
    coords_display = coords_display.astype(np.float32, copy=False)

    pair_specs, pair_meta = _greedy_antipode_component_pairs(coords_display, components)
    global_center = np.asarray(pair_meta["global_center"], dtype=np.float32)

    seen_pairs = {tuple(sorted((int(e["source"]), int(e["target"])))) for e in base_edges}
    edges = list(base_edges)
    added = 0

    for pair_index, spec in enumerate(pair_specs, start=1):
        ci = spec["component_a"]
        cj = spec["component_b"]
        top_pairs = _top_inward_antipode_pairs(
            vecs,
            coords_display,
            components[ci],
            components[cj],
            global_center=global_center,
            limit=max(1, int(bridges_per_pair)),
            inward_pool=max(1, int(inward_pool)),
            seen_pairs=seen_pairs,
        )

        for rank, (u, v, sim) in enumerate(top_pairs, start=1):
            seen_pairs.add((u, v))
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "similarity": float(sim),
                    "rank_hint": rank,
                    "edge_kind": "spatial_antipode_bridge",
                    "bridge_group": pair_index,
                    "bridge_basis": "spatial_antipode_components + embedding_cosine",
                    "antipode_centroid_dot": float(spec["centroid_dot"]),
                    "antipode_centroid_distance": float(spec["centroid_distance"]),
                }
            )
            added += 1

    stats = {
        "enabled": True,
        "strategy": "spatial_antipode_component_pairs",
        "components_before": len(components),
        "pairs_selected": len(pair_specs),
        "bridges_per_pair": int(bridges_per_pair),
        "inward_pool": int(inward_pool),
        "added_edges": int(added),
        "global_center": pair_meta["global_center"],
    }
    return edges, stats


def build_nodes(metadata, coords_raw, coords_tsne_norm, coords_cloud_norm):
    nodes = []
    for i, (m, raw, tsne_norm, cloud_norm) in enumerate(
        zip(metadata, coords_raw, coords_tsne_norm, coords_cloud_norm)
    ):
        path_value = normalize_path_for_web(m.get("path"))
        source_folder = normalize_path_for_web(m.get("source_folder"))
        node = {
            "id": i,
            "species": m.get("species"),
            "filename": m.get("filename"),
            "crop_size": m.get("crop_size"),
            "source_folder": source_folder,
            "path": path_value,
                                                                      
            "position": {"x": float(cloud_norm[0]), "y": float(cloud_norm[1]), "z": float(cloud_norm[2])},
            "position_cloud": {
                "x": float(cloud_norm[0]),
                "y": float(cloud_norm[1]),
                "z": float(cloud_norm[2]),
            },
                                                                          
            "position_tsne": {
                "x": float(tsne_norm[0]),
                "y": float(tsne_norm[1]),
                "z": float(tsne_norm[2]),
            },
            "position_raw": {"x": float(raw[0]), "y": float(raw[1]), "z": float(raw[2])},
        }
        nodes.append(node)
    return nodes


def build_species_summary(nodes, palette):
    summary = {}
    for node in nodes:
        species = node["species"] or "unknown"
        info = summary.setdefault(
            species,
            {
                "species": species,
                "count": 0,
                "crop_sizes": {},
                "color": palette.get(species, {"hex": "#cccccc", "rgba": [0.8, 0.8, 0.8, 1.0]}),
            },
        )
        info["count"] += 1
        crop = node.get("crop_size") or "unknown"
        info["crop_sizes"][crop] = info["crop_sizes"].get(crop, 0) + 1
    return [summary[k] for k in sorted(summary.keys())]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_package(
    embeddings_path,
    out_dir,
    perplexity,
    n_iter,
    k_neighbors,
    seed,
    cloud_whiten,
    cloud_tanh_gain,
    cloud_z_boost,
    force_bridges,
    species_bridge_ratio,
    species_bridge_inward_pool,
    species_bridge_antipode_bias,
    species_bridge_source_isolation_boost,
    species_bridge_source_edge_bias,
    spatial_antipode_bridge_per_pair,
    spatial_antipode_inward_pool,
    component_bridge_per_link,
    display_cloudify_species,
    display_cloudify_strength,
    display_stretch_species,
    display_stretch_factor,
    display_stretch_axis,
):
    embeddings_path = Path(embeddings_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_embeddings(embeddings_path)
    if len(embeddings) != len(metadata):
        raise ValueError(f"Embeddings/metadata length mismatch: {len(embeddings)} vs {len(metadata)}")

    print(f"Loaded {len(embeddings)} embeddings from {embeddings_path}")
    print(f"Running 3D t-SNE (perplexity={perplexity}, iterations={n_iter}, seed={seed})...")
    coords_raw = compute_tsne_3d(embeddings, perplexity, n_iter, seed)
    coords_tsne_norm, mins, maxs = normalize_coords(coords_raw)
    coords_cloud, cloud_transform = compute_cloud_display_coords(
        coords_raw,
        whiten_strength=cloud_whiten,
        tanh_gain=cloud_tanh_gain,
        z_boost=cloud_z_boost,
    )

    print(f"Building k-NN edges (k={k_neighbors})...")
    edges = build_knn_edges(embeddings, k_neighbors)

    species_ratio_bridge_stats = {"enabled": False, "added_edges": 0}
    if species_bridge_ratio > 0:
        print(
            "Adding species ratio bridges "
            f"(ratio={species_bridge_ratio}, inward pool={species_bridge_inward_pool})..."
        )
        edges, species_ratio_bridge_stats = add_species_ratio_bridge_edges(
            embeddings,
            metadata,
            coords_cloud,
            edges,
            species_bridge_ratio=species_bridge_ratio,
            inward_pool=species_bridge_inward_pool,
            antipode_bias=species_bridge_antipode_bias,
            source_isolation_boost=species_bridge_source_isolation_boost,
            source_edge_bias=species_bridge_source_edge_bias,
        )

    manual_bridge_stats = {"enabled": False, "added_edges": 0}
    if force_bridges:
        print(f"Adding explicit manual bridge edges ({len(force_bridges)} requested pair(s))...")
        edges, manual_bridge_stats = add_manual_bridge_edges(embeddings, edges, force_bridges)

    antipode_bridge_stats = {"enabled": False, "added_edges": 0}
    if spatial_antipode_bridge_per_pair > 0:
        print(
            "Adding spatial antipode bridge edges "
            f"(per paired island={spatial_antipode_bridge_per_pair}, inward pool={spatial_antipode_inward_pool})..."
        )
        edges, antipode_bridge_stats = add_spatial_antipode_bridge_edges(
            embeddings,
            coords_cloud,
            edges,
            bridges_per_pair=spatial_antipode_bridge_per_pair,
            inward_pool=spatial_antipode_inward_pool,
        )

    bridge_stats = {"enabled": False, "added_edges": 0}
    if component_bridge_per_link > 0:
        print(
            "Adding component bridge edges "
            f"(per component MST link={component_bridge_per_link})..."
        )
        edges, bridge_stats = add_component_bridge_edges(
            embeddings,
            edges,
            bridges_per_link=component_bridge_per_link,
            coords_display=coords_cloud,
        )

    species_display_cloudify_stats = {"enabled": False, "applied": False}
    if display_cloudify_species and float(display_cloudify_strength) > 0:
        print(
            "Applying display-only species cloudify warp "
            f"(species={display_cloudify_species}, strength={display_cloudify_strength})..."
        )
        coords_cloud, species_display_cloudify_stats = apply_species_display_cloudify(
            coords_cloud,
            metadata,
            species_name=display_cloudify_species,
            cloudify_strength=display_cloudify_strength,
        )

    species_display_stretch_stats = {"enabled": False, "applied": False}
    if display_stretch_species and float(display_stretch_factor) != 1.0:
        print(
            "Applying display-only species axis stretch "
            f"(species={display_stretch_species}, factor={display_stretch_factor}, axis={display_stretch_axis})..."
        )
        coords_cloud, species_display_stretch_stats = apply_species_display_axis_stretch(
            coords_cloud,
            metadata,
            species_name=display_stretch_species,
            stretch_factor=display_stretch_factor,
            axis_index=display_stretch_axis,
        )

    species_names = sorted({(m.get("species") or "unknown") for m in metadata})
    palette = build_species_palette(species_names)
    nodes = build_nodes(metadata, coords_raw, coords_tsne_norm, coords_cloud)
    species_summary = build_species_summary(nodes, palette)

    write_json(out_dir / "nodes.json", nodes)
    write_json(out_dir / "edges.json", edges)
    write_json(out_dir / "species.json", species_summary)

    manifest = {
        "format": "nest_hypercube_export_v2",
        "source_embeddings": str(embeddings_path).replace("\\", "/"),
        "generated_files": {
            "nodes": "nodes.json",
            "edges": "edges.json",
            "species": "species.json",
        },
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "species": len(species_summary),
        },
        "tsne": {
            "dimensions": 3,
            "perplexity": perplexity,
            "iterations": n_iter,
            "seed": seed,
            "raw_bounds": {
                "min": [float(x) for x in mins],
                "max": [float(x) for x in maxs],
            },
            "normalized_space": "[-1, 1] per axis",
        },
        "display_coordinates": {
            "default_field": "position_tsne",
            "available_fields": [
                "position",
                "position_cloud",
                "position_tsne",
                "position_raw",
            ],
            "position": "cloud-friendly transformed coordinates with isotropic radius normalization (optional stylized display)",
            "position_tsne": "direct min-max normalized t-SNE coordinates in [-1, 1] (default free-form display)",
            "cloud_transform": cloud_transform,
            "species_display_cloudify": species_display_cloudify_stats,
            "species_display_stretch": species_display_stretch_stats,
        },
        "graph": {
            "edge_source": "embedding cosine kNN",
            "k_neighbors": k_neighbors,
            "similarity_metric": "cosine",
            "species_ratio_bridges": species_ratio_bridge_stats,
            "manual_bridges": manual_bridge_stats,
            "spatial_antipode_bridges": antipode_bridge_stats,
            "bridge_edges": bridge_stats,
        },
        "notes": [
            "manifest.display_coordinates.default_field is position_tsne for free-form web display",
            "nodes.position is cloud-friendly transformed display coordinates (optional stylized alternative)",
            "nodes.position_cloud is the same cloud-friendly transformed coordinate set",
            "nodes.position_tsne preserves direct normalized t-SNE coordinates",
            "nodes.position_raw preserves original t-SNE coordinates",
            "edges.edge_kind may be 'knn', 'species_ratio_bridge', 'manual_bridge', "
            "'spatial_antipode_bridge', or 'component_bridge'",
            "nodes.path may need remapping/copying in a separate website repo",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"Exported hypercube package to: {out_dir}")
    print(f"  nodes.json:   {len(nodes)}")
    print(f"  edges.json:   {len(edges)}")
    print(f"  species.json: {len(species_summary)} species")
    print("  manifest.json")


def main():
    parser = argparse.ArgumentParser(
        description="Export a web-friendly hypercube/graph package from NEST (Nearest-Extant Similarity Tool) embeddings (3D t-SNE + kNN edges)"
    )
    parser.add_argument("--embeddings", "-e", default="embeddings/grain_embeddings.npy", help="Path to embeddings .npy")
    parser.add_argument("--out", "-o", default="exports/hypercube", help="Output directory for JSON package")
    parser.add_argument("--perplexity", "-p", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--iterations", "-n", type=int, default=1000, help="t-SNE iterations")
    parser.add_argument("--k", type=int, default=6, help="k nearest neighbors for graph edges")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE")
    parser.add_argument(
        "--cloud-whiten",
        type=float,
        default=0.6,
        help="Cloud display transform partial whitening strength (0 disables, 1 is stronger)",
    )
    parser.add_argument(
        "--cloud-tanh-gain",
        type=float,
        default=1.2,
        help="Cloud display transform outlier soft-clipping gain",
    )
    parser.add_argument(
        "--cloud-z-boost",
        type=float,
        default=1.4,
        help="Extra boost applied to the smallest PCA axis for cloud display coords",
    )
    parser.add_argument(
        "--force-bridge",
        action="append",
        default=[],
        help=(
            "Repeatable explicit bridge edge pair for presentation, e.g. --force-bridge 803:66. "
            "Adds an edge if it does not already exist."
        ),
    )
    parser.add_argument(
        "--species-bridge-ratio",
        type=float,
        default=0.0,
        help=(
            "Presentation bridge budget per source species as a multiple of total species count. "
            "Bridges are distributed across receiving species in proportion to receiving species size. "
            "0 disables."
        ),
    )
    parser.add_argument(
        "--species-bridge-inward-pool",
        type=int,
        default=12,
        help=(
            "Candidate nodes per species for species ratio bridges; picks nodes closest to the "
            "global center to create a stronger visual core."
        ),
    )
    parser.add_argument(
        "--species-bridge-antipode-bias",
        type=float,
        default=0.0,
        help=(
            "Bias species bridge quotas toward spatially opposite species. "
            "0 disables; higher values make opposite-side islands receive more bridges."
        ),
    )
    parser.add_argument(
        "--species-bridge-source-isolation-boost",
        type=float,
        default=0.0,
        help=(
            "Increase outbound bridge budget for species farther from the global center. "
            "Sender size is still ignored."
        ),
    )
    parser.add_argument(
        "--species-bridge-source-edge-bias",
        type=float,
        default=0.0,
        help=(
            "Bias outgoing species bridge edges to originate from nodes nearer the local "
            "boundary of the source species island (higher = more edge-origin bridges)."
        ),
    )
    parser.add_argument(
        "--spatial-antipode-bridge-per-pair",
        type=int,
        default=0,
        help=(
            "Pair disconnected islands with spatially opposite islands and add this many "
            "center-biased cosine bridge edges per pair (0 disables)."
        ),
    )
    parser.add_argument(
        "--spatial-antipode-inward-pool",
        type=int,
        default=12,
        help=(
            "Candidate nodes per island for spatial antipode bridges; nodes closest to the "
            "global center are considered first."
        ),
    )
    parser.add_argument(
        "--component-bridge-per-link",
        type=int,
        default=12,
        help=(
            "Add cross-component bridge edges after kNN using a component MST and "
            "boundary-facing, cosine-scored node links. Value is number of node-pair "
            "edges added per MST component link (0 disables)."
        ),
    )
    parser.add_argument(
        "--display-cloudify-species",
        type=str,
        default="",
        help=(
            "Optional display-only cloudify target species (exact species key, "
            "e.g. quararibea_pterocalyx). Makes a ribbon-like cluster more isotropic."
        ),
    )
    parser.add_argument(
        "--display-cloudify-strength",
        type=float,
        default=0.0,
        help="Display-only cloudify strength in [0,1] for the selected species (0 disables).",
    )
    parser.add_argument(
        "--display-stretch-species",
        type=str,
        default="",
        help=(
            "Optional display-only anisotropic stretch target species (exact species key, "
            "e.g. quararibea_pterocalyx). Does not alter embeddings."
        ),
    )
    parser.add_argument(
        "--display-stretch-factor",
        type=float,
        default=1.0,
        help="Display-only stretch multiplier for the selected species local PCA axis (1.0 disables).",
    )
    parser.add_argument(
        "--display-stretch-axis",
        type=int,
        default=1,
        help="Local PCA axis index to stretch for the selected species (0=major, 1=middle, 2=minor).",
    )
    args = parser.parse_args()

    export_package(
        embeddings_path=args.embeddings,
        out_dir=args.out,
        perplexity=args.perplexity,
        n_iter=args.iterations,
        k_neighbors=args.k,
        seed=args.seed,
        cloud_whiten=args.cloud_whiten,
        cloud_tanh_gain=args.cloud_tanh_gain,
        cloud_z_boost=args.cloud_z_boost,
        force_bridges=args.force_bridge,
        species_bridge_ratio=args.species_bridge_ratio,
        species_bridge_inward_pool=args.species_bridge_inward_pool,
        species_bridge_antipode_bias=args.species_bridge_antipode_bias,
        species_bridge_source_isolation_boost=args.species_bridge_source_isolation_boost,
        species_bridge_source_edge_bias=args.species_bridge_source_edge_bias,
        spatial_antipode_bridge_per_pair=args.spatial_antipode_bridge_per_pair,
        spatial_antipode_inward_pool=args.spatial_antipode_inward_pool,
        component_bridge_per_link=args.component_bridge_per_link,
        display_cloudify_species=args.display_cloudify_species,
        display_cloudify_strength=args.display_cloudify_strength,
        display_stretch_species=args.display_stretch_species,
        display_stretch_factor=args.display_stretch_factor,
        display_stretch_axis=args.display_stretch_axis,
    )


if __name__ == "__main__":
    main()
