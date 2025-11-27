import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
MODEL_OUTPUTS = ROOT / "MODEL OUTPUTS"
TARGET_DIR = ROOT / "target"
EXPORT_ROOT = ROOT / "compiled_pdfs"


def find_split_target_dirs(root: Path) -> List[Path]:
    """Fallback: discover split-specific target directories under dataset/*/target."""
    dirs = []
    ds = root / "dataset"
    if ds.exists():
        for split in ds.iterdir():
            if split.is_dir():
                td = split / "target"
                if td.exists():
                    dirs.append(td)
    return dirs


def class_color_map() -> Dict[int, Tuple[int, int, int]]:
    """Return solid RGB colors for classes using the requested palette.
    Mapping (by label index):
      0 background -> #440154
      1 intact     -> #3b528b
      2 minor      -> #21918c
      3 major      -> #fce625
      4 destroyed  -> fallback distinct color (not specified by user)
    """
    return {
        0: (68, 1, 84),      # background #440154
        1: (59, 82, 139),    # intact #3b528b
        2: (33, 145, 140),   # minor #21918c
        3: (252, 230, 37),   # major #fce625
        4: (230, 97, 1),     # destroyed (fallback distinct)
    }


def resolve_gt_mask(name_base: str, extra_target_dirs: List[Path]) -> Path:
    """Resolve ground-truth mask path for a given base name (without suffix/extension).

    Expected filename: {base}_building_damage.tif (or .png)
    Search order: ROOT/target -> dataset/*/target
    """
    # Try .tif first, then .png
    for ext in ['.tif', '.png']:
        fname = f"{name_base}_building_damage{ext}"
        p = TARGET_DIR / fname
        if p.exists():
            return p
        for d in extra_target_dirs:
            cand = d / fname
            if cand.exists():
                return cand
    return None


def colorize_mask(mask_path: Path) -> Image.Image:
    """Load a label mask and convert to an RGB image with a fixed solid-color palette."""
    try:
        m = Image.open(mask_path).convert('L')
    except Exception:
        return None
    arr = np.array(m)
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    cm = class_color_map()
    for cls_idx, color in cm.items():
        rgb[arr == cls_idx] = color
    return Image.fromarray(rgb, mode='RGB')


def collect_run_roots(model_outputs: Path) -> List[Path]:
    """Collect leaves like .../BRIGHT/<ModelName>/ that contain the task folders."""
    run_roots = []
    if not model_outputs.exists():
        return run_roots
    # Traverse up to reasonable depth
    for root, dirs, files in os.walk(model_outputs):
        root_p = Path(root)
        # Heuristic: a run root has subdirs building_localization_map and/or damage_classification_map
        if "building_localization_map" in dirs or "damage_classification_map" in dirs:
            run_roots.append(root_p)
    return sorted(set(run_roots))


def parse_run_identity(run_root: Path) -> Tuple[str, str, str]:
    """Parse (set_group, variant, run_id) from path.
    set_group in {denseset, normalset}
    variant in {base, thesis, thesis2, thesis3, etc.}
    run_id is the numeric suffix if present, else last segment
    """
    parts = list(run_root.parts)
    # try to find 'MODEL OUTPUTS'
    try:
        i = parts.index("MODEL OUTPUTS")
    except ValueError:
        return ("unknown", "base", run_root.name)
    tail = parts[i+1:]  # e.g., ['denseset','basedense','resultsbasedense1','BRIGHT','MambaBDA_Tiny']
    set_group = tail[0] if tail else "unknown"
    variant_dir = tail[1] if len(tail) > 1 else ""
    run_dir = tail[2] if len(tail) > 2 else run_root.name
    # decide base/thesis/thesis2/etc by dir
    variant = "base"
    if "thesis2dense" in variant_dir or "thesis2normal" in variant_dir:
        variant = "thesis2"
    elif "thesis3dense" in variant_dir or "thesis3normal" in variant_dir:
        variant = "thesis3"
    elif any(k in variant_dir for k in ["thesis", "thesisdense", "thesisnormal"]):
        variant = "thesis"
    # extract numeric id
    run_id = "".join([c for c in run_dir if c.isdigit()]) or run_dir
    return (set_group, variant, run_id)


def group_runs_by_set(run_roots: List[Path]) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Group runs by set_group, run_id, and variant.
    Returns: {set_group: {run_id: {variant: path}}}
    """
    grouped: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for rr in run_roots:
        set_group, variant, run_id = parse_run_identity(rr)
        if set_group not in grouped:
            grouped[set_group] = {}
        if run_id not in grouped[set_group]:
            grouped[set_group][run_id] = {}
        grouped[set_group][run_id][variant] = rr
    return grouped


def list_pngs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".png"]) 


def base_from_png(p: Path) -> str:
    return p.stem  # e.g., "ukraine-conflict_00000442"


def human_run_label(run_root: Path) -> str:
    """Build a readable label from path components (model and run)."""
    parts = list(run_root.parts)
    try:
        i = parts.index("MODEL OUTPUTS")
    except ValueError:
        return run_root.name
    tail = parts[i+1:]  # e.g., ['denseset','basedense','resultsbasedense1','BRIGHT','MambaBDA_Tiny']
    return "/".join(tail)


# removed overlay on SAR/post imagery; we render GT as solid-color mask only


def make_comparison_page(fig, axes, gt_vis: Image.Image, images: List[Image.Image], labels: List[str], title: str):
    axes[0].imshow(gt_vis, interpolation='nearest')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    for i, (img, label) in enumerate(zip(images, labels), start=1):
        axes[i].imshow(img, interpolation='nearest')
        axes[i].set_title(label)
        axes[i].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def compile_runs_to_pdfs(run_dict: Dict[str, Path], set_group: str, run_id: str, out_root: Path,
                         extra_target_dirs: List[Path], max_pages: int = None):
    """Create two PDFs with multi-column pages: GT, Base, Thesis, Thesis2, etc.
    Separate PDFs for localization and classification maps.
    run_dict: {variant: path} e.g., {'base': path1, 'thesis': path2, 'thesis2': path3}
    """
    # Get base run (required)
    if 'base' not in run_dict:
        return
    
    # Collect all variants in order
    variant_order = ['base', 'thesis', 'thesis2', 'thesis3', 'thesis4']
    variants = [(v, run_dict[v]) for v in variant_order if v in run_dict]
    
    if len(variants) < 2:  # Need at least base + one thesis variant
        return
    
    # Build maps for each variant
    loc_maps = {v: path / "building_localization_map" for v, path in variants}
    cls_maps = {v: path / "damage_classification_map" for v, path in variants}

    # Prepare output dir structure
    set_dir = out_root / set_group
    set_dir.mkdir(parents=True, exist_ok=True)

    # Helper to compute common stems across all variants
    def common_stems_multi(maps: Dict[str, Path]) -> List[str]:
        stem_sets = [{p.stem for p in list_pngs(path)} for path in maps.values() if path.exists()]
        if not stem_sets:
            return []
        inter = sorted(set.intersection(*stem_sets))
        return inter if max_pages is None else inter[:max_pages]

    # Build descriptive name with all variants
    variant_names = '_'.join([v for v, _ in variants])
    pdf_base_name = f"{set_group}_{run_id}_{variant_names}"

    # PDF for localization
    loc_stems = common_stems_multi(loc_maps)
    if loc_stems:
        pdf_loc_path = set_dir / f"{pdf_base_name}_loc.pdf"
        with PdfPages(pdf_loc_path) as pdf:
            desc = f"PDF (loc) {set_group} {variant_names}"
            for stem in tqdm(loc_stems, desc=desc):
                gt_path = resolve_gt_mask(stem, extra_target_dirs)
                if gt_path is None:
                    continue
                try:
                    gt_vis = colorize_mask(gt_path)
                    gt_arr = np.asarray(gt_vis)
                    
                    # Load all variant images
                    variant_imgs = []
                    variant_labels = []
                    for v, _ in variants:
                        img_path = loc_maps[v] / f"{stem}.png"
                        if img_path.exists():
                            img = Image.open(img_path).convert('RGB')
                            variant_imgs.append(np.asarray(img))
                            variant_labels.append(v.capitalize())
                    
                    if not variant_imgs:
                        continue
                except Exception:
                    continue
                
                n_cols = len(variant_imgs) + 1  # +1 for GT
                fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=120)
                make_comparison_page(fig, axes, gt_arr, variant_imgs, variant_labels, title=stem)
                pdf.savefig(fig)
                plt.close(fig)

    # PDF for classification
    cls_stems = common_stems_multi(cls_maps)
    if cls_stems:
        pdf_cls_path = set_dir / f"{pdf_base_name}_cls.pdf"
        with PdfPages(pdf_cls_path) as pdf:
            desc = f"PDF (cls) {set_group} {variant_names}"
            for stem in tqdm(cls_stems, desc=desc):
                gt_path = resolve_gt_mask(stem, extra_target_dirs)
                if gt_path is None:
                    continue
                try:
                    gt_vis = colorize_mask(gt_path)
                    gt_arr = np.asarray(gt_vis)
                    
                    # Load all variant images
                    variant_imgs = []
                    variant_labels = []
                    for v, _ in variants:
                        img_path = cls_maps[v] / f"{stem}.png"
                        if img_path.exists():
                            img = Image.open(img_path).convert('RGB')
                            variant_imgs.append(np.asarray(img))
                            variant_labels.append(v.capitalize())
                    
                    if not variant_imgs:
                        continue
                except Exception:
                    continue
                
                n_cols = len(variant_imgs) + 1  # +1 for GT
                fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), dpi=120)
                make_comparison_page(fig, axes, gt_arr, variant_imgs, variant_labels, title=stem)
                pdf.savefig(fig)
                plt.close(fig)


def cleanup_previous_outputs():
    # Remove PDFs from previous runs in both old and new output roots
    old_roots = [ROOT / "model_pdfs", EXPORT_ROOT, MODEL_OUTPUTS]
    for r in old_roots:
        if not r.exists():
            continue
        for root, dirs, files in os.walk(r):
            for f in files:
                if f.lower().endswith('.pdf'):
                    try:
                        (Path(root) / f).unlink()
                    except Exception:
                        pass


def main(max_pages: int = None):
    # Clean previous generated PDFs as requested
    cleanup_previous_outputs()
    extra_targets = find_split_target_dirs(ROOT)
    run_roots = collect_run_roots(MODEL_OUTPUTS)
    if not run_roots:
        print(f"No model runs found under: {MODEL_OUTPUTS}")
        return 0
    
    # Group runs by set and run_id
    grouped = group_runs_by_set(run_roots)
    if not grouped:
        print("No runs found.")
        return 0
    
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Process each set_group and run_id combination
    for set_group, run_ids in grouped.items():
        for run_id, variants in run_ids.items():
            if 'base' in variants and len(variants) > 1:
                compile_runs_to_pdfs(variants, set_group, run_id, EXPORT_ROOT, extra_targets, max_pages=max_pages)
    
    print(f"Done. PDFs are in: {EXPORT_ROOT}")
    return 0


if __name__ == "__main__":
    # Optional arg: max_pages to limit output while testing
    mp = None
    if len(sys.argv) > 1:
        try:
            mp = int(sys.argv[1])
        except Exception:
            mp = None
    sys.exit(main(max_pages=mp))
