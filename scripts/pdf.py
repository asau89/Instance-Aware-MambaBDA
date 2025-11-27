import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
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

    Expected filename: {base}_building_damage.png
    Search order: ROOT/target -> dataset/*/target
    """
    fname = f"{name_base}_building_damage.png"
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
    variant in {base, thesis}
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
    # decide base/thesis by dir
    variant = "base"
    if any(k in variant_dir for k in ["thesis", "thesisdense", "thesisnormal"]):
        variant = "thesis"
    # extract numeric id
    run_id = "".join([c for c in run_dir if c.isdigit()]) or run_dir
    return (set_group, variant, run_id)


def pair_base_thesis_runs(run_roots: List[Path]) -> List[Tuple[Path, Path, str]]:
    """Return list of (base_run_root, thesis_run_root, set_group) paired by set and run_id.
    If exact run_id match not found, skip pairing.
    """
    buckets: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for rr in run_roots:
        set_group, variant, run_id = parse_run_identity(rr)
        key = (set_group, variant)
        buckets.setdefault(key, {})[run_id] = rr
    pairs: List[Tuple[Path, Path, str]] = []
    for set_group in {k[0] for k in buckets.keys()}:
        base_map = buckets.get((set_group, "base"), {})
        ths_map = buckets.get((set_group, "thesis"), {})
        for rid, base_rr in base_map.items():
            th_rr = ths_map.get(rid)
            if th_rr is not None:
                pairs.append((base_rr, th_rr, set_group))
    return pairs


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


def make_triplet_page(fig, axes, gt_vis: Image.Image, base_img: Image.Image, thesis_img: Image.Image, title: str):
    axes[0].imshow(gt_vis, interpolation='nearest')
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    axes[1].imshow(base_img, interpolation='nearest')
    axes[1].set_title("Base")
    axes[1].axis('off')
    axes[2].imshow(thesis_img, interpolation='nearest')
    axes[2].set_title("Thesis")
    axes[2].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def compile_pair_to_pdfs(base_run: Path, thesis_run: Path, set_group: str, out_root: Path,
                         extra_target_dirs: List[Path], max_pages: int = None):
    """Create two PDFs with 3-column pages: GT, Base, Thesis.
    Separate PDFs for localization and classification maps.
    """
    loc_base = base_run / "building_localization_map"
    loc_thes = thesis_run / "building_localization_map"
    cls_base = base_run / "damage_classification_map"
    cls_thes = thesis_run / "damage_classification_map"

    # Prepare output dir structure
    set_dir = out_root / set_group
    set_dir.mkdir(parents=True, exist_ok=True)

    # Helper to compute common stems
    def common_stems(a: Path, b: Path) -> List[str]:
        a_stems = {p.stem for p in list_pngs(a)}
        b_stems = {p.stem for p in list_pngs(b)}
        inter = sorted(a_stems & b_stems)
        return inter if max_pages is None else inter[:max_pages]

    # Helper to get run folder names (e.g., resultsbasedense1)
    def run_dir_name(rr: Path) -> str:
        parts = list(rr.parts)
        try:
            i = parts.index("MODEL OUTPUTS")
            tail = parts[i+1:]
            return tail[2] if len(tail) > 2 else rr.name
        except ValueError:
            return rr.name

    base_run_name = run_dir_name(base_run)
    thesis_run_name = run_dir_name(thesis_run)

    # PDF for localization
    loc_stems = common_stems(loc_base, loc_thes)
    if loc_stems:
        pdf_loc_path = set_dir / f"{set_group}_{base_run_name}_vs_{thesis_run_name}_loc.pdf"
        with PdfPages(pdf_loc_path) as pdf:
            for stem in tqdm(loc_stems, desc=f"PDF (loc) {set_group} base/thesis"):
                gt_path = resolve_gt_mask(stem, extra_target_dirs)
                if gt_path is None:
                    continue
                try:
                    gt_vis = colorize_mask(gt_path)
                    bimg = Image.open(loc_base / f"{stem}.png").convert('RGB')
                    timg = Image.open(loc_thes / f"{stem}.png").convert('RGB')
                    # Convert to arrays to reduce downstream resampling costs
                    gt_arr = np.asarray(gt_vis)
                    b_arr = np.asarray(bimg)
                    t_arr = np.asarray(timg)
                except Exception:
                    continue
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
                make_triplet_page(fig, axes, gt_arr, b_arr, t_arr, title=stem)
                pdf.savefig(fig)
                plt.close(fig)

    # PDF for classification
    cls_stems = common_stems(cls_base, cls_thes)
    if cls_stems:
        pdf_cls_path = set_dir / f"{set_group}_{base_run_name}_vs_{thesis_run_name}_cls.pdf"
        with PdfPages(pdf_cls_path) as pdf:
            for stem in tqdm(cls_stems, desc=f"PDF (cls) {set_group} base/thesis"):
                gt_path = resolve_gt_mask(stem, extra_target_dirs)
                if gt_path is None:
                    continue
                try:
                    gt_vis = colorize_mask(gt_path)
                    bimg = Image.open(cls_base / f"{stem}.png").convert('RGB')
                    timg = Image.open(cls_thes / f"{stem}.png").convert('RGB')
                    gt_arr = np.asarray(gt_vis)
                    b_arr = np.asarray(bimg)
                    t_arr = np.asarray(timg)
                except Exception:
                    continue
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
                make_triplet_page(fig, axes, gt_arr, b_arr, t_arr, title=stem)
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
    pairs = pair_base_thesis_runs(run_roots)
    if not pairs:
        print("No base/thesis run pairs found. Ensure directory names follow base/thesis patterns.")
        return 0
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for base_run, thesis_run, set_group in pairs:
        compile_pair_to_pdfs(base_run, thesis_run, set_group, EXPORT_ROOT, extra_targets, max_pages=max_pages)
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
