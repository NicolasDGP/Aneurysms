# pip install pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-rle numpy napari pandas nibabel

import os, ast, json
import numpy as np
import pandas as pd

# --- DICOM dependencies (for your original workflow) ---
import pydicom

# --- NIfTI + resampling for segmentation ---
import nibabel as nib
from nibabel.processing import resample_from_to

# Optional: set this True to print more info
VERBOSE = True

# DICOM loading & utilities 

def read_series(folder, target_series_uid=None):
    """
    Return (datasets_sorted, spacing_zyx, sop_to_index, series_uid).

    Unchanged from your original logic: scans a DICOM tree, picks a series,
    sorts along the slice normal, returns spacing and SOP index mapping.
    """
    stubs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                p = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                except Exception:
                    continue
                if hasattr(ds, "SeriesInstanceUID") and hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
                    ds._path = p
                    stubs.append(ds)
    if not stubs:
        raise RuntimeError("No DICOM images with geometry found in folder.")

    # pick series
    if target_series_uid is None:
        by_series = {}
        for ds in stubs:
            by_series.setdefault(ds.SeriesInstanceUID, []).append(ds)
        series_uid = max(by_series, key=lambda k: len(by_series[k]))
        stubs = by_series[series_uid]
    else:
        stubs = [ds for ds in stubs if ds.SeriesInstanceUID == target_series_uid]
        if not stubs:
            raise ValueError(f"Series {target_series_uid} not found.")
        series_uid = target_series_uid

    # load full datasets (pixel-ready)
    dsets = [pydicom.dcmread(ds._path, force=True) for ds in stubs]

    # sort along slice normal
    iop = np.array(dsets[0].ImageOrientationPatient, dtype=float)
    row, col = iop[:3], iop[3:]
    normal = np.cross(row, col)
    pos = [float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal)) for ds in dsets]
    order = np.argsort(pos)
    dsets = [dsets[i] for i in order]

    # spacing (z from positions, y/x from PixelSpacing)
    dots = np.array([float(np.dot(np.array(ds.ImagePositionPatient, dtype=float), normal)) for ds in dsets])
    z = float(np.mean(np.abs(np.diff(dots)))) if len(dsets) > 1 else float(dsets[0].get("SliceThickness", 1.0))
    ps = dsets[0].PixelSpacing  # [row, col]
    spacing_zyx = (z, float(ps[0]), float(ps[1]))

    sop_to_index = {ds.SOPInstanceUID: i for i, ds in enumerate(dsets)}
    return dsets, spacing_zyx, sop_to_index, series_uid

def to_hu(ds, arr):
    slope = float(ds.get("RescaleSlope", 1) or 1)
    inter = float(ds.get("RescaleIntercept", 0) or 0)
    return arr.astype(np.float32) * slope + inter

def parse_coord_cell(cell, rows, cols):
    """
    Accepts:
      - dict-like {'x': 123.4, 'y': 234.5}
      - JSON string / repr of that dict
      - normalized floats 0..1 (we scale to pixel indices)
    Returns (row, col) in pixel indices (0-based, float).
    """
    if isinstance(cell, str):
        try:
            val = json.loads(cell)
        except Exception:
            val = ast.literal_eval(cell)
    else:
        val = cell

    x = float(val["x"])
    y = float(val["y"])

    # normalized case
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x = x * (cols - 1)
        y = y * (rows - 1)

    # 1-based indexing case: if values are exact-ish integers
    if 1 <= x <= cols and abs(x - round(x)) < 1e-6 and 1 <= y <= rows and abs(y - round(y)) < 1e-6:
        x -= 1.0
        y -= 1.0

    # clamp to image
    x = min(max(x, 0.0), cols - 1.0)
    y = min(max(y, 0.0), rows - 1.0)
    return float(y), float(x)  # return (row, col)

def mm_radius_to_points_size(radius_mm, spacing_zyx):
    """
    Convert a sphere radius (mm) into a single napari Points 'size' value
    (diameter in voxels) using geometric-mean voxel size so the marker is
    reasonably sized in 3D.
    """
    if radius_mm is None:
        return 6.0  # small visible dot in voxels
    z, y, x = map(float, spacing_zyx)
    v_mean = (z * y * x) ** (1.0 / 3.0)
    return float(max(3.0, 2.0 * radius_mm / v_mean))

# =========================================================
# ===================== NIfTI helpers =====================
# =========================================================

def _to_zyx_and_spacing(nib_img):
    """
    Reorient to RAS, then convert to Z,Y,X array for napari and return spacing in ZYX.
    """
    img_canon = nib.as_closest_canonical(nib_img)
    data_xyz = img_canon.get_fdata(dtype=np.float32)   # shape (X, Y, Z)
    vs_xyz = nib.affines.voxel_sizes(img_canon.affine) # (vx_x, vx_y, vx_z) in mm
    data_zyx = np.transpose(data_xyz, (2, 1, 0))       # -> (Z, Y, X) for napari
    spacing_zyx = (float(vs_xyz[2]), float(vs_xyz[1]), float(vs_xyz[0]))
    return data_zyx, spacing_zyx, img_canon

def load_nifti_with_optional_seg(image_path, seg_path=None):
    """
    Load a NIfTI image (and optional segmentation). If segmentation is provided,
    resample it onto the image grid (nearest-neighbor), then return both arrays
    in Z,Y,X with matching spacing_zyx, plus the canonical image object (affine).
    """
    img = nib.load(image_path)
    img_zyx, spacing_zyx, img_canon = _to_zyx_and_spacing(img)

    seg_zyx = None
    if seg_path is not None and os.path.exists(seg_path):
        seg = nib.load(seg_path)
        seg_canon = nib.as_closest_canonical(seg)

        # If not the same grid, resample seg -> image (nearest for labels)
        need_resample = (
            img_canon.shape != seg_canon.shape or
            not np.allclose(img_canon.affine, seg_canon.affine, atol=1e-3)
        )
        if need_resample:
            if VERBOSE:
                print("[INFO] Resampling segmentation to the image grid (nearest-neighbor).")
            seg_on_img = resample_from_to(seg_canon, img_canon, order=0)
        else:
            seg_on_img = seg_canon

        seg_xyz = seg_on_img.get_fdata(dtype=np.float32)
        seg_zyx = np.transpose(seg_xyz, (2, 1, 0)).astype(np.int16)

    # NEW: return the canonical image object (for affine/world<->voxel)
    return img_zyx, seg_zyx, spacing_zyx, img_canon

def segmentation_stats(seg_zyx, spacing_zyx):
    """
    Compute simple per-label volumes (mm^3 and mL).
    """
    if seg_zyx is None:
        return pd.DataFrame(columns=["label", "voxels", "volume_mm3", "volume_ml"])

    labels, counts = np.unique(seg_zyx, return_counts=True)
    z, y, x = spacing_zyx
    voxel_mm3 = float(z * y * x)

    rows = []
    for lbl, cnt in zip(labels, counts):
        if int(lbl) == 0:
            continue
        vol_mm3 = float(cnt) * voxel_mm3
        rows.append({
            "label": int(lbl),
            "voxels": int(cnt),
            "volume_mm3": vol_mm3,
            "volume_ml": vol_mm3 / 1000.0
        })
    return pd.DataFrame(rows).sort_values("label")

# ----------------------- NEW HELPERS -----------------------

def _labels_nonzero(seg_zyx):
    """Return sorted list of non-zero labels present in the segmentation."""
    if seg_zyx is None:
        return []
    labels = np.unique(seg_zyx)
    labels = [int(l) for l in labels.tolist() if int(l) != 0]
    return sorted(labels)

def _centroid_of_label_zyx(seg_zyx, label_id):
    """Compute the centroid (mean of voxel indices) of the specified label."""
    mask = (seg_zyx == int(label_id))
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)  # list of [z, y, x]
    c = coords.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])  # (z, y, x)

def _clamp_point_zyx(pt, shape_zyx):
    """Clamp a (z,y,x) float point to be within the array bounds."""
    z, y, x = pt
    Z, Y, X = shape_zyx
    z = min(max(z, 0.0), Z - 1.0)
    y = min(max(y, 0.0), Y - 1.0)
    x = min(max(x, 0.0), X - 1.0)
    return float(z), float(y), float(x)

def _parse_point_mm_to_zyx(point_mm, img_canon):
    """
    Convert a world RAS (mm) point into voxel indices on the canonical image grid,
    then map to (z, y, x) order used in the viewer.
    Accepts dict-like {'x':..., 'y':..., 'z':...} or iterable [x, y, z].
    """
    if isinstance(point_mm, dict):
        Xmm = float(point_mm["x"]); Ymm = float(point_mm["y"]); Zmm = float(point_mm["z"])
    else:
        Xmm, Ymm, Zmm = [float(v) for v in point_mm]

    affine = img_canon.affine  # maps voxel (x,y,z,1) -> world (X,Y,Z,1)
    inv_aff = np.linalg.inv(affine)
    vxyz = inv_aff @ np.array([Xmm, Ymm, Zmm, 1.0], dtype=np.float64)
    vx, vy, vz, _ = vxyz
    # Reorder from (x,y,z) to (z,y,x)
    return float(vz), float(vy), float(vx)

# =========================================================
# ======================== Viewer =========================
# =========================================================

def show_nifti_with_seg(
    image_nii,
    seg_nii=None,
    intensity_window=None,   # e.g., (-200, 1500) for CTA; if None, auto via percentiles
    opacity=0.35,
    # -------- NEW aneurysm point options --------
    aneurysm_point_zyx=None,   # tuple/list (z, y, x) in voxel indices of the displayed grid
    aneurysm_point_mm=None,    # dict or (x,y,z) in world RAS mm (we convert via affine)
    aneurysm_label=None,       # int label id whose centroid will be used
    aneurysm_radius_mm=3.0,    # size control for the Points layer
    aneurysm_color="yellow"
):
    import napari

    img_zyx, seg_zyx, spacing_zyx, img_canon = load_nifti_with_optional_seg(image_nii, seg_nii)

    # auto window if not provided
    clim = intensity_window
    if clim is None:
        p1, p99 = np.percentile(img_zyx[np.isfinite(img_zyx)], [1, 99])
        clim = (float(p1), float(p99))

    viewer = napari.Viewer()
    viewer.add_image(
        img_zyx,
        name="Image (NIfTI)",
        scale=spacing_zyx,
        colormap="gray",
        contrast_limits=clim,
        rendering="attenuated_mip",
    )

    # Keep track for optional aneurysm point placement
    aneurysm_pt_zyx = None

    if seg_zyx is not None and np.any(seg_zyx > 0):
        viewer.add_labels(
            seg_zyx,
            name="Segmentation",
            scale=spacing_zyx,
            opacity=float(opacity),
        )

        # Print basic stats and also show in the console
        df = segmentation_stats(seg_zyx, spacing_zyx)
        if not df.empty:
            print("\nPer-label volumes (derived from segmentation):")
            print(df.to_string(index=False))
        else:
            print("[INFO] Segmentation present but contains only background (label 0).")

        # ---------------- NEW: compute/display aneurysm point ----------------
        # Priority: explicit voxel -> explicit mm -> centroid of specified label -> auto (single label)
        if aneurysm_point_zyx is not None:
            aneurysm_pt_zyx = tuple([float(v) for v in aneurysm_point_zyx])

        elif aneurysm_point_mm is not None:
            try:
                aneurysm_pt_zyx = _parse_point_mm_to_zyx(aneurysm_point_mm, img_canon)
            except Exception as e:
                print(f"[WARN] Failed to parse aneurysm_point_mm -> voxel: {e}")

        elif aneurysm_label is not None:
            c = _centroid_of_label_zyx(seg_zyx, int(aneurysm_label))
            if c is None:
                print(f"[WARN] Label {aneurysm_label} not found in segmentation; cannot place aneurysm point.")
            else:
                aneurysm_pt_zyx = c

        else:
            # If exactly one non-zero label exists, auto-use its centroid.
            nonzero = _labels_nonzero(seg_zyx)
            if len(nonzero) == 1:
                aneurysm_pt_zyx = _centroid_of_label_zyx(seg_zyx, nonzero[0])
                if VERBOSE:
                    print(f"[INFO] Using centroid of the only non-zero label ({nonzero[0]}) as aneurysm point.")
            else:
                if VERBOSE:
                    print("[INFO] Multiple labels present and no aneurysm specified; no point will be added.")

        if aneurysm_pt_zyx is not None:
            aneurysm_pt_zyx = _clamp_point_zyx(aneurysm_pt_zyx, seg_zyx.shape)
            size_scalar = mm_radius_to_points_size(aneurysm_radius_mm, spacing_zyx)
            viewer.add_points(
                np.array([aneurysm_pt_zyx], dtype=float),
                name="Aneurysm (point)",
                size=size_scalar,
                face_color=aneurysm_color,
                edge_color="black",
                opacity=0.98,
                scale=spacing_zyx,
                properties={"label": ["aneurysm"]},
                text={"string": "{label}", "size": 14, "color": aneurysm_color, "anchor": "upper_left"},
            )
            # Center the viewer on the aneurysm slice
            viewer.dims.set_current_step(0, int(round(aneurysm_pt_zyx[0])))

    else:
        print("[INFO] No segmentation loaded or all zeros.")

    napari.run()

def show_dicom_with_points_and_optional_seg(
    dicom_folder,
    df_points=None,         # DataFrame with columns: SeriesInstanceUID, SOPInstanceUID, coordinates, location
    series_uid=None,
    points_radius_mm=3.0
):
    """
    Original DICOM viewer for HU + points. (This path does not resample a NIfTI
    segmentation onto the DICOM grid; if you want segmentation, prefer the NIfTI
    path 'show_nifti_with_seg'.)
    """
    import napari

    dsets, spacing_zyx, sop_to_idx, series_uid = read_series(dicom_folder, series_uid)

    slices = [to_hu(ds, ds.pixel_array) for ds in dsets]
    volume_hu = np.stack(slices, axis=0).astype(np.float32)  # (Z,Y,X)
    rows, cols = int(dsets[0].Rows), int(dsets[0].Columns)

    viewer = napari.Viewer()
    viewer.add_image(
        volume_hu,
        name="CTA (HU)",
        scale=spacing_zyx,
        colormap="gray",
        contrast_limits=(-200, 1500),
        rendering="attenuated_mip",
    )

    # Optional points overlay (as in your original code)
    points = []
    labels = []
    if df_points is not None and len(df_points):
        for _, r in df_points.iterrows():
            sop = str(r["SOPInstanceUID"])
            if sop not in sop_to_idx:
                print(f"Warning: SOP {sop} not found in this series; skipping.")
                continue
            z = sop_to_idx[sop]
            y, x = parse_coord_cell(r["coordinates"], rows, cols)
            points.append([float(z), float(y), float(x)])
            labels.append(r.get("location", ""))

    if points:
        points = np.array(points, dtype=float)
        size_scalar = mm_radius_to_points_size(points_radius_mm, spacing_zyx)
        viewer.add_points(
            points,
            name="findings",
            size=size_scalar,
            face_color="yellow",
            edge_color="black",
            opacity=0.95,
            scale=spacing_zyx,
            properties={"label": labels},
            text={"string": "{label}", "size": 12, "color": "yellow", "anchor": "upper_left"},
        )
        viewer.dims.set_current_step(0, int(points[0, 0]))  # jump to first

    napari.run()

# =========================================================
# ========================== MAIN =========================
# =========================================================

if __name__ == "__main__":
    # --- Choose ONE of the two usage paths below ---

    # -------- 1) NIfTI + Segmentation (recommended for your uploads) --------
    IMAGE_NIFTI = "segmentations/1.2.826.0.1.3680043.8.498.11999987145696510072091906561590137848/1.2.826.0.1.3680043.8.498.11999987145696510072091906561590137848.nii"
    SEGMENT_NIFTI = "segmentations/1.2.826.0.1.3680043.8.498.11999987145696510072091906561590137848/1.2.826.0.1.3680043.8.498.11999987145696510072091906561590137848_cowseg.nii"

    if os.path.exists(IMAGE_NIFTI):
        if VERBOSE:
            print(f"[INFO] Loading NIfTI image: {IMAGE_NIFTI}")
            if os.path.exists(SEGMENT_NIFTI):
                print(f"[INFO] Loading segmentation: {SEGMENT_NIFTI}")
            else:
                print("[INFO] No segmentation file found; proceeding without it.")

        # EXAMPLE OPTIONS to make the aneurysm point appear:
        #   (pick ONE of these examples and comment the others)
        #
        # 1) Provide a voxel-space point directly (z, y, x):
        ANEURYSM_POINT_ZYX = None  # e.g., (120.4, 256.3, 310.1)
        #
        # 2) Provide a world-space point in mm (RAS):
        ANEURYSM_POINT_MM = None   # e.g., {"x": 12.3, "y": -18.5, "z": 42.0}
        #
        # 3) Provide a label id whose centroid will be used:
        ANEURYSM_LABEL_ID = 0   # e.g., 1

        # If none of the above are provided and the segmentation only has one
        # non-zero label, its centroid will be used automatically.

        show_nifti_with_seg(
            image_nii=IMAGE_NIFTI,
            seg_nii=SEGMENT_NIFTI,
            intensity_window=None,  # e.g., (-200, 1500) for CTA
            opacity=0.35,
            aneurysm_point_zyx=ANEURYSM_POINT_ZYX,
            aneurysm_point_mm=ANEURYSM_POINT_MM,
            aneurysm_label=ANEURYSM_LABEL_ID,
            aneurysm_radius_mm=3.0,
            aneurysm_color="yellow",
        )

    else:
        # -------- 2) DICOM + points (original workflow preserved) --------
        # NOTE: This path does NOT overlay a NIfTI segmentation on DICOM. Use the NIfTI path above for seg.
        dicom_folder = "series/1.2.826.0.1.3680043.8.498.11999987145696510072091906561590137848"

        # Example 'df_points' matching your original code (replace with your CSV if needed)
        df_points = pd.DataFrame([{
            "SeriesInstanceUID": "1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317",
            "SOPInstanceUID": "1.2.826.0.1.3680043.8.498.87636090859887160412247303157695174056",
            "coordinates": {"x": 345.3863056586271, "y": 273.92949907235624},
            "location": "Aneurysm center",
        }])

        if VERBOSE:
            print(f"[INFO] Loading DICOM folder: {dicom_folder}")
        show_dicom_with_points_and_optional_seg(
            dicom_folder=dicom_folder,
            df_points=df_points,
            series_uid=df_points.iloc[0]["SeriesInstanceUID"],
            points_radius_mm=3.0
        )
