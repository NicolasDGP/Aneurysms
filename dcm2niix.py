import SimpleITK as sitk
import os

dicom_dir = "/Users/nicolas/Desktop/tochange"
out_path = "/Users/nicolas/Desktop/changed.nii.gz"

# Find available SeriesInstanceUIDs in the folder
series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
if not series_ids:
    raise RuntimeError(f"No DICOM series found in: {dicom_dir}")


# If there's only one series in the folder, use it
series_id = series_ids[0]

# Get the ordered file list for that series (sorting is handled here)
file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_id)

reader = sitk.ImageSeriesReader()
reader.SetFileNames(file_names)

img3d = reader.Execute()  # 3D volume with spacing/origin/direction set

# Write to NIfTI
sitk.WriteImage(img3d, out_path, useCompression=True)

print("Wrote:", out_path)
print("Size (x,y,z):", img3d.GetSize())
print("Spacing (x,y,z) mm:", img3d.GetSpacing())
print("Origin:", img3d.GetOrigin())
print("Direction:", img3d.GetDirection())
