# Data Directory

The raw DFC2020 data is **not** committed to git (too large, and licensing restrictions apply). To reproduce this project, you must obtain the data yourself and place it in this directory.

## Source

2020 IEEE GRSS Data Fusion Contest dataset, available at:
https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest

The relevant zip files are:

| File | Size | Contents |
| --- | --- | --- |
| `s1_validation.zip` | 948 MB | Sentinel-1 GRD GeoTIFFs (VV + VH), validation split |
| `s2_validation.zip` | 633 MB | Sentinel-2 L2A GeoTIFFs (13 bands), validation split |
| `dfc_validation.zip` | 6 MB | High-quality DFC2020 land cover labels, validation split |
| `s1_0.zip` | 4.96 GB | Sentinel-1 GeoTIFFs, test split |
| `s2_0.zip` | 3.75 GB | Sentinel-2 GeoTIFFs, test split |
| `dfc_0.zip` | 36 MB | DFC labels, test split |

## Expected layout after extraction

After extracting and normalizing folder structure:
```text
data/
└── raw/
    ├── ROIs0000_validation/   # 986 patches per modality
    │   ├── s1_validation/
    │   ├── s2_validation/
    │   └── dfc_validation/
    └── ROIs0000_test/         # 5,128 patches per modality
        ├── s1_0/
        ├── s2_0/
        └── dfc_0/
```

Each patch is **256 × 256 pixels**. Patch IDs (`_p0`, `_p1`, ...) are consistent across S1, S2, and DFC for the same location, so file pairing is by ID.

## Notes on the official distribution

Some modality zips extract with a wrapping subfolder; others extract flat. The expected layout above is the *normalized* version we use here — see `scripts/extract_data.ps1` (if present) or the project setup notes for the move commands used to normalize.

The `__MACOSX` folders that appear in some extractions are macOS metadata artifacts and should be deleted.

## Patch dimensions and dtypes (verified)

| Modality | Shape | Dtype | File size |
| --- | --- | --- | --- |
| Sentinel-1 | (2, 256, 256) | float64 (will cast to float32) | ~1.05 MB |
| Sentinel-2 | (13, 256, 256) | uint16 | ~1.71 MB |
| DFC label | (256, 256) | uint8 | ~66 KB |

## What is **not** used

The original release also includes `lc_*.zip` files (MODIS-derived land cover, lower-quality labels) and a `DFC_Public_Dataset.zip` bundle. We use neither — all training and evaluation here use the high-quality `dfc_*` labels exclusively. The MODIS-based `lc_*` labels are noisier and would only introduce label-quality confounds in our fusion comparison.
