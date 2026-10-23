"""NOAA source discovery, bounded downloads, calibration and spatial registration.

GOES supplies current B2/CSM observations; NBM supplies weather conditioning.
Fitted B2 processing and input packing live in dataset.py. Heavy decoding and
projection dependencies are imported where needed, so the GRIB subprocess never
imports pyproj. Both HTTP block and working-file caches remain bounded.
"""