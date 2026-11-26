"""Initial DEM reprojection to the global processing CRS/grid."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import rasterio
from pyproj import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

from .exceptions import InputError, ProcessingError
from .logging_utils import get_logger

log = get_logger("reprojection")

_RESAMPLING = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "cubic": Resampling.cubic}


def reproject_dem(source: Path, destination: Path, target_crs: str, resolution: float | None,
                  resampling: str, *, resume: bool = False) -> Dict:
    """Reproject a single-band DEM. The output grid is the master grid for all subsequent stages."""
    try:
        with rasterio.open(source) as src:
            if src.count != 1:
                raise InputError(f"DEM must be single-band; found {src.count} bands.")
            if src.nodata is None:
                raise InputError("The source DEM has no nodata value in its metadata; define nodata before reprojection.")
            if src.crs is None:
                raise InputError("The source DEM has no CRS; it cannot be reprojected.")
            dst_crs = CRS.from_user_input(target_crs)
            if not dst_crs.is_projected:
                raise InputError(f"Processing CRS must be projected, got {dst_crs.name} ({target_crs}).")
            if resume and destination.exists():
                with rasterio.open(destination) as old:
                    tags = old.tags()
                    st = source.stat()
                    if (tags.get("SOURCE_PATH") == str(source.resolve()) and
                        tags.get("SOURCE_SIZE") == str(st.st_size) and
                        tags.get("SOURCE_MTIME_NS") == str(st.st_mtime_ns) and
                        tags.get("TARGET_CRS") == dst_crs.to_string() and
                        tags.get("TARGET_RESOLUTION") == str(resolution)):
                        log.info("Reusing existing reprojected DEM: %s", destination)
                        return {"path": str(destination), "reused": True, "crs": old.crs.to_string(),
                                "resolution": [old.res[0], old.res[1]], "shape": [old.height, old.width]}
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=resolution,
            )
            if width < 2 or height < 2:
                raise InputError("Reprojected DEM would have fewer than 2 rows or columns.")
            nodata = src.nodata if src.nodata is not None else -9999.0
            profile = {"driver": "GTiff", "height": height, "width": width, "count": 1,
                       "dtype": "float32", "crs": dst_crs, "transform": transform,
                       "nodata": nodata, "tiled": True, "blockxsize": 512, "blockysize": 512,
                       "compress": "deflate", "BIGTIFF": "YES"}
            tmp = destination.with_name(destination.name + ".partial")
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with rasterio.open(tmp, "w", **profile) as dst:
                    reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                              src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                              dst_transform=transform, dst_crs=dst_crs, dst_nodata=nodata,
                              resampling=_RESAMPLING[resampling], num_threads=2)
                    st = source.stat()
                    dst.update_tags(SOURCE_PATH=str(source.resolve()), SOURCE_SIZE=st.st_size,
                                    SOURCE_MTIME_NS=st.st_mtime_ns, TARGET_CRS=dst_crs.to_string(),
                                    TARGET_RESOLUTION=str(resolution), RESAMPLING=resampling)
                tmp.replace(destination)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        return {"path": str(destination), "reused": False, "crs": dst_crs.to_string(),
                "resolution": [abs(transform.a), abs(transform.e)], "shape": [height, width],
                "resampling": resampling}
    except (InputError, ProcessingError):
        raise
    except Exception as exc:
        raise ProcessingError(f"Could not reproject DEM to {target_crs}: {exc}") from exc
