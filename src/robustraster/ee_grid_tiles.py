import math
import ee

def ee_covering_grid_tiles(aoi, crs: str, scale: float, tile_max_pixels: int):
    """
    Returns an ee.FeatureCollection of tile geometries covering the AOI (server-side).
    Tiles are aligned to the specified CRS and sized based on scale + tile_max_pixels.
    """
    # Normalize AOI to ee.Geometry
    if isinstance(aoi, ee.FeatureCollection):
        aoi_geom = aoi.geometry()
    elif isinstance(aoi, ee.Feature):
        aoi_geom = aoi.geometry()
    else:
        aoi_geom = ee.Geometry(aoi)

    # Compute tile side (meters) from pixel budget
    tile_side_px = int(math.sqrt(int(tile_max_pixels)))
    if tile_side_px <= 0:
        raise ValueError("tile_max_pixels must be > 0")

    tile_side_m = int(tile_side_px * float(scale))
    if tile_side_m <= 0:
        raise ValueError("scale must be > 0")

    # Projection defines grid alignment and cell size
    proj = ee.Projection(crs).atScale(tile_side_m)

    # Create grid cells covering AOI
    grid_fc = aoi_geom.coveringGrid(proj)

    # Intersect each cell with AOI so tiles don't extend beyond it
    def clip_cell(f):
        f = ee.Feature(f)
        g = f.geometry().intersection(aoi_geom, ee.ErrorMargin(1))
        return ee.Feature(g).copyProperties(f)

    tiles_fc = grid_fc.map(clip_cell)

    # Remove any empty geometries
    #tiles_fc = tiles_fc.filter(ee.Filter.geometry())

    return tiles_fc

def iter_tiles_from_fc(tiles_fc, batch_size=200):
    """
    Generator that yields ee.Geometry tiles from a server-side FeatureCollection.
    Tiles are fetched from Earth Engine in batches (default 200) so we don't
    pull the entire FeatureCollection client-side at once.
    """
    total = tiles_fc.size().getInfo()  # one safe getInfo() for tile count

    for offset in range(0, total, batch_size):
        batch_list = tiles_fc.toList(batch_size, offset)
        batch_len = min(batch_size, total - offset)

        for i in range(batch_len):
            geom = ee.Feature(batch_list.get(i)).geometry()
            yield geom