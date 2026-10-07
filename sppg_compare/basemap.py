"""Georeferenced raster background for the user's Pedurenan OSM view."""
import math
import os
import time
import urllib.request
from pathlib import Path
from PIL import Image

VIEW_CENTER = (-6.22253, 106.69706)
VIEW_ZOOM = 16
VIEW_URL = "https://www.openstreetmap.org/#map=16/-6.22253/106.69706"
CACHE = Path(__file__).resolve().parent.parent / "data" / "map_tiles"


def pixel(lat, lon, zoom=VIEW_ZOOM):
    size = 256 * 2**zoom
    return (lon+180)/360*size, (1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*size


def latlon(x, y, zoom=VIEW_ZOOM):
    size = 256 * 2**zoom
    return math.degrees(math.atan(math.sinh(math.pi*(1-2*y/size)))), x/size*360-180


cx, cy = pixel(*VIEW_CENTER)
south, west = latlon(cx-594, cy+350.5)
north, east = latlon(cx+594, cy-350.5)
VIEW_BBOX = (south, west, north, east)


def load_view_tiles():
    """Fetch only the displayed fixed viewport, with a minimum seven-day cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    left, top = pixel(VIEW_BBOX[2], VIEW_BBOX[1])
    right, bottom = pixel(VIEW_BBOX[0], VIEW_BBOX[3])
    x0, y0 = math.floor(left/256), math.floor(top/256)
    x1, y1 = math.floor(right/256), math.floor(bottom/256)
    image = Image.new("RGB", ((x1-x0+1)*256, (y1-y0+1)*256), "#f1efe9")
    template = os.environ.get("SPPG_TILE_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
    for x in range(x0, x1+1):
        for y in range(y0, y1+1):
            path = CACHE / f"{VIEW_ZOOM}_{x}_{y}.png"
            if not path.exists() or time.time()-path.stat().st_mtime > 7*86400:
                request = urllib.request.Request(template.format(z=VIEW_ZOOM, x=x, y=y), headers={
                    "User-Agent": "SPPG-Lab/3.2 (+https://github.com/Heatbless/GA_PSO_ACO)"})
                with urllib.request.urlopen(request, timeout=15) as response:
                    raw = response.read()
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(raw)
                with Image.open(temporary) as tile:
                    tile.verify()
                temporary.replace(path)
            with Image.open(path) as tile:
                image.paste(tile.convert("RGB"), ((x-x0)*256, (y-y0)*256))
    return image, (x0*256, y0*256)


def paint_view(raster, origin, bbox, size):
    left, top = pixel(bbox[2], bbox[1])
    right, bottom = pixel(bbox[0], bbox[3])
    return raster.transform(size, Image.Transform.EXTENT,
                            (left-origin[0], top-origin[1], right-origin[0], bottom-origin[1]),
                            Image.Resampling.BILINEAR, fillcolor="#f1efe9")
