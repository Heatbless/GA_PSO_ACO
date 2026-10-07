"""Small offline OSM-format fixture; never used by the application."""
import math
import unittest
from functools import lru_cache
from unittest.mock import patch
from sppg_compare.osm import parse_network


@lru_cache(maxsize=1)
def network_fixture():
    nodes = [{"type": "node", "id": i, "lat": -6.224+.006*math.sin(i*math.pi/50),
              "lon": 106.702+.007*math.cos(i*math.pi/50)} for i in range(100)]
    ways = [{"type": "way", "id": 1000, "nodes": list(range(100))+[0], "tags": {"highway": "residential"}},
            {"type": "way", "id": 1001, "nodes": [0, 50], "tags": {"highway": "primary", "name": "Jalan Ciledug Raya"}},
            {"type": "way", "id": 1002, "nodes": [25, 75], "tags": {"highway": "secondary", "oneway": "yes"}}]
    # Match the current real extract's count while keeping tests fully offline.
    schools = [{"type": "node", "id": 2000+i, "lat": -6.224+.004*math.sin(i*2*math.pi/73),
                "lon": 106.702+.006*math.cos(i*2*math.pi/73),
                "tags": {"amenity": "school", "name": f"Fixture school {i}"}} for i in range(73)]
    return parse_network({"elements": nodes+ways+schools}, "Ciledug", "offline-test-fixture")


class OsmFixtureTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(patch.stopall)
        patch("sppg_compare.osm.load_network", return_value=network_fixture()).start()
        from PIL import Image
        from sppg_compare.basemap import pixel, VIEW_BBOX
        origin = pixel(VIEW_BBOX[2], VIEW_BBOX[1])
        patch("sppg_compare.gui.load_view_tiles", return_value=(Image.new("RGB", (1188, 701), "white"), origin)).start()
