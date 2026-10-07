import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import unittest
from unittest.mock import patch

from sppg_compare.algorithms import SolverSettings, solve
from sppg_compare.experiments import export_demo
from sppg_compare.model import Scenario, generate_scenario
from sppg_compare.osm import parse_network, load_network, query_for, speed_value, xml_payload


def fixture():
    elements = [{"type": "node", "id": i, "lat": -6.24+.006*math.sin(i*math.pi/10),
                 "lon": 106.705+.006*math.cos(i*math.pi/10)} for i in range(20)]
    elements.extend([
        {"type": "way", "id": 40, "nodes": list(range(20))+[0], "tags": {"highway": "residential"}},
        {"type": "way", "id": 41, "nodes": [0, 2], "tags": {"highway": "primary", "name": "Jalan Ciledug Raya", "oneway": "yes", "maxspeed": "30"}},
        {"type": "way", "id": 42, "nodes": [3, 7], "tags": {"highway": "tertiary", "oneway": "-1"}},
        {"type": "way", "id": 43, "nodes": [4, 8], "tags": {"highway": "secondary", "maxspeed:forward": "40", "maxspeed:backward": "20"}},
        {"type": "way", "id": 44, "nodes": [5, 9], "tags": {"highway": "service", "access": "private"}},
        {"type": "way", "id": 45, "nodes": [6, 10], "tags": {"highway": "footway"}},
    ])
    elements.extend({"type": "node", "id": 100+i, "lat": -6.24+.004*math.sin(i*math.pi/10),
                     "lon": 106.705+.004*math.cos(i*math.pi/10),
                     "tags": {"amenity": "school", "name": f"School {i}"}} for i in range(20))
    return {"elements": elements, "osm3s": {"timestamp_osm_base": "2026-10-07T00:00:00Z"}}


class OsmTests(unittest.TestCase):
    def test_topology_direction_speed_class_and_restrictions(self):
        n = parse_network(fixture(), "Ciledug")
        self.assertEqual(len(n.nodes), 20)
        self.assertEqual(len(n.schools), 20)
        raya = [e for e in n.edges if e.name == "Jalan Ciledug Raya"]
        self.assertEqual(len(raya), 1)
        self.assertEqual((raya[0].source, raya[0].target, raya[0].speed_kph), ("0", "2", 50))
        self.assertTrue(raya[0].one_way)
        self.assertTrue(all(e.speed_kph == 25 for e in n.edges if e.road_class in ("residential", "tertiary")))
        self.assertTrue(all("user average" in e.speed_source for e in n.edges))
        self.assertTrue(any(e.source == "7" and e.target == "3" for e in n.edges))
        self.assertFalse(any(e.source == "3" and e.target == "7" for e in n.edges))
        self.assertEqual(next(e.speed_kph for e in n.edges if (e.source, e.target) == ("4", "8")), 50)
        self.assertEqual(next(e.speed_kph for e in n.edges if (e.source, e.target) == ("8", "4")), 50)
        self.assertFalse(any((e.source, e.target) == ("5", "9") for e in n.edges))
        self.assertFalse(any((e.source, e.target) == ("6", "10") for e in n.edges))
        self.assertEqual(n.metadata["restricted_ways_skipped"], 1)
        self.assertGreater(n.metadata["ciledug_raya_directed_segments"], 0)

    def test_speed_fallback_and_mph(self):
        self.assertAlmostEqual(speed_value("30 mph", "primary")[0], 48.28032)
        self.assertEqual(speed_value("20 km/h", "primary")[0], 20)
        for invalid in (None, "signals", "0", "999", "40;60"):
            speed, source = speed_value(invalid, "residential")
            self.assertEqual(speed, 20)
            self.assertIn("assumed", source)

    def test_cache_and_reproducible_scenario_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"ciledug.json").write_text(json.dumps(fixture()), encoding="utf-8")
            with patch("sppg_compare.osm.urllib.request.urlopen", side_effect=AssertionError("Network should not be used")):
                network = load_network("Ciledug", directory)
            with patch("sppg_compare.osm.load_network", return_value=network):
                a = generate_scenario(map_source="OpenStreetMap", osm_city="Ciledug")
                b = generate_scenario(map_source="OpenStreetMap", osm_city="Ciledug", radius=8, difficulty="Easy")
            self.assertEqual(a.road_nodes, b.road_nodes)
            self.assertEqual(a.road_edges, b.road_edges)
            self.assertEqual(a.schools, b.schools)
            self.assertEqual(a.travel_times, b.travel_times)
            self.assertEqual(Scenario.from_dict(a.to_dict()), a)
            self.assertEqual(a.generator_version, "ciledug-all-schools-routing-v3")
            self.assertIn("OSM school", a.road_source["school_locations"])
            self.assertTrue(a.road_source["raw_sha256"])
            self.assertEqual(len(a.road_source["school_features"]), 20)
            detached = a.to_dict()
            detached["road_source"]["city"] = "changed outside scenario"
            self.assertEqual(a.road_source["city"], "Ciledug")
            run = solve(a, "PSO", settings=SolverSettings(oracle_time_limit=.15, budget=31))
            export_demo(a, [run], directory)
            payload = json.loads((Path(directory)/"benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["scenarios"][0]["road_source"], a.road_source)
            self.assertEqual(run.evaluations, 31)
            self.assertIn(run.oracle.status, ("optimal", "limit_reached"))

    def test_ciledug_query_includes_named_road_and_corridor(self):
        query = query_for("Ciledug")
        self.assertIn("Ciledug Raya", query)
        self.assertIn("around.raya:450", query)
        self.assertIn("106.7534", query)
        self.assertIn("-6.21118", query)
        with self.assertRaises(ValueError):
            query_for("Unknown")

    def test_incomplete_or_missing_required_road_is_rejected(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            parse_network(dict(data, remark="timeout"), "Ciledug")
        bad = copy.deepcopy(data)
        bad["elements"] = [e for e in bad["elements"] if e["id"] != 2]
        with self.assertRaisesRegex(ValueError, "missing referenced"):
            parse_network(bad, "Ciledug")
        bad = copy.deepcopy(data)
        bad["elements"] = [e for e in bad["elements"] if e["id"] != 41]
        with self.assertRaisesRegex(ValueError, "missing Ciledug Raya"):
            parse_network(bad, "Ciledug")

    def test_map_api_conversion_retains_roads_and_school_centres(self):
        root = ET.fromstring('''<osm><node id="1" lat="-6.24" lon="106.70"/>
            <node id="2" lat="-6.239" lon="106.701"/>
            <node id="3" lat="-6.238" lon="106.702"/>
            <way id="4"><nd ref="1"/><nd ref="2"/><tag k="highway" v="primary"/><tag k="name" v="Jalan Ciledug Raya"/></way>
            <way id="5"><nd ref="2"/><nd ref="3"/><tag k="amenity" v="school"/><tag k="name" v="School"/></way>
            <way id="6"><nd ref="1"/><nd ref="3"/><tag k="building" v="yes"/></way></osm>''')
        payload = xml_payload([root, root])
        self.assertEqual(len(payload["elements"]), 5)
        school = next(e for e in payload["elements"] if e["id"] == 5)
        self.assertAlmostEqual(school["center"]["lat"], -6.2385)
        self.assertFalse(any(e["id"] == 6 for e in payload["elements"]))
        self.assertEqual(payload["download_api"], "OpenStreetMap map API bounded tiles")

    def test_school_relations_are_retained_with_member_geometry(self):
        root = ET.fromstring('''<osm><node id="1" lat="-6.24" lon="106.70"/>
            <node id="2" lat="-6.239" lon="106.701"/>
            <way id="3"><nd ref="1"/><nd ref="2"/><tag k="building" v="yes"/></way>
            <relation id="4"><member type="way" ref="3" role="outer"/>
            <tag k="amenity" v="school"/><tag k="name" v="School campus"/></relation></osm>''')
        payload = xml_payload([root])
        relation = next(e for e in payload["elements"] if e["type"] == "relation")
        self.assertEqual(relation["tags"]["name"], "School campus")
        self.assertAlmostEqual(relation["center"]["lat"], -6.2395)
        self.assertEqual(payload["school_import_version"], 2)
