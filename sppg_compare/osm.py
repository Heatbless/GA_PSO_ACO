"""Cached OSM road imports, retaining node topology and travel direction."""
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .roads import RoadEdge, RoadNode

CITIES = {"Ciledug": (-6.275, 106.675, -6.205, 106.810)}
FOCUS_CENTER = (-6.23118, 106.72340)
FOCUS_BBOX = (-6.25118, 106.69340, -6.21118, 106.75340)
FOCUS_URL = "https://www.openstreetmap.org/#map=15/-6.23118/106.72340"
SPEEDS = {"motorway": 70., "trunk": 50., "primary": 40., "secondary": 35.,
          "tertiary": 30., "unclassified": 25., "residential": 20.,
          "living_street": 10., "service": 15.}
ENDPOINTS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter")
ATTRIBUTION = "© OpenStreetMap contributors"
LICENSE_URL = "https://www.openstreetmap.org/copyright"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "osm_cache"
CILEDUG_TILES = {"ciledug_core": (106.6865, -6.2577, 106.7234, -6.2246),
                 "ciledug_raya_west_south": (106.7234, -6.247, 106.755, -6.2335),
                 "ciledug_raya_west_north": (106.7234, -6.2335, 106.755, -6.220),
                 "ciledug_raya_east_sw": (106.755, -6.247, 106.775, -6.2335),
                 "ciledug_raya_east_nw": (106.755, -6.2335, 106.775, -6.220),
                 "ciledug_raya_east_se": (106.775, -6.247, 106.795, -6.2335),
                 "ciledug_raya_east_ne": (106.775, -6.2335, 106.795, -6.220)}
CILEDUG_TILES.update({"linked_focus_north_west": (106.69340, -6.2246, 106.7234, -6.21118),
                      "linked_focus_north_east": (106.7234, -6.220, 106.7534, -6.21118),
                      "linked_focus_south_east": (106.7234, -6.25118, 106.7534, -6.247)})


def map_api_fallback(directory):
    """Bounded Ciledug extracts when public Overpass services are unavailable."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def download_tile(name, bbox, depth=0):
        path = directory / (name + ".osm")
        if not path.exists():
            request = urllib.request.Request("https://api.openstreetmap.org/api/0.6/map?bbox=" + ",".join(map(str, bbox)),
                       headers={"User-Agent": "SPPG-Assignment-Lab/2.1 (educational bounded road extract)"})
            try:
                with urllib.request.urlopen(request, timeout=55) as response:
                    raw = response.read()
            except urllib.error.HTTPError as error:
                reason = error.read().decode("utf-8", errors="replace")
                if error.code != 400 or "too many nodes" not in reason.lower() or depth >= 5:
                    raise ValueError(f"OSM map extract failed: HTTP {error.code}: {reason}") from error
                west, south, east, north = bbox
                if east-west >= north-south:
                    middle = (west+east)/2
                    halves = ((west, south, middle, north), (middle, south, east, north))
                else:
                    middle = (south+north)/2
                    halves = ((west, south, east, middle), (west, middle, east, north))
                return download_tile(name+"_a", halves[0], depth+1) + download_tile(name+"_b", halves[1], depth+1)
            ET.fromstring(raw)
            path.write_bytes(raw)
        return [ET.fromstring(path.read_bytes())]
    parts = []
    for name, bbox in CILEDUG_TILES.items():
        parts.extend(download_tile(name, bbox))
    return xml_payload(parts)


def xml_payload(parts):
    elements = {}
    for root in parts:
        for element in root:
            kind = element.tag
            if kind not in ("node", "way", "relation"):
                continue
            record = {"type": kind, "id": int(element.attrib["id"]),
                      "tags": {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}}
            if kind == "node":
                record.update(lat=float(element.attrib["lat"]), lon=float(element.attrib["lon"]))
            elif kind == "way":
                record["nodes"] = [int(node.attrib["ref"]) for node in element.findall("nd")]
            else:
                record["members"] = [dict(member.attrib) for member in element.findall("member")]
            elements[kind, record["id"]] = record
    records = []
    needed_nodes = set()
    def geometry_nodes(record, seen=None):
        seen = set() if seen is None else seen
        key = (record["type"], record["id"])
        if key in seen:
            return set()
        seen.add(key)
        if record["type"] == "node":
            return {record["id"]}
        if record["type"] == "way":
            return set(record["nodes"])
        keys = set()
        for member in record.get("members", []):
            child = elements.get((member["type"], int(member["ref"])))
            if child:
                keys.update(geometry_nodes(child, seen))
        return keys
    for record in elements.values():
        tags = record.get("tags", {})
        if record["type"] == "way" and (tags.get("highway", "").removesuffix("_link") in SPEEDS or tags.get("amenity") == "school"):
            records.append(record)
            needed_nodes.update(record["nodes"])
            if tags.get("amenity") == "school":
                points = [elements["node", key] for key in record["nodes"] if ("node", key) in elements]
                if points:
                    record["center"] = {axis: sum(p[axis] for p in points)/len(points) for axis in ("lat", "lon")}
        elif record["type"] == "node" and tags.get("amenity") == "school":
            needed_nodes.add(record["id"])
        elif record["type"] == "relation" and tags.get("amenity") == "school":
            keys = geometry_nodes(record)
            points = [elements["node", key] for key in sorted(keys) if ("node", key) in elements]
            if points:
                record["center"] = {axis: sum(p[axis] for p in points)/len(points) for axis in ("lat", "lon")}
            records.append(record)
            needed_nodes.update(keys)
    records.extend(elements["node", key] for key in sorted(needed_nodes) if ("node", key) in elements)
    return {"elements": records, "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "download_api": "OpenStreetMap map API bounded tiles",
            "scenario_focus_bbox": list(FOCUS_BBOX),
            "school_import_version": 2,
            "download_tile_bboxes_west_south_east_north": list(CILEDUG_TILES.values())}


@dataclass(frozen=True)
class RoadNetwork:
    nodes: tuple
    edges: tuple
    schools: tuple
    metadata: dict = field(hash=False)


def speed_value(value, highway):
    """Numeric speed tags (including mph); conservative stated class fallback."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kph)?\s*", str(value or ""), re.I)
    if match:
        speed = float(match[1]) * (1.609344 if (match[2] or "").lower() == "mph" else 1)
        if 0 < speed <= 160:
            return speed, "OSM maxspeed (free-flow assumption)"
    return SPEEDS.get(highway.removesuffix("_link"), 20.), "assumed highway-class speed"


def query_for(city):
    if city not in CITIES:
        raise ValueError(f"Unknown OSM city: {city}")
    bbox = ",".join(map(str, CITIES[city]))
    kinds = "|".join(SPEEDS) + "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
    if city == "Ciledug":
        core = ",".join(map(str, FOCUS_BBOX))
        return (f'[out:json][timeout:40];way["highway"]["name"~"Ciledug Raya",i]({bbox})->.raya;'
                f'(way["highway"~"^({kinds})$"]({core});.raya;'
                f'way(around.raya:450)["highway"~"^({kinds})$"];'
                f'nwr["amenity"="school"]({core}););out body center;>;out skel qt;')
    return f'[out:json][timeout:40];(way["highway"~"^({kinds})$"]({bbox});nwr["amenity"="school"]({bbox}););out body center;>;out skel qt;'


def planning_map_fallback(directory):
    """Import just the screenshot extent in bounded map-API pieces."""
    from .basemap import VIEW_BBOX
    south, west, north, east = VIEW_BBOX
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def part(name, bbox, depth=0):
        path = directory/(name+".osm")
        if path.exists():
            return [ET.fromstring(path.read_bytes())]
        request = urllib.request.Request("https://api.openstreetmap.org/api/0.6/map?bbox="+",".join(map(str, bbox)),
                          headers={"User-Agent": "SPPG-Lab/3.2 (+https://github.com/Heatbless/GA_PSO_ACO)"})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            reason = error.read().decode("utf-8", errors="replace")
            if error.code != 400 or "too many nodes" not in reason.lower() or depth >= 5:
                raise ValueError(f"Planning map import: HTTP {error.code}: {reason}") from error
            w, s, e, n = bbox
            mid = (w+e)/2
            return part(name+"a", (w, s, mid, n), depth+1)+part(name+"b", (mid, s, e, n), depth+1)
        root = ET.fromstring(raw)
        path.write_bytes(raw)
        return [root]
    parts = []
    for x in range(4):
        for y in range(2):
            parts.extend(part(f"pedurenan_{x}_{y}", (west+(east-west)*x/4, south+(north-south)*y/2,
                                                    west+(east-west)*(x+1)/4, south+(north-south)*(y+1)/2)))
    return xml_payload(parts)


@lru_cache(maxsize=4)
def load_network(city="Ciledug", cache_dir=None, *, planning=False):
    query = query_for(city)
    path = Path(cache_dir or CACHE_DIR) / f"{city.lower()}.json"
    if path.exists():
        raw = path.read_bytes()
    else:
        failures = []
        for endpoint in ENDPOINTS:
            try:
                request = urllib.request.Request(endpoint + "?" + urllib.parse.urlencode({"data": query}),
                    headers={"User-Agent": "SPPG-Assignment-Lab/2.1 (educational OSM road comparison)"})
                with urllib.request.urlopen(request, timeout=55) as response:
                    raw = response.read()
                payload = json.loads(raw)
                if payload.get("remark") or not payload.get("elements"):
                    raise ValueError(payload.get("remark", "Empty OSM response"))
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(raw)
                temporary.replace(path)
                break
            except (OSError, ValueError) as error:
                failures.append(f"{endpoint}: {error}")
        else:
            if city == "Ciledug":
                try:
                    payload = map_api_fallback(path.parent)
                    raw = json.dumps(payload, separators=(",", ":")).encode()
                    path.write_bytes(raw)
                except (OSError, ValueError, ET.ParseError) as error:
                    raise ValueError("OSM download failed. Cached tiles are retained for retry. "
                                     + "; ".join(failures) + f"; map API: {error}") from error
            else:
                raise ValueError("OSM download failed. Retry later or supply a cached Overpass JSON at "
                                 + str(path) + ". " + "; ".join(failures))
    payload = json.loads(raw)
    if payload.get("download_api") and (payload.get("scenario_focus_bbox") != list(FOCUS_BBOX) or payload.get("school_import_version", 1) < 2):
        # Reuse old downloaded tiles and add only the missing focused coverage.
        payload = map_api_fallback(path.parent)
        raw = json.dumps(payload, separators=(",", ":")).encode()
        path.with_name("ciledug_previous_area.json").write_bytes(path.read_bytes())
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)
    if planning and not payload.get("planning_view_version"):
        from .basemap import VIEW_BBOX
        kinds = "|".join(SPEEDS) + "|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"
        extra_query = '[out:json][timeout:35];way["highway"~"^('+kinds+')$"]('+','.join(map(str, VIEW_BBOX))+');out body;>;out skel qt;'
        failures = []
        try:
            extra = planning_map_fallback(path.parent)
        except (OSError, ValueError, ET.ParseError) as error:
            failures.append(str(error))
            extra = None
        for endpoint in (() if extra is not None else ENDPOINTS):
            try:
                request = urllib.request.Request(endpoint+"?"+urllib.parse.urlencode({"data": extra_query}),
                          headers={"User-Agent": "SPPG-Lab/3.2 (+https://github.com/Heatbless/GA_PSO_ACO)"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    extra = json.loads(response.read())
                if extra.get("remark") or not extra.get("elements"):
                    raise ValueError(extra.get("remark", "Empty planning area extract"))
                break
            except (OSError, ValueError) as error:
                failures.append(str(error))
                extra = None
        if extra is None:
            raise ValueError("Could not load roads for the supplied Pedurenan view: " + "; ".join(failures))
        payload["elements"].extend(extra["elements"])
        payload["planning_view_version"] = 1
        payload["planning_view_bbox"] = list(VIEW_BBOX)
        raw = json.dumps(payload, separators=(",", ":")).encode()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)
    return parse_network(payload, city, hashlib.sha256(raw).hexdigest())


def parse_network(payload, city, digest="fixture"):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if payload.get("remark"):
        raise ValueError("Incomplete Overpass response: " + str(payload["remark"]))
    bbox = CITIES[city]
    lat0, lon0 = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
    def project(lat, lon):
        return (6371.0088 * math.radians(lon-lon0)*math.cos(math.radians(lat0)),
                6371.0088 * math.radians(lat-lat0))
    elements = {}
    for element in payload.get("elements", []):
        key = (element["type"], element["id"])
        # Recursion may repeat nodes without tags; keep the richer record.
        elements[key] = dict(elements.get(key, {}), **element)
    coordinates = {str(e["id"]): project(e["lat"], e["lon"])
                   for e in elements.values() if e["type"] == "node" and "lat" in e}
    inside = {str(e["id"]) for e in elements.values() if e["type"] == "node" and "lat" in e
              and bbox[0] <= e["lat"] <= bbox[2] and bbox[1] <= e["lon"] <= bbox[3]}
    edges, schools = [], []
    skipped = 0
    for e in elements.values():
        tags = e.get("tags", {})
        if tags.get("amenity") == "school":
            location = e if "lat" in e else e.get("center")
            if location:
                x, y = project(location["lat"], location["lon"])
                schools.append((f"{e['type']}/{e['id']}", tags.get("name", "Unnamed school"), x, y))
        highway = tags.get("highway", "")
        if e["type"] != "way" or highway.removesuffix("_link") not in SPEEDS:
            continue
        access = tags.get("motorcar", tags.get("motor_vehicle", tags.get("vehicle", tags.get("access", "yes"))))
        if access in ("no", "private") or tags.get("area") == "yes":
            skipped += 1
            continue
        direction = tags.get("oneway:motorcar", tags.get("oneway:motor_vehicle", tags.get("oneway:vehicle",
                    tags.get("oneway", "yes" if tags.get("junction") == "roundabout" or highway in ("motorway", "motorway_link") else "no"))))
        for first, last in zip(e.get("nodes", []), e.get("nodes", [])[1:]):
            a, b = str(first), str(last)
            if a not in coordinates or b not in coordinates:
                raise ValueError("OSM road geometry is incomplete: missing referenced nodes.")
            if a == b:
                continue
            # Map API returns complete ways that can extend well outside the
            # requested region. Keep routing and map extents within the area.
            if a not in inside or b not in inside:
                continue
            directions = ((a, b, "forward"),) if direction in ("yes", "1", "true") else ((b, a, "backward"),) if direction == "-1" else ((a, b, "forward"), (b, a, "backward"))
            for source, target, suffix in directions:
                main = highway.removesuffix("_link") in ("motorway", "trunk", "primary", "secondary")
                speed = 50. if main else 25.
                source_tag = "assumed user average: main road 50 km/h" if main else "assumed user average: local street 25 km/h"
                edges.append(RoadEdge(source, target, speed, True, highway, tags.get("name", ""), source_tag))
    if not edges:
        raise ValueError("No usable drivable OSM roads found.")
    keys = sorted({key for edge in edges for key in (edge.source, edge.target)})
    index = {key: i for i, key in enumerate(keys)}
    graph = coo_matrix((np.ones(len(edges)), ([index[e.source] for e in edges], [index[e.target] for e in edges])),
                       shape=(len(keys), len(keys))).tocsr()
    _, labels = connected_components(graph, directed=True, connection="strong")
    winner = int(np.argmax(np.bincount(labels)))
    retained = {key for key in keys if labels[index[key]] == winner}
    if len(retained) < 10:
        raise ValueError("OSM area has no sufficiently large mutually reachable driving network.")
    edges = tuple(e for e in edges if e.source in retained and e.target in retained)
    nodes = tuple(RoadNode(key, *coordinates[key]) for key in keys if key in retained)
    metadata = {"source": "OpenStreetMap", "city": city, "bbox_south_west_north_east": list(bbox),
                "attribution": ATTRIBUTION, "license_url": LICENSE_URL, "raw_sha256": digest,
                "osm_timestamp": payload.get("osm3s", {}).get("timestamp_osm_base"),
                "projection": "local equirectangular kilometres", "projection_origin_lat_lon": [lat0, lon0],
                "focus_center_lat_lon": list(FOCUS_CENTER), "focus_zoom": 15,
                "focus_bbox_south_west_north_east": list(FOCUS_BBOX), "focus_url": FOCUS_URL,
                "routing": "largest strongly connected driving component; one-way tags respected",
                "nodes_dropped_outside_component": len(keys)-len(nodes), "restricted_ways_skipped": skipped,
                "time_model": "static user average: main roads 50 km/h, local streets 25 km/h; maxspeed tags do not override averages; simulated school access 10 km/h; no traffic or turn restrictions",
                "average_speed_kph": {"main_road": 50., "local_street": 25.},
                "main_road_classes": ["motorway", "trunk", "primary", "secondary"]}
    metadata["download_api"] = payload.get("download_api", "Overpass API")
    metadata["downloaded_utc"] = payload.get("downloaded_utc")
    metadata["tagged_speed_segments"] = sum(e.speed_source.startswith("OSM maxspeed") for e in edges)
    metadata["assumed_speed_segments"] = sum(e.speed_source.startswith("assumed") for e in edges)
    if city == "Ciledug":
        raya = [e for e in edges if "ciledug raya" in e.name.lower()]
        if not raya:
            raise ValueError("Imported network is missing Ciledug Raya. Remove the stale cache and retry the corridor query.")
        metadata["coverage"] = ("Ciledug core plus adjoining Ciledug Raya road neighborhoods through bounded map tiles"
                                if payload.get("download_api") else "Ciledug district bounding box plus Ciledug Raya and a 450 m corridor of connecting streets")
        if payload.get("download_tile_bboxes_west_south_east_north"):
            metadata["download_tile_bboxes_west_south_east_north"] = payload["download_tile_bboxes_west_south_east_north"]
        metadata["ciledug_raya_directed_segments"] = len(raya)
        metadata["ciledug_raya_names"] = sorted({e.name for e in raya})
    return RoadNetwork(nodes, edges, tuple(sorted(schools)), metadata)


def generate_osm_scenario(size, seed, radius, slack, difficulty, city, *, network=None):
    from .model import Kitchen, School, Scenario, SIZES
    network = network or load_network(city)
    nk, _, default_seed = SIZES[size]
    seed = default_seed if seed is None else seed
    rng = random.Random(seed)
    candidates = list(network.nodes)
    site_candidates = candidates
    if city == "Ciledug":
        lat0, lon0 = network.metadata["projection_origin_lat_lon"]
        def in_core(n):
            lat = lat0 + math.degrees(n.y/6371.0088)
            lon = lon0 + math.degrees(n.x/(6371.0088*math.cos(math.radians(lat0))))
            return FOCUS_BBOX[0] <= lat <= FOCUS_BBOX[2] and FOCUS_BBOX[1] <= lon <= FOCUS_BBOX[3]
        site_candidates = [n for n in candidates if in_core(n)]
        if len(site_candidates) < 2*nk:
            raise ValueError("Too few usable road nodes in Ciledug core for the requested scenario size.")
    chosen = [rng.choice(site_candidates)]
    while len(chosen) < nk:
        chosen.append(max(site_candidates, key=lambda n: min(math.hypot(n.x-k.x, n.y-k.y) for k in chosen)))
    nodes, edges = list(network.nodes), list(network.edges)
    source = dict(network.metadata)
    bindings, schools, school_sources = [], [], []
    mapped_schools = [s for s in network.schools if in_core(RoadNode("school", s[2], s[3]))]
    if mapped_schools:
        selected = sorted(mapped_schools)
        for i, (osm_id, name, x, y) in enumerate(selected):
            key = f"school@{i}"
            nearest = min(candidates, key=lambda n: math.hypot(n.x-x, n.y-y))
            nodes.append(RoadNode(key, x, y))
            edges.append(RoadEdge(key, nearest.id, 10., False, "access", "Simulated school access", "assumed access speed"))
            bindings.append(key)
            # Demand belongs to the OSM school identity, independent of both
            # facility configuration and scenario/optimizer seeds.
            demand = random.Random("ciledug-school-demand-v1:"+osm_id).randint(80, 300)
            schools.append(School(f"S{i+1:02d}", x, y, demand))
            school_sources.append({"school_id": schools[-1].id, "osm_id": osm_id, "name": name,
                                   "access_km": math.hypot(nearest.x-x, nearest.y-y)})
        source["school_locations"] = "OSM school points/centres; straight simulated access links to nearest retained road node"
    else:
        raise ValueError("No mapped OSM schools have usable locations inside the selected Ciledug area. No simulated school fallback is used.")
    ns = len(schools)
    source["school_features"] = school_sources
    source["school_selection"] = "all located amenity=school OSM nodes, ways and relations inside the focus bbox; no subsampling"
    source["school_feature_count"] = ns
    source["school_demand_seed_policy"] = "deterministic per OSM feature ID; identical across facility configurations and seeds"
    source["kitchen_locations"] = "simulated candidate SPPG sites at OSM road nodes"
    source["demand_capacity"] = "synthetic daily meal demand and capacity; not official SPPG data"
    source["site_sampling"] = "SPPG candidates and mapped schools inside the fixed bbox around linked point -6.23118,106.72340; requested radius does not resample sites"
    loads, remaining = [0]*nk, set(range(ns))
    # Give every kitchen a school, then nearest-site assignments define base loads.
    for j, kitchen in enumerate(chosen):
        if not remaining:
            break
        i = min(remaining, key=lambda i: math.hypot(schools[i].x-kitchen.x, schools[i].y-kitchen.y))
        remaining.remove(i)
        loads[j] += schools[i].demand
    for i in sorted(remaining):
        j = min(range(nk), key=lambda j: math.hypot(schools[i].x-chosen[j].x, schools[i].y-chosen[j].y))
        loads[j] += schools[i].demand
    kitchens = tuple(Kitchen(f"K{j+1}", n.x, n.y, max(1, math.ceil(loads[j]*(1+slack)))) for j, n in enumerate(chosen))
    return Scenario(size, seed, radius, kitchens, tuple(schools), slack, difficulty, "osm-driving-v1",
                    tuple(nodes), tuple(edges), tuple(n.id for n in chosen), tuple(bindings), "travel_time", source)
