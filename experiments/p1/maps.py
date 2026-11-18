"""Visual validation of GPS-to-reference alignment on development trips."""
import json

import numpy as np


def write_map(path, geometry, route, aligned):
    trip = geometry.trip
    node_pairs = trip.nodes[trip.arcs]
    track = trip.track[:, :2]
    distance = geometry.length
    lo, hi = track[geometry.valid].min(axis=0) - 0.002, track[geometry.valid].max(axis=0) + 0.002
    in_view = ((node_pairs >= lo) & (node_pairs <= hi)).all(axis=2).any(axis=1)
    roads = node_pairs[in_view, :, ::-1].round(6).tolist()
    reference = node_pairs[route, :, ::-1].round(6).tolist()
    gps = track[geometry.valid, ::-1].round(6).tolist()
    spokes = []
    for i, arc, error in aligned:
        if i % 12 and error <= 40:
            continue
        _, fraction, _ = geometry.measure(i, np.array([arc]))
        a, b = node_pairs[arc]
        snapped = a + fraction[0] * (b - a)
        spokes.append({"points": [track[i, ::-1].tolist(), snapped[::-1].tolist()],
                       "label": f"Observation {i}, arc {arc}, error {error:.1f} m"})
    try:
        import folium
    except ModuleNotFoundError as error:
        if error.name != "folium":
            raise
        write_leaflet(path, trip.trip_id, roads, reference, gps, spokes)
        return
    result = folium.Map()
    for name, lines, color, weight in (
        ("Road network", roads, "#64748b", 1),
        ("Reference route", reference, "#16a34a", 4),
        ("GPS trace", [gps], "#2563eb", 2),
    ):
        layer = folium.FeatureGroup(name=name, show=True).add_to(result)
        folium.PolyLine(lines, color=color, weight=weight).add_to(layer)
        if name == "GPS trace":
            for point in gps:
                folium.CircleMarker(point, radius=2, color=color, fill=True).add_to(layer)
    layer = folium.FeatureGroup(name="GPS-to-reference alignments", show=True).add_to(result)
    for entry in spokes:
        folium.PolyLine(entry["points"], color="#ea580c", weight=1, tooltip=entry["label"]).add_to(layer)
    result.fit_bounds(gps)
    folium.LayerControl().add_to(result)
    result.save(path)


def write_leaflet(path, trip_id, roads, reference, gps, spokes):
    data = json.dumps(dict(roads=roads, reference=reference, gps=gps, spokes=spokes), separators=(",", ":"))
    template = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap alignment — __TRIP__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>body{margin:0;font:15px system-ui;color:#172554}header{padding:10px 16px}#map{height:calc(100dvh - 60px)}</style>
<header><strong>Trip __TRIP__</strong> · GPS and reference alignment</header><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data=__DATA__;
const map=L.map('map',{preferCanvas:true}).fitBounds(data.gps);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
 {attribution:'&copy; OpenStreetMap contributors',maxZoom:19}).addTo(map);
const roads=L.polyline(data.roads,{color:'#64748b',weight:1,opacity:.65}).addTo(map);
const route=L.polyline(data.reference,{color:'#16a34a',weight:4}).addTo(map);
const gps=L.layerGroup([L.polyline(data.gps,{color:'#2563eb',weight:2}),
 ...data.gps.map(p=>L.circleMarker(p,{radius:2,color:'#2563eb',fillOpacity:.75}))]).addTo(map);
const links=L.layerGroup(data.spokes.map(s=>L.polyline(s.points,{color:'#ea580c',weight:1})
 .bindTooltip(s.label))).addTo(map);
L.control.layers({}, {'Road network (gray)':roads,'Reference route (green)':route,
 'GPS points and trace (blue)':gps,'Sampled alignments (orange)':links},{collapsed:false}).addTo(map);
</script></html>"""
    path.write_text(template.replace("__TRIP__", trip_id).replace("__DATA__", data), encoding="utf-8")

