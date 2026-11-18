"""Offline diagnostic overview for the training trip with reference breaks."""
from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from alignment import align_reference
from data import Geometry, Trip, load_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    g = Geometry(Trip.load(args.trip_dir))
    route = load_reference(args.trip_dir, len(g.trip.arcs))
    matched = align_reference(g, route)
    gps = g.gps[g.valid]
    road = g.nodes[g.trip.arcs]
    reference = road[route]
    extent = np.vstack((gps, reference.reshape(-1, 2)))
    lo, hi = extent.min(axis=0), extent.max(axis=0)
    width, height = 1240, 860
    canvas = Image.new("RGB", (width, height), "#f5f7fa")
    d = ImageDraw.Draw(canvas)
    d.rectangle((32, 132, width - 32, height - 38), fill="white", outline="#c7d2e0")
    scale = min((width - 130) / max(hi[0] - lo[0], 1), (height - 238) / max(hi[1] - lo[1], 1))
    middle = (hi + lo) / 2
    view = np.array([width / 2, (height + 94) / 2])

    def screen(points):
        return ((points - middle) * [scale, -scale] + view).round().astype(int)

    in_view = ((road >= lo - 80) & (road <= hi + 80)).all(axis=2).any(axis=1)
    for a, b in screen(road[in_view]):
        d.line((tuple(a), tuple(b)), fill="#d1dae5", width=2)
    for a, b in screen(reference):
        d.line((tuple(a), tuple(b)), fill="#168047", width=5)
    for i in np.flatnonzero(g.valid):
        arc = route[matched[i]]
        distance, fraction, _ = g.measure(i, np.array([arc]))
        if i % 9 == 0 or distance[0] > 40:
            p, q = screen(np.stack((g.gps[i], g.start[arc] + fraction[0] * g.vector[arc])))
            d.line((tuple(p), tuple(q)), fill="#cb561a", width=2)
    for x, y in screen(gps):
        d.ellipse((x-2, y-2, x+2, y+2), fill="#2055c4")
    arc_endpoints = g.trip.arcs[route]
    breaks = np.flatnonzero(arc_endpoints[:-1, 1] != arc_endpoints[1:, 0])
    for x, y in screen(g.end[route[breaks]]):
        d.ellipse((x-8, y-8, x+8, y+8), outline="#a21b5b", width=3)

    d.rectangle((0, 0, width, 131), fill="#f5f7fa")
    d.rectangle((0, height - 37, width, height), fill="#f5f7fa")
    d.rectangle((0, 132, 31, height - 38), fill="#f5f7fa")
    d.rectangle((width - 31, 132, width, height - 38), fill="#f5f7fa")
    d.rectangle((32, 132, width - 32, height - 38), outline="#c7d2e0")
    d.text((34, 26), f"GPS-to-reference alignment   |   {g.trip.trip_id}", fill="#14253d",
           font=ImageFont.load_default(size=31))
    d.text((34, 72), f"{len(gps):,} GPS points   ·   {len(route):,} reference arcs   ·   {len(breaks)} directed reference breaks",
           fill="#465870", font=ImageFont.load_default(size=18))
    x = 34
    font = ImageFont.load_default(size=16)
    for label, color in (("Road network", "#a7b9ce"), ("Reference route", "#168047"),
                         ("GPS points", "#2055c4"), ("Sampled alignment", "#cb561a"),
                         ("Reference break", "#a21b5b")):
        d.rectangle((x, 107, x + 16, 116), fill=color)
        d.text((x + 24, 103), label, fill="#14253d", font=font)
        x += int(font.getlength(label)) + 57
    canvas.save(args.output)


if __name__ == "__main__":
    main()

