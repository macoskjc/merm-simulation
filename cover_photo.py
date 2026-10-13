"""
Fit the cover-art RNP simulation into a micrograph of a real cell.

The cell photo (nucleus in blue, cytoplasm in green) is segmented, the model's nucleus is
shaped and placed to match the photo's, and each RNP slot is filled with a simulated path
that stays inside the green cytoplasm.

    python cover_photo.py --photo cover_output/cell_photo.png --px-per-um 26.2

26.2 px/um is the cover photo's scale: it is the middle cell of Fig. 1F in Moran et al.
2022 (J Virol, doi:10.1128/jvi.00934-22; A549 cells, VSV-PeGFP green, DAPI blue), and the
crop is 11.2x the figure, whose 25 um bar is 58.4 px (2.34 px/um).

Geometry (--geometry; the RNP dynamics are simulation.py's, unchanged, in every case):
  ellipse (default)  the model's nucleus is an ellipse fitted to the photo's (by moments,
                     enlarged 6% to cover its blurred edge), so RNPs that reach it slide
                     along the real nucleus outline; each RNP starts 2.5 um outside it in
                     its own direction, as in the model (5 um nucleus -> start at 7.5 um).
                     Done by swapping simulation.py's outside_nucleus() before it compiles.
  photo              a circle just covering the photo's nucleus, start 2.5 um outside it
  model              simulation.py's own 5 um nucleus and 15 um cell

How the paths are chosen (illustration only):
  * Each slot (start angle) is simulated --tries times (seeded), in its own process so
    simulation.py compiles with that slot's start position; slots run in parallel.
  * Most 20-min paths leave a cell this size, so only paths that stay inside the photo's
    cytoplasm, and within their own wedge around the nucleus, are kept. Slots then choose in
    turn (fewest candidates first), each taking its best path that stays --min-gap-um clear
    of every path already chosen, so no two RNPs touch. "Best" is the farthest-travelling
    path (--prefer travel) or the one running along the most nucleus wall (--prefer hug).
    The script prints what fraction fitted each slot. The chosen paths are cached in
    --outdir, so redrawing with the same settings skips the simulation.
  * Because the RNPs do not interact, this is equivalent to resimulating whole 8-RNP
    draws until all eight fit, only much faster.

Outputs (in --outdir):
    photo_overlay.png    transparent, 3x the photo's pixel size: RNP paths (white) with
                         directed runs (orange), the protein (purple), legend and scale bar
                         (and, with --bacteria, goodsell.py's panels in a black corner)
    photo_overlay_nolabels.png   the same without any text, legend or scale bar
    cover_photo_final.png        the cover: 8.5 x 11 in at 300 dpi, full-bleed black, no text,
                                 the cell as large as fits below a masthead band (top 26%),
                                 the photo smoothed back from its blocky enlargement, isolated
                                 from neighbouring cells and lifted, and the paths glowing
    cover_photo_final_preview.png  the same with the masthead area outlined
    photo_preview.png    the photo with the overlay on top
    photo_choice.json    scale, geometry and seeds, so the choice can be reproduced
"""

import os
import tempfile

os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="merm_cover_numba_")

import argparse
import importlib
import hashlib
import json
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from numba import njit
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from cover_art import DIRECTED, TEXT, runs_by_state, seed_numba, simulate_protein

UPSCALE = 3              # overlay pixels per photo pixel
DPI = 300


def masks(photo, green_thr, smooth_px, edge_margin_px, nucleus_margin_px):
    """Cytoplasm where paths may go, the nucleus, and the nucleus centroid (photo pixels)."""
    im = np.asarray(Image.open(photo).convert("RGB")).astype(float)
    g = ndi.gaussian_filter(im[..., 1], smooth_px)
    b = ndi.gaussian_filter(im[..., 2], smooth_px)

    def largest(m):
        lab, n = ndi.label(m)
        return lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)

    cell = largest(ndi.binary_fill_holes((g > green_thr) | (b > 40)))
    nuc = ndi.binary_fill_holes(largest((b > 60) & (b > g)))
    ys, xs = np.nonzero(nuc)
    centre = np.array([xs.mean(), ys.mean()])
    reach = np.hypot(xs - centre[0], ys - centre[1]).max()
    ok = (ndi.binary_erosion(cell, iterations=edge_margin_px)
          & ~ndi.binary_dilation(nuc, iterations=nucleus_margin_px))
    return im.astype(np.uint8), cell, nuc, ok, centre, reach


def to_px(xy_um, centre, s):
    return np.c_[centre[0] + xy_um[:, 0] * s, centre[1] - xy_um[:, 1] * s]


def fits(xy_um, ok, centre, s):
    p = np.rint(to_px(xy_um, centre, s)).astype(int)
    H, W = ok.shape
    inside = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    return bool(inside.all() and ok[p[:, 1], p[:, 0]].all())


def nucleus_ellipse(nuc, s, scale):
    """Moment-fitted ellipse of the nucleus mask: semi-axes (um) and the major axis's angle
    in model coordinates (y up), enlarged by `scale` to cover the blurred edge."""
    ys, xs = np.nonzero(nuc)
    ev, evec = np.linalg.eigh(np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()])))
    a, b = 2 * np.sqrt(ev[::-1]) * scale / s
    major = evec[:, 1]
    return a, b, np.arctan2(-major[1], major[0])      # image y points down, the model's up


def ellipse_radius(a, b, phi, theta):
    t = theta - phi
    return a * b / np.hypot(b * np.cos(t), a * np.sin(t))


def make_outside_ellipse(a_m, b_m, phi):
    """Drop-in for simulation.outside_nucleus: True outside the ellipse (metres)."""
    c, sn, ia2, ib2 = np.cos(phi), np.sin(phi), 1 / a_m ** 2, 1 / b_m ** 2

    @njit(cache=False)
    def outside_nucleus(x, y):
        u = x * c + y * sn
        v = -x * sn + y * c
        return u * u * ia2 + v * v * ib2 > 1.0

    return outside_nucleus


def simulate_slot(job):
    """Simulate RNPs from one start angle. Runs in its own process, so simulation.py compiles
    with this slot's geometry: its numba functions read outside_nucleus() and CELL_RADIUS
    (start = CELL_RADIUS / 2) at compile time. Search mode returns every path that fits, best
    first, as coarse outlines; replay mode (job["replay"] = a seed) returns that path in full."""
    random.seed(job["seed"])                   # simulation.py's import-time k draw
    sim = importlib.import_module("simulation")
    if job["ellipse_m"]:
        sim.outside_nucleus = make_outside_ellipse(*job["ellipse_m"])
    else:
        sim.NUCLEUS_RADIUS = job["nucleus_radius_m"]
    sim.CELL_RADIUS = 2 * job["start_m"]
    cfg = sim.SimulationConfig(total_time=int(job["minutes"] * 60), p_driv=job["p_driv"])
    every = int(round(0.010 / cfg.dt))
    a, ok, centre, s = job["angle"], job["ok"], job["centre"], job["s"]

    def run(seed):
        seed_numba(seed)
        o = sim.move(cfg, theta=a, stop_on_cell_exit=False)
        return o, np.c_[o.x[::every], o.y[::every]] * 1e6

    if job.get("replay") is not None:
        o, xy = run(job["replay"])
        # 10-ms lines that are part of a hop to a new trap: from the moment the trap centre
        # jumps (simulation.py's hop times; hop_t[0] is the first trap) until the RNP has
        # settled into the new trap, 3 relaxation times (drag / stiffness) later
        step, settle = every * cfg.dt, 3 * sim.g / sim.k
        hops = np.unique(np.concatenate(
            [np.arange(int(t // step), int((t + settle) // step) + 1) for t in o.hop_t[1:]]
            or [np.zeros(0, int)]))
        return job["slot"], {"seed": job["replay"], "xy": xy,
                             "state": o.state_dt[::every].astype(bool), "hops": hops[hops < len(xy) - 1]}

    cands, n_fit = [], 0
    for j in range(job["tries"]):
        seed = job["seed"] * 1_000_000 + job["slot"] * 10_000 + j
        o, xy = run(seed)
        side = (np.arctan2(xy[:, 1], xy[:, 0]) - a + np.pi) % (2 * np.pi) - np.pi
        if np.abs(side).max() > job["half_wedge"]:
            continue
        if not (fits(xy[::20], ok, centre, s) and fits(xy, ok, centre, s)):
            continue
        n_fit += 1
        travel = float(np.hypot(*(xy - xy[0]).T).max())
        # how much of the nucleus outline the path runs along: 1-degree bins with a point
        # within 0.4 um of the wall
        th = np.arctan2(xy[:, 1], xy[:, 0])
        near = np.hypot(xy[:, 0], xy[:, 1]) - ellipse_radius(*job["wall_um"], th) < 0.4
        hug = len(np.unique(np.floor(np.degrees(th[near]))))
        score = (hug // 5, travel) if job["prefer"] == "hug" else (0, travel)
        cands.append((score, seed, travel, hug, xy[::5].astype(np.float32)))   # 50-ms outline
    cands.sort(key=lambda c: c[0], reverse=True)
    return job["slot"], n_fit, cands[:job["keep"]]


def simulated_cell(cell, ellipse, centre, s, k, window, size, args):
    """A simulated fluorescence image of the model cell at cover resolution. RNPs are
    diffraction-limited green specks at random positions in the cytoplasm (Gaussian spots,
    --psf-um) over a dim diffuse green; the nucleus is the model's ellipse in blue with a
    smooth random texture. Only the cell's outline comes from the photo (its segmented
    boundary, smoothed): no pixel of the micrograph is used."""
    PW, PH = size
    X0, Y0 = window[0], window[1]
    um = s * k                                            # cover pixels per um
    rng = np.random.default_rng(args.seed)
    soft = ndi.gaussian_filter(cell.astype(float), 6)
    soft = Image.fromarray((soft * 255).astype(np.uint8)).resize(
        (round(cell.shape[1] * k), round(cell.shape[0] * k)), Image.BILINEAR)
    placed = Image.new("L", (PW, PH), 0)
    placed.paste(soft, (round(-X0 * k), round(-Y0 * k)))
    inside = np.asarray(placed) > 127
    cyto = ndi.gaussian_filter(inside.astype(float), 0.5 * um)          # soft membrane edge
    x = (np.arange(PW)[None, :] / k + X0 - centre[0]) / s                # model um, y up
    y = -(np.arange(PH)[:, None] / k + Y0 - centre[1]) / s
    a, b, phi = ellipse
    u, v = x * np.cos(phi) + y * np.sin(phi), -x * np.sin(phi) + y * np.cos(phi)
    nuc_hard = (u / a) ** 2 + (v / b) ** 2 <= 1
    nuc = ndi.gaussian_filter(nuc_hard.astype(float), 0.15 * um)
    cyto = cyto * (1 - nuc)

    def smooth_noise(scale_um):
        n = max(4, int(PW / (scale_um * um)))
        lr = rng.standard_normal((int(n * PH / PW) + 3, n + 3))
        z = ndi.zoom(lr, (PH / (lr.shape[0] - 3), PW / (lr.shape[1] - 3)), order=3)[:PH, :PW]
        return z / z.std()

    area_um2 = (cyto > 0.5).sum() / um ** 2
    n = int(args.puncta_per_um2 * area_um2)
    idx = rng.choice(np.flatnonzero((cyto > 0.5).ravel()), size=n, replace=False)
    spots = np.zeros(PH * PW)
    np.add.at(spots, idx, rng.lognormal(0.0, 0.45, n))
    sigma = args.psf_um * um
    spots = ndi.gaussian_filter(spots.reshape(PH, PW), sigma) * 2 * np.pi * sigma ** 2
    green = 0.20 * cyto * (1 + 0.25 * smooth_noise(2.0)) + args.puncta_brightness * spots
    blue = 0.55 * nuc * (1 + 0.18 * smooth_noise(1.0))
    rgb = green[..., None] * np.array([0.15, 1.0, 0.25]) + blue[..., None] * np.array([0.12, 0.25, 1.0])
    print(f"simulated cell: {n} RNP specks over {area_um2:.0f} um^2 of cytoplasm")
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-rnps", type=int, default=8)
    ap.add_argument("--rnp-minutes", type=float, default=20.0)
    ap.add_argument("--p-driv", type=float, default=0.004)
    ap.add_argument("--tries", type=int, default=1000, help="RNP paths simulated per slot")
    ap.add_argument("--min-gap-um", type=float, default=0.8,
                    help="closest any two RNP paths may come, so each stays distinct")
    ap.add_argument("--prefer", choices=("hug", "travel"), default="hug",
                    help="among paths that fit: the one running along the most nucleus wall "
                         "(ties: farthest travelled), or simply the farthest travelled")
    ap.add_argument("--wedge", type=float, default=1.0,
                    help="how far an RNP may wander sideways, as a fraction of half the slot spacing")
    ap.add_argument("--rnp-width-um", type=float, default=0.02)
    ap.add_argument("--trap-fraction", type=float, default=0.10,
                    help="fraction of the lines within a trap drawn (hops, including the RNP "
                         "settling into its new trap, and directed runs always are)")
    ap.add_argument("--rnp-alpha", type=float, default=0.85)
    ap.add_argument("--rnp-color", default="#FFFFFF")
    ap.add_argument("--capsule-color", default="#1E222A", help="bacterium fill")
    ap.add_argument("--capsule-edge", default="#5C6370", help="bacterium outline")
    ap.add_argument("--protein-color", default="#D44DFF",
                    help="protein path, in the cell and the bacterium panels (purple)")
    ap.add_argument("--rotation-deg", type=float, default=0.0)
    ap.add_argument("--px-per-um", type=float, default=None,
                    help="photo scale (default: model nucleus just covers the photo's)")
    ap.add_argument("--geometry", choices=("ellipse", "photo", "model"), default="ellipse",
                    help="the model's nucleus (see above)")
    ap.add_argument("--ellipse-scale", type=float, default=1.06,
                    help="enlarge the fitted ellipse so it covers the nucleus's blurred edge")
    ap.add_argument("--bacteria", action="store_true",
                    help="add goodsell.py's bacterium panels in a black corner")
    ap.add_argument("--protein-seconds", type=float, default=1.0)
    ap.add_argument("--green-threshold", type=float, default=35)
    ap.add_argument("--edge-margin-px", type=int, default=8)
    ap.add_argument("--cover-in", type=float, nargs=2, default=(8.5, 11.0), metavar=("W", "H"),
                    help="cover size in inches (at 300 dpi)")
    ap.add_argument("--masthead-frac", type=float, default=0.26,
                    help="fraction of the cover height kept clear at the top for the masthead")
    ap.add_argument("--source-enlargement", type=float, default=11.2,
                    help="how much the photo was enlarged from its source figure")
    ap.add_argument("--glow", type=float, default=0.6, help="glow strength on the cover paths")
    ap.add_argument("--cover-rnp-alpha", type=float, default=0.68,
                    help="RNP path opacity on the cover (lower = dense spots build up gradually)")
    ap.add_argument("--photo-gamma", type=float, default=0.8,
                    help="brightness curve for the cover photo (<1 lifts the dim green)")
    ap.add_argument("--directed-width-um", type=float, default=0.15)
    ap.add_argument("--background", choices=("photo", "simulated"), default="photo",
                    help="cover background: the micrograph, or a simulated fluorescence image "
                         "of the model cell (RNP specks, diffuse green, textured nucleus)")
    ap.add_argument("--puncta-brightness", type=float, default=0.4,
                    help="simulated background: peak brightness of a typical RNP speck (0-1)")
    ap.add_argument("--puncta-per-um2", type=float, default=1.0,
                    help="simulated background: RNP specks per um^2 of cytoplasm")
    ap.add_argument("--psf-um", type=float, default=0.16,
                    help="simulated background: Gaussian s.d. of each speck (60x, NA 0.7)")
    ap.add_argument("--protein-sample-ms", type=float, default=1.0,
                    help="draw the protein's path every this many ms")
    ap.add_argument("--outdir", type=Path, default=Path("cover_output"))
    args = ap.parse_args()

    random.seed(args.seed)                     # simulation.py's import-time k draw
    sim = importlib.import_module("simulation")
    g = importlib.import_module("goodsell")

    im, cell, nuc, ok, centre, reach = masks(args.photo, args.green_threshold, 6,
                                             args.edge_margin_px, 3)
    s = args.px_per_um or reach / (sim.NUCLEUS_RADIUS * 1e6)
    angles = np.deg2rad(args.rotation_deg) + 2 * np.pi * np.arange(args.n_rnps) / args.n_rnps
    ellipse = None
    if args.geometry == "ellipse":
        ellipse = nucleus_ellipse(nuc, s, args.ellipse_scale)
        R_nuc = ellipse[0]                      # semi-major (for the protein's reflection)
        starts = [ellipse_radius(*ellipse, a) + 2.5 for a in angles]
        desc = (f"ellipse {ellipse[0]:.2f} x {ellipse[1]:.2f} um, major axis at "
                f"{np.degrees(ellipse[2]):.0f} deg")
    else:
        R_nuc = reach / s if args.geometry == "photo" else sim.NUCLEUS_RADIUS * 1e6
        starts = [R_nuc + 2.5 if args.geometry == "photo" else sim.CELL_RADIUS * 1e6 / 2] * len(angles)
        desc = f"circle of radius {R_nuc:.2f} um"
    R_cell = 2 * max(starts)                    # only bounds the protein; RNPs are not stopped
    print(f"photo {im.shape[1]}x{im.shape[0]} px at {s:.1f} px/um; nucleus centre "
          f"({centre[0]:.0f}, {centre[1]:.0f})")
    print(f"model nucleus ({args.geometry}): {desc}; RNPs start "
          f"{min(starts):.2f}-{max(starts):.2f} um from its centre")

    # RNP paths, sampled every 10 ms (as drawn): each slot simulated in its own process
    half = np.pi / args.n_rnps * args.wedge
    key = hashlib.sha1(args.photo.read_bytes() + json.dumps(
        [args.seed, args.tries, args.rnp_minutes, args.p_driv, args.n_rnps, args.rotation_deg,
         args.wedge, s, args.green_threshold, args.edge_margin_px, args.geometry,
         args.ellipse_scale, args.prefer, "hops+settle", "gap", args.min_gap_um]).encode()).hexdigest()[:12]
    cache = args.outdir / f"rnp_paths_{key}.npz"
    results = {}
    if cache.exists():
        z = np.load(cache)
        for i in range(args.n_rnps):
            if f"xy{i}" in z:
                results[i] = (int(z["n_fit"][i]), {"seed": int(z["seed"][i]), "travel": float(z["travel"][i]),
                                                   "hug_deg": int(z["hug_deg"][i]),
                                                   "xy": z[f"xy{i}"], "state": z[f"state{i}"],
                                                   "hops": z[f"hops{i}"]})
            else:
                results[i] = (int(z["n_fit"][i]), None)
        print(f"loaded the chosen RNP paths from {cache}")
    else:
        jobs = [{"slot": i, "angle": float(a), "start_m": st * 1e-6, "seed": args.seed,
                 "ellipse_m": (ellipse[0] * 1e-6, ellipse[1] * 1e-6, ellipse[2]) if ellipse else None,
                 "nucleus_radius_m": R_nuc * 1e-6, "minutes": args.rnp_minutes,
                 "p_driv": args.p_driv, "tries": args.tries, "half_wedge": half,
                 "ok": ok, "centre": centre, "s": s, "prefer": args.prefer,
                 "wall_um": ellipse if ellipse else (R_nuc, R_nuc, 0.0), "keep": 200}
                for i, (a, st) in enumerate(zip(angles, starts))]
        workers = min(len(jobs), os.cpu_count() or 1)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
            found = {slot: (n_fit, cands) for slot, n_fit, cands in ex.map(simulate_slot, jobs)}
        # one path per slot, every pair at least --min-gap-um apart so no two RNPs touch: slots
        # with the fewest candidates choose first, each taking its best candidate that stays clear
        picked, trees = {}, []
        for i in sorted(found, key=lambda i: len(found[i][1])):
            for score, seed, travel, hug, outline in found[i][1]:
                if all(np.isinf(t.query(outline, distance_upper_bound=args.min_gap_um)[0]).all()
                       for t in trees):
                    picked[i] = (seed, travel, hug)
                    trees.append(cKDTree(outline))
                    break
        with ProcessPoolExecutor(min(len(picked), workers) or 1, mp_context=ctx) as ex:
            full = dict(ex.map(simulate_slot, [dict(jobs[i], replay=picked[i][0]) for i in picked]))
        for i in range(args.n_rnps):
            results[i] = (found[i][0], dict(full[i], travel=picked[i][1], hug_deg=picked[i][2])
                          if i in picked else None)
        arrays = {"n_fit": np.array([results[i][0] for i in range(args.n_rnps)]),
                  "seed": np.array([results[i][1]["seed"] if results[i][1] else -1 for i in range(args.n_rnps)]),
                  "travel": np.array([results[i][1]["travel"] if results[i][1] else 0.0 for i in range(args.n_rnps)]),
                  "hug_deg": np.array([results[i][1]["hug_deg"] if results[i][1] else 0 for i in range(args.n_rnps)])}
        for i in range(args.n_rnps):
            if results[i][1]:
                arrays[f"xy{i}"] = results[i][1]["xy"].astype(np.float32)
                arrays[f"state{i}"] = results[i][1]["state"]
                arrays[f"hops{i}"] = results[i][1]["hops"]
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **arrays)
        print(f"simulated {args.tries} RNP paths of {args.rnp_minutes:g} min per slot, "
              f"{workers} slots at a time (k = {sim.k:.3e} N/m)")

    chosen = []
    for i, a in enumerate(angles):
        n_fit, best = results[i]
        if best is None:
            print(f"  slot {i} ({np.degrees(a) % 360:5.1f} deg): no path fits; left empty")
            continue
        chosen.append({"slot": i, "angle_deg": float(np.degrees(a) % 360), "start_um": starts[i],
                       "numba_seed": best["seed"], "fitted": n_fit, "farthest_um": best["travel"],
                       "along_nucleus_deg": best["hug_deg"],
                       "xy": best["xy"], "state": best["state"], "hops": best["hops"]})
        print(f"  slot {i} ({np.degrees(a) % 360:5.1f} deg): {n_fit}/{args.tries} paths fit "
              f"({100 * n_fit / args.tries:.2f}%); using seed {best['seed']}: runs along "
              f"{best['hug_deg']} deg of the nucleus, reaches {best['travel']:.1f} um from its start")

    # protein: 1 s of free diffusion, started at the point deepest inside the cytoplasm
    # along each candidate direction (right side first, as in the mock-up)
    gcfg = g.GoodsellConfig()
    depth = ndi.distance_transform_edt(ok)
    prot = None
    for ang in (15, 35, 55, -10, 75, 300, 120):
        u = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        radii = np.linspace(R_nuc, 25.0, 400)
        p = np.rint(to_px(radii[:, None] * u, centre, s)).astype(int)
        keep = (p[:, 0] >= 0) & (p[:, 0] < ok.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < ok.shape[0])
        r0 = radii[keep][np.argmax(depth[p[keep, 1], p[keep, 0]])]
        start = r0 * u
        for k in range(200):
            pxy, _ = simulate_protein(gcfg.D_protein, args.protein_seconds, start,
                                      args.seed * 1000 + k, R_nuc, R_cell, step_um=0.01)
            if fits(pxy, ok, centre, s):
                prot = {"angle_deg": ang, "start_um": start.tolist(), "seed": args.seed * 1000 + k}
                break
        if prot:
            break
    print("protein: " + (f"started at {prot['angle_deg']} deg, seed {prot['seed']}" if prot
                         else "no 1 s path fitted; omitted"))

    # ---- geometry, in photo pixels (y down), shared by every rendering below
    H, W = ok.shape
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    # RNP lines, one per 10-ms step. Hops between traps (from the trap centre's jump until the
    # RNP has settled in the new trap), directed runs and the joins between them are all drawn;
    # of the lines with the RNP settled in a trap only a random --trap-fraction is, so each trap
    # shows as a "snowflake" of jumps around its centre instead of a solid blob.
    rng = np.random.default_rng(args.seed)
    rnp_diff, rnp_dir = [], []
    for c in chosen:
        p = to_px(c["xy"].astype(float), centre, s)
        st = c["state"].astype(bool)
        rnp_dir += runs_by_state(p[:, 0], p[:, 1], st, 1)[1]
        seg = np.stack([p[:-1], p[1:]], axis=1)
        hop = np.zeros(len(seg), bool)
        hop[c["hops"]] = True
        join = st[:-1] != st[1:]
        keep = ~(st[:-1] & st[1:]) & (hop | join | (rng.random(len(seg)) < args.trap_fraction))
        rnp_diff.append(seg[keep])
    rnp_diff = np.concatenate(rnp_diff)
    prot_px = None
    if prot:
        pxy, pt = simulate_protein(gcfg.D_protein, args.protein_seconds, np.array(prot["start_um"]),
                                   prot["seed"], R_nuc, R_cell, step_um=0.01)
        # drawn every --protein-sample-ms (still one continuous path, just sampled less often)
        idx = np.unique(np.r_[np.searchsorted(pt, np.arange(0, pt[-1], args.protein_sample_ms * 1e-3)),
                              len(pt) - 1])
        prot_px = to_px(pxy[idx], centre, s)

    # bacterium (goodsell.py's Figure 1 run): the 5, 10 and 15 ms panels magnified, and the
    # 15 ms panel at the photo's scale, in whichever black corner holds them largest. Dark
    # capsules keep the cell the focus; paths fade into the capsule colour.
    P, S, t = g.simulate(gcfg, g.SEED)
    rc = replace(g.RenderConfig(), protein_color=args.protein_color, cytoplasm=args.capsule_color)
    a_um = gcfg.display_pixel_nm * 1e-3
    Pxy, Pt = g.display_path(P, t, a_um)
    Sxy, St = g.display_path(S, t, a_um)
    Rb, hb = gcfg.cell_diameter_um / 2, (gcfg.cell_length_um - gcfg.cell_diameter_um) / 2
    keep_clear = ndi.binary_dilation(cell, iterations=int(0.02 * W))
    fs_px = 0.026 * W                       # text height, relative to the photo (not in um)
    unit = 0.04 * W
    label_w = 0.55 * fs_px * len("×1")

    def clear(x0, y0, x1, y1):
        """Does this box (photo px) stay off the cell? Outside the photo counts as clear."""
        xa, ya, xb, yb = max(int(x0), 0), max(int(y0), 0), min(int(np.ceil(x1)), W), min(int(np.ceil(y1)), H)
        return xa >= xb or ya >= yb or not keep_clear[ya:yb, xa:xb].any()

    def bacteria_layout(window, labelled, corners):
        X0, Y0, X1, Y1 = window
        m = 0.035 * (X1 - X0)
        for mag in (6, 5, 4.5, 4, 3.5, 3, 2.5, 2):
            pw, ph = 2 * Rb * mag * s, 2 * (Rb + hb) * mag * s
            if labelled:                    # room for the time labels and the "x1" label
                gap = max(0.15 * pw, 0.55 * fs_px * len("15 ms") - pw + 0.6 * fs_px)
                sgap, small_w, h = max(0.5 * pw, 0.6 * label_w), max(2 * Rb * s, label_w), 2.8 * fs_px + ph
            else:
                gap, sgap, small_w, h = 0.15 * pw, 0.5 * pw, 2 * Rb * s, ph
            w = 3 * pw + 2 * gap + sgap + small_w
            for corner in corners:
                x0 = X1 - m - w if corner.endswith("right") else X0 + m
                y0 = Y1 - m - h if corner.startswith("bottom") else Y0 + m
                if clear(x0, y0, x0 + w, y0 + h):
                    return dict(mag=mag, corner=corner, pw=pw, ph=ph, gap=gap, sgap=sgap,
                                small_w=small_w, x0=x0, y0=y0, top=y0 + (1.4 * fs_px if labelled else 0))
        return None

    def render(window, k, groups, path, L, rnp_alpha=None):
        """Draw `groups` ("paths", "bacteria", "labels") over the photo-pixel window
        (x0, y0, x1, y1) at k output pixels per photo pixel, to a transparent PNG."""
        X0, Y0, X1, Y1 = window
        fig = plt.figure(figsize=((X1 - X0) * k / DPI, (Y1 - Y0) * k / DPI))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(X0, X1)
        ax.set_ylim(Y1, Y0)
        ax.axis("off")
        pt_per_px = k * 72 / DPI
        pt_per_um = s * pt_per_px
        fs = fs_px * pt_per_px
        labelled = "labels" in groups

        def text(*a, **kw):
            ax.text(*a, fontsize=fs, color=TEXT, zorder=6, **kw)

        if "paths" in groups:
            ax.add_collection(LineCollection(rnp_diff, colors=args.rnp_color,
                                             linewidths=args.rnp_width_um * pt_per_um,
                                             alpha=rnp_alpha or args.rnp_alpha, capstyle="round",
                                             joinstyle="round", zorder=2))
            ax.add_collection(LineCollection(rnp_dir, colors=DIRECTED,
                                             linewidths=args.directed_width_um * pt_per_um,
                                             capstyle="round", zorder=3))
            if prot_px is not None:
                ax.add_collection(LineCollection([prot_px], colors=args.protein_color,
                                                 linewidths=0.03 * pt_per_um, capstyle="round",
                                                 joinstyle="round", zorder=3.5))

        def bacterium(cx, cy, px_per_um, t_now):
            outline = g.capsule_xy(cx, cy, Rb * px_per_um, hb * px_per_um)
            body = ax.add_patch(Polygon(outline, closed=True, fc=rc.cytoplasm, ec=args.capsule_edge,
                                        lw=0.0015 * W * pt_per_px, zorder=2))
            runs = (g.tinted_runs(Sxy, St, t_now, g.FADE_T_S, rc.sugar_color, rc, np.zeros(2))
                    + g.tinted_runs(Pxy, Pt, t_now, g.FADE_T_S, rc.protein_color, rc, np.zeros(2)))
            kk = a_um * px_per_um
            lc = ax.add_collection(LineCollection(
                [np.c_[cx + r[0][:, 0] * kk, cy - r[0][:, 1] * kk] for r in runs],
                colors=[r[1] for r in runs], linewidths=rc.stroke_px * kk * pt_per_px, zorder=3,
                capstyle="projecting", joinstyle="miter"))
            lc.set_clip_path(body)

        if "bacteria" in groups and L:
            top = L["top"]
            for i, t_now in enumerate(g.PANEL_TIMES_S):
                cx = L["x0"] + L["pw"] / 2 + i * (L["pw"] + L["gap"])
                bacterium(cx, top + L["ph"] / 2, s * L["mag"], t_now)
                if labelled:
                    text(cx, top + L["ph"] + 0.3 * fs_px, g.label(t_now), ha="center", va="top")
            sx = L["x0"] + 3 * L["pw"] + 2 * L["gap"] + L["sgap"] + L["small_w"] / 2
            bacterium(sx, top + L["ph"] - (Rb + hb) * s, s, g.PANEL_TIMES_S[-1])
            if labelled:
                text(L["x0"], top - 0.3 * fs_px, f"bacterium, ×{L['mag']:g}", ha="left", va="bottom")
                text(sx, top + L["ph"] + 0.3 * fs_px, "×1", ha="center", va="top")

        if labelled:
            # legend (top right) and scale bar (bottom right, or bottom left if the bacteria are)
            x_leg, y_leg = W - 0.30 * W, 0.045 * H
            entries = ((args.rnp_color, 0.14, f"RNPs, {args.rnp_minutes:g} min"),
                       (DIRECTED, 0.12, "directed runs"),
                       (args.protein_color, 0.14, f"protein, {args.protein_seconds:g} s"))
            if L:
                entries += ((rc.sugar_color, 0.14, "sugar (bacterium)"),)
            for i, (colour, lw_um, label) in enumerate(entries):
                y = y_leg + 0.95 * i * unit
                ax.plot([x_leg - 1.2 * unit, x_leg - 0.3 * unit], [y, y], color=colour,
                        lw=max(lw_um * pt_per_um, 0.004 * W * pt_per_px), solid_capstyle="round")
                text(x_leg, y, label, ha="left", va="center")
            xb = 0.06 * W if L and L["corner"] == "bottom-right" else W - 0.06 * W - 5 * s
            yb = H - 0.04 * H
            ax.plot([xb, xb + 5 * s], [yb, yb], color=TEXT, lw=0.004 * W * pt_per_px,
                    solid_capstyle="butt")
            text(xb + 2.5 * s, yb - 0.25 * unit, "5 µm", ha="center", va="bottom")
        fig.savefig(path, dpi=DPI, transparent=True)
        plt.close(fig)

    # ---- overlays for the photo itself (same size as the photo, at UPSCALE)
    args.outdir.mkdir(parents=True, exist_ok=True)
    whole = (0, 0, W, H)
    L = bacteria_layout(whole, True, ("bottom-left", "bottom-right", "top-left")) if args.bacteria else None
    if args.bacteria:
        print("bacterium panels: " + (f"x{L['mag']:g} in the {L['corner']} corner" if L
                                      else "no black corner is big enough; left out"))
    render(whole, UPSCALE, {"paths", "bacteria", "labels"}, args.outdir / "photo_overlay.png", L)
    render(whole, UPSCALE, {"paths", "bacteria"}, args.outdir / "photo_overlay_nolabels.png", L)
    over = Image.open(args.outdir / "photo_overlay.png").convert("RGBA")
    base = Image.fromarray(im).resize(over.size, Image.BICUBIC).convert("RGBA")
    Image.alpha_composite(base, over).convert("RGB").save(args.outdir / "photo_preview.png")

    # ---- the cover: full-bleed, no text, the cell as large as fits below a masthead band
    PW, PH = int(round(args.cover_in[0] * DPI)), int(round(args.cover_in[1] * DPI))
    ys, xs = np.nonzero(cell)
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    band, margin = args.masthead_frac * PH, 0.035 * PW
    k = min((PW - 2 * margin) / (bx1 - bx0), (PH - band - 2 * margin) / (by1 - by0))
    X0 = (bx0 + bx1) / 2 - PW / (2 * k)
    Y0 = (by0 + by1) / 2 - (band + (PH - band) / 2) / k
    window = (X0, Y0, X0 + PW / k, Y0 + PH / k)
    Lc = bacteria_layout(window, False, ("bottom-right", "bottom-left")) if args.bacteria else None
    tmp_paths, tmp_bact = args.outdir / "_cover_paths.png", args.outdir / "_cover_bacteria.png"
    render(window, k, {"paths"}, tmp_paths, None, rnp_alpha=args.cover_rnp_alpha)
    render(window, k, {"bacteria"}, tmp_bact, Lc)

    def photo_page(k):
        # photo: back to the source figure's own resolution, then smoothly up to the cover's
        # (removes the enlargement's blocky pixels without inventing detail)
        src = Image.fromarray(im)
        native = src.resize((round(W / args.source_enlargement), round(H / args.source_enlargement)),
                            Image.BOX)
        smooth = native.resize((round(W * k), round(H * k)), Image.BICUBIC).filter(
            ImageFilter.GaussianBlur(0.35 * args.source_enlargement * k / 2))
        # isolate this cell (fade out neighbours beyond a soft edge) and lift the dim green
        halo = ndi.gaussian_filter(ndi.binary_dilation(cell, iterations=int(0.03 * W)).astype(float),
                                   0.02 * W)
        halo = np.asarray(Image.fromarray((halo * 255).astype(np.uint8)).resize(smooth.size, Image.BILINEAR))
        lifted = (np.asarray(smooth).astype(float) / 255) ** args.photo_gamma * (halo[..., None] / 255)
        page = Image.new("RGB", (PW, PH), "black")
        page.paste(Image.fromarray((lifted * 255).round().astype(np.uint8)), (round(-X0 * k), round(-Y0 * k)))
        return page

    if args.background == "simulated":
        page = simulated_cell(cell, ellipse or (R_nuc, R_nuc, 0.0), centre, s, k, window,
                              (PW, PH), args)
    else:
        page = photo_page(k)
    name = f"cover_{args.background}_final"

    def layer(path):
        a = np.asarray(Image.open(path).convert("RGBA")).astype(float) / 255
        return a[..., :3], a[..., 3:]

    rgb, alpha = layer(tmp_paths)
    glow = sum(gain * ndi.gaussian_filter(rgb * alpha, (sig * PW, sig * PW, 0))
               for sig, gain in ((0.004, args.glow), (0.014, 0.6 * args.glow)))
    out = np.asarray(page).astype(float) / 255
    out = 1 - (1 - out) * (1 - np.clip(glow, 0, 1))          # screen blend: light adds up
    out = out * (1 - alpha) + rgb * alpha                     # crisp paths on top
    b_rgb, b_alpha = layer(tmp_bact)
    out = out * (1 - b_alpha) + b_rgb * b_alpha
    cover = Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))
    cover.save(args.outdir / f"{name}.png", dpi=(DPI, DPI))
    tmp_paths.unlink()
    tmp_bact.unlink()
    preview = cover.copy()
    d = ImageDraw.Draw(preview)
    d.rectangle([0.02 * PW, 0.02 * PH, 0.98 * PW, band - 0.02 * PH], outline=(110, 110, 120), width=4)
    d.text((PW / 2, band / 2), "[ masthead ]", fill=(140, 140, 150), anchor="mm",
           font=ImageFont.load_default(size=int(0.04 * PW)))
    preview.save(args.outdir / f"{name}_preview.png")
    print(f"cover ({args.background} background): {PW}x{PH} px ({args.cover_in[0]:g} x "
          f"{args.cover_in[1]:g} in at {DPI} dpi), cell enlarged {k:.2f}x" + ("" if not args.bacteria
          else "; bacterium panels " + (f"x{Lc['mag']:g} in the {Lc['corner']} corner" if Lc
                                        else "left out (no room)")))

    (args.outdir / "photo_choice.json").write_text(json.dumps(
        {"photo": str(args.photo), "px_per_um": s, "nucleus_centre_px": centre.tolist(),
         "seed": args.seed, "tries_per_slot": args.tries, "rnp_minutes": args.rnp_minutes,
         "p_driv": args.p_driv, "k": sim.k, "geometry": args.geometry, "background": args.background,
         "nucleus": ({"semi_axes_um": ellipse[:2], "major_axis_deg": float(np.degrees(ellipse[2])),
                      "scale": args.ellipse_scale} if ellipse else {"radius_um": R_nuc}),
         "wedge": args.wedge, "protein": prot,
         "rnps": [{k_: v for k_, v in c.items() if k_ not in ("xy", "state", "hops")} for c in chosen]},
        indent=2))
    print(f"wrote {args.outdir}/photo_overlay.png, photo_overlay_nolabels.png, photo_preview.png, "
          f"{name}.png, {name}_preview.png, photo_choice.json")


if __name__ == "__main__":
    main()
