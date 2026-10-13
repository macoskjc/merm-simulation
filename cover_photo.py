"""
Fit the cover-art RNP simulation into a micrograph of a real cell.

The cell photo (nucleus in blue, cytoplasm in green) is segmented, the model's circular
nucleus is centred on the photo's nucleus and scaled so that it just covers it, and each
RNP slot is filled with a simulated 20-min path that stays inside the green cytoplasm.

    python cover_photo.py --photo cover_output/cell_photo.png     # -> cover_output/photo_*

How the paths are chosen (illustration only):
  * The model is rotationally symmetric (circular cell and nucleus, radial directed runs,
    isotropic trap hops), so a path simulated from angle 0 can be rotated to start at any
    angle. A pool of paths is simulated once (seeded) and rotated into each slot.
  * Most 20-min paths leave a cell this size, so only paths that stay inside the photo's
    cytoplasm, and within their own wedge around the nucleus (so clusters do not overlap),
    are kept; of those, each slot gets the one that travels farthest. The script prints
    what fraction of the pool fitted each slot. A small summary of the pool is cached
    in --outdir, so redrawing with the same settings skips the simulation.
  * Because the RNPs do not interact, this is equivalent to resimulating whole 8-RNP
    draws until all eight fit, only much faster.

Outputs (in --outdir):
    photo_overlay.png    transparent, 3x the photo's pixel size: paths, legend, scale bar
    photo_preview.png    the photo with the overlay on top
    photo_choice.json    scale, seeds and pool indices, so the choice can be reproduced
"""

import os
import tempfile

os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="merm_cover_numba_")

import argparse
import importlib
import hashlib
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from PIL import Image
from scipy import ndimage as ndi

from cover_art import DIRECTED, PROTEIN, RNP, TEXT, runs_by_state, seed_numba, simulate_protein

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
    return im.astype(np.uint8), ok, centre, reach


def to_px(xy_um, centre, s):
    return np.c_[centre[0] + xy_um[:, 0] * s, centre[1] - xy_um[:, 1] * s]


def fits(xy_um, ok, centre, s):
    p = np.rint(to_px(xy_um, centre, s)).astype(int)
    H, W = ok.shape
    inside = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    return bool(inside.all() and ok[p[:, 1], p[:, 0]].all())


def rotate(xy, a):
    c, s = np.cos(a), np.sin(a)
    return np.c_[c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photo", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n-rnps", type=int, default=8)
    ap.add_argument("--rnp-minutes", type=float, default=20.0)
    ap.add_argument("--p-driv", type=float, default=0.004)
    ap.add_argument("--pool", type=int, default=8000, help="RNP paths simulated to choose from")
    ap.add_argument("--wedge", type=float, default=1.0,
                    help="how far an RNP may wander sideways, as a fraction of half the slot spacing")
    ap.add_argument("--rnp-width-um", type=float, default=0.025)
    ap.add_argument("--rnp-alpha", type=float, default=0.9)
    ap.add_argument("--rotation-deg", type=float, default=0.0)
    ap.add_argument("--px-per-um", type=float, default=None,
                    help="photo scale (default: model nucleus just covers the photo's)")
    ap.add_argument("--protein-seconds", type=float, default=1.0)
    ap.add_argument("--green-threshold", type=float, default=35)
    ap.add_argument("--edge-margin-px", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=Path("cover_output"))
    args = ap.parse_args()

    random.seed(args.seed)                     # simulation.py's import-time k draw
    sim = importlib.import_module("simulation")
    g = importlib.import_module("goodsell")
    R_nuc, R_cell = sim.NUCLEUS_RADIUS * 1e6, sim.CELL_RADIUS * 1e6

    im, ok, centre, reach = masks(args.photo, args.green_threshold, 6, args.edge_margin_px, 3)
    s = args.px_per_um or reach / R_nuc
    print(f"photo {im.shape[1]}x{im.shape[0]} px; nucleus centre ({centre[0]:.0f}, {centre[1]:.0f}), "
          f"farthest nucleus pixel {reach:.0f} px -> {s:.1f} px/um")

    # Pool of RNP paths, all started at angle 0 and sampled every 10 ms (as drawn). Paths are
    # simulated one at a time and only a summary is kept: whether each fits each slot, stays
    # in its own wedge (start angle +/- half the slot spacing x --wedge, so clusters do not
    # pile up), and how far it travels. The chosen paths are re-simulated from their seeds.
    cfg = sim.SimulationConfig(total_time=int(args.rnp_minutes * 60), p_driv=args.p_driv)
    every = int(round(0.010 / cfg.dt))

    def rnp_path(j):
        seed_numba(args.seed * 1_000_000 + j)
        o = sim.move(cfg, theta=0.0, stop_on_cell_exit=True)
        return np.c_[o.x[::every], o.y[::every]] * 1e6, o.state_dt[::every].astype(bool)

    angles = np.deg2rad(args.rotation_deg) + 2 * np.pi * np.arange(args.n_rnps) / args.n_rnps
    key = hashlib.sha1(args.photo.read_bytes() + json.dumps(
        [args.seed, args.pool, args.rnp_minutes, args.p_driv, args.n_rnps, args.rotation_deg,
         args.wedge, s, args.green_threshold, args.edge_margin_px]).encode()).hexdigest()[:12]
    cache = args.outdir / f"pool_summary_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        fit, travel = z["fit"], z["travel"]
        print(f"loaded the pool summary from {cache}")
    else:
        half = np.pi / args.n_rnps * args.wedge
        fit = np.zeros((args.pool, args.n_rnps), bool)
        travel = np.zeros(args.pool)
        for j in range(args.pool):
            xy, _ = rnp_path(j)
            travel[j] = np.hypot(*(xy - xy[0]).T).max()
            if np.abs(np.arctan2(xy[:, 1], xy[:, 0])).max() > half:
                continue
            for i, a in enumerate(angles):
                fit[j, i] = (fits(rotate(xy[::20], a), ok, centre, s)
                             and fits(rotate(xy, a), ok, centre, s))
        args.outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fit=fit, travel=travel)
        print(f"simulated {args.pool} RNP paths of {args.rnp_minutes:g} min (k = {sim.k:.3e} N/m)")

    chosen, used = [], set()
    for i, a in enumerate(angles):
        cands = [j for j in np.flatnonzero(fit[:, i]) if j not in used]
        if not cands:
            print(f"  slot {i} ({np.degrees(a) % 360:5.1f} deg): no path fits; left empty")
            continue
        j = int(max(cands, key=lambda j: travel[j]))
        used.add(j)
        chosen.append({"slot": i, "angle_deg": float(np.degrees(a) % 360), "pool_index": j,
                       "numba_seed": args.seed * 1_000_000 + j, "farthest_um": float(travel[j])})
        print(f"  slot {i} ({np.degrees(a) % 360:5.1f} deg): {len(cands)}/{args.pool} paths fit "
              f"({100 * len(cands) / args.pool:.2f}%); using #{j}, which reaches "
              f"{travel[j]:.1f} um from its start")

    # protein: 1 s of free diffusion, started at the point deepest inside the cytoplasm
    # along each candidate direction (right side first, as in the mock-up)
    gcfg = g.GoodsellConfig()
    depth = ndi.distance_transform_edt(ok)
    prot = None
    for ang in (15, 35, 55, -10, 75, 300, 120):
        u = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        radii = np.linspace(R_nuc, R_cell, 400)
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

    # draw at UPSCALE x the photo's pixels, in photo-pixel data coordinates
    H, W = ok.shape
    fig = plt.figure(figsize=(W * UPSCALE / DPI, H * UPSCALE / DPI))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    pt_per_um = s * UPSCALE * 72 / DPI
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    diff_all, dir_all = [], []
    for c in chosen:
        a = np.deg2rad(c["angle_deg"])
        xy, state = rnp_path(c["pool_index"])
        p = to_px(rotate(xy, a), centre, s)
        d, r = runs_by_state(p[:, 0], p[:, 1], state, 1)
        diff_all += d
        dir_all += r
    ax.add_collection(LineCollection(diff_all, colors=RNP, linewidths=args.rnp_width_um * pt_per_um,
                                     alpha=args.rnp_alpha, capstyle="round", joinstyle="round",
                                     zorder=2))
    ax.add_collection(LineCollection(dir_all, colors=DIRECTED, linewidths=0.12 * pt_per_um,
                                     capstyle="round", zorder=3))
    if prot:
        pxy, _ = simulate_protein(gcfg.D_protein, args.protein_seconds, np.array(prot["start_um"]),
                                  prot["seed"], R_nuc, R_cell, step_um=0.01)
        pp = to_px(pxy, centre, s)
        ax.add_collection(LineCollection([pp], colors=PROTEIN, linewidths=0.03 * pt_per_um,
                                         capstyle="round", joinstyle="round", zorder=3.5))
        low = pp[np.argmax(pp[:, 1])]
        fs = 0.7 * pt_per_um
        ax.text(low[0], low[1] + 0.4 * s, f"protein, {args.protein_seconds:g} s", ha="center",
                va="top", fontsize=fs, color=TEXT, zorder=6)
    fs = 0.7 * pt_per_um
    x_leg, y_leg = W - 0.30 * W, 0.06 * H
    for dy, colour, lw, text in ((0, RNP, 0.14, f"RNPs, {args.rnp_minutes:g} min"),
                                 (1.05, DIRECTED, 0.12, "directed runs")):
        y = y_leg + dy * s
        ax.plot([x_leg - 1.5 * s, x_leg - 0.4 * s], [y, y], color=colour, lw=lw * pt_per_um,
                solid_capstyle="round")
        ax.text(x_leg, y, text, ha="left", va="center", fontsize=fs, color=TEXT)
    xb, yb = W - 0.06 * W - 5 * s, H - 0.04 * H
    ax.plot([xb, xb + 5 * s], [yb, yb], color=TEXT, lw=0.08 * pt_per_um, solid_capstyle="butt")
    ax.text(xb + 2.5 * s, yb - 0.35 * s, "5 µm", ha="center", va="bottom", fontsize=fs, color=TEXT)

    args.outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.outdir / "photo_overlay.png", dpi=DPI, transparent=True)
    plt.close(fig)
    over = Image.open(args.outdir / "photo_overlay.png").convert("RGBA")
    base = Image.fromarray(im).resize(over.size, Image.BICUBIC).convert("RGBA")
    Image.alpha_composite(base, over).convert("RGB").save(args.outdir / "photo_preview.png")
    (args.outdir / "photo_choice.json").write_text(json.dumps(
        {"photo": str(args.photo), "px_per_um": s, "nucleus_centre_px": centre.tolist(),
         "seed": args.seed, "pool": args.pool, "rnp_minutes": args.rnp_minutes,
         "p_driv": args.p_driv, "k": sim.k, "rnps": chosen, "protein": prot}, indent=2))
    print(f"wrote {args.outdir}/photo_overlay.png ({over.size[0]}x{over.size[1]}), "
          "photo_preview.png, photo_choice.json")


if __name__ == "__main__":
    main()
