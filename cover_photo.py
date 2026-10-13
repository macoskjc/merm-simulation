"""
Fit the cover-art RNP simulation into a micrograph of a real cell.

The cell photo (nucleus in blue, cytoplasm in green) is segmented, the model's circular
nucleus is centred on the photo's nucleus, and each RNP slot is filled with a simulated
path that stays inside the green cytoplasm.

    python cover_photo.py --photo cover_output/cell_photo.png --px-per-um 26.2

26.2 px/um is the cover photo's scale: it is the middle cell of Fig. 1F in Moran et al.
2022 (J Virol, doi:10.1128/jvi.00934-22; A549 cells, VSV-PeGFP green, DAPI blue), and the
crop is 11.2x the figure, whose 25 um bar is 58.4 px (2.34 px/um). With --geometry photo
(default) the model's nucleus is resized to cover this cell's (8.5 um) and RNPs start
2.5 um outside it, as in the model (5 -> 7.5 um); the RNP dynamics are unchanged.

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
    photo_overlay.png    transparent, 3x the photo's pixel size: RNP paths (white) with
                         directed runs (orange), the protein (purple), the bacterium panels
                         in a black corner, legend and scale bar
    photo_overlay_nolabels.png   the same without any text, legend or scale bar
    cover_photo_final.png        the cover: 8.5 x 11 in at 300 dpi, full-bleed black, no text,
                                 the cell as large as fits below a masthead band (top 26%),
                                 the photo smoothed back from its blocky enlargement, isolated
                                 from neighbouring cells and lifted, and the paths glowing
    cover_photo_final_preview.png  the same with the masthead area outlined
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
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

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
    return im.astype(np.uint8), cell, ok, centre, reach


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
    ap.add_argument("--rnp-width-um", type=float, default=0.02)
    ap.add_argument("--rnp-alpha", type=float, default=0.85)
    ap.add_argument("--rnp-color", default="#FFFFFF")
    ap.add_argument("--capsule-color", default="#1E222A", help="bacterium fill")
    ap.add_argument("--capsule-edge", default="#5C6370", help="bacterium outline")
    ap.add_argument("--protein-color", default="#D44DFF",
                    help="protein path, in the cell and the bacterium panels (purple)")
    ap.add_argument("--rotation-deg", type=float, default=0.0)
    ap.add_argument("--px-per-um", type=float, default=None,
                    help="photo scale (default: model nucleus just covers the photo's)")
    ap.add_argument("--geometry", choices=("photo", "model"), default="photo",
                    help="photo: model nucleus resized to just cover the photo's, RNPs start "
                         "2.5 um outside it (as in the model: 5 -> 7.5 um); model: 5 um nucleus, "
                         "15 um cell, unchanged")
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
    ap.add_argument("--outdir", type=Path, default=Path("cover_output"))
    args = ap.parse_args()

    random.seed(args.seed)                     # simulation.py's import-time k draw
    sim = importlib.import_module("simulation")
    g = importlib.import_module("goodsell")

    im, cell, ok, centre, reach = masks(args.photo, args.green_threshold, 6, args.edge_margin_px, 3)
    s = args.px_per_um or reach / (sim.NUCLEUS_RADIUS * 1e6)
    if args.geometry == "photo":
        # simulation.py's numba functions read these globals when they compile (first call,
        # into this run's fresh cache); _move starts RNPs at CELL_RADIUS / 2
        sim.NUCLEUS_RADIUS = reach / s * 1e-6
        sim.CELL_RADIUS = 2 * (sim.NUCLEUS_RADIUS + 2.5e-6)
    R_nuc, R_cell = sim.NUCLEUS_RADIUS * 1e6, sim.CELL_RADIUS * 1e6
    print(f"photo {im.shape[1]}x{im.shape[0]} px at {s:.1f} px/um; nucleus centre "
          f"({centre[0]:.0f}, {centre[1]:.0f}), reaching {reach:.0f} px = {reach / s:.2f} um")
    print(f"model geometry ({args.geometry}): nucleus radius {R_nuc:.2f} um, RNPs start at "
          f"{R_cell / 2:.2f} um, cell radius {R_cell:.2f} um")

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

    # ---- geometry, in photo pixels (y down), shared by every rendering below
    H, W = ok.shape
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    rnp_diff, rnp_dir = [], []
    for c in chosen:
        xy, state = rnp_path(c["pool_index"])
        p = to_px(rotate(xy, np.deg2rad(c["angle_deg"])), centre, s)
        d, r = runs_by_state(p[:, 0], p[:, 1], state, 1)
        rnp_diff += d
        rnp_dir += r
    prot_px = None
    if prot:
        pxy, _ = simulate_protein(gcfg.D_protein, args.protein_seconds, np.array(prot["start_um"]),
                                  prot["seed"], R_nuc, R_cell, step_um=0.01)
        prot_px = to_px(pxy, centre, s)

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
                       (args.protein_color, 0.14, f"protein, {args.protein_seconds:g} s"),
                       (rc.sugar_color, 0.14, "sugar"))
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
    L = bacteria_layout(whole, True, ("bottom-left", "bottom-right", "top-left"))
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
    Lc = bacteria_layout(window, False, ("bottom-right", "bottom-left"))
    tmp_paths, tmp_bact = args.outdir / "_cover_paths.png", args.outdir / "_cover_bacteria.png"
    render(window, k, {"paths"}, tmp_paths, None, rnp_alpha=args.cover_rnp_alpha)
    render(window, k, {"bacteria"}, tmp_bact, Lc)

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
    cover.save(args.outdir / "cover_photo_final.png", dpi=(DPI, DPI))
    tmp_paths.unlink()
    tmp_bact.unlink()
    preview = cover.copy()
    d = ImageDraw.Draw(preview)
    d.rectangle([0.02 * PW, 0.02 * PH, 0.98 * PW, band - 0.02 * PH], outline=(110, 110, 120), width=4)
    d.text((PW / 2, band / 2), "[ masthead ]", fill=(140, 140, 150), anchor="mm",
           font=ImageFont.load_default(size=int(0.04 * PW)))
    preview.save(args.outdir / "cover_photo_final_preview.png")
    print(f"cover: {PW}x{PH} px ({args.cover_in[0]:g} x {args.cover_in[1]:g} in at {DPI} dpi), "
          f"photo enlarged {k:.2f}x; bacterium panels "
          + (f"x{Lc['mag']:g} in the {Lc['corner']} corner" if Lc else "left out (no room)"))

    (args.outdir / "photo_choice.json").write_text(json.dumps(
        {"photo": str(args.photo), "px_per_um": s, "nucleus_centre_px": centre.tolist(),
         "seed": args.seed, "pool": args.pool, "rnp_minutes": args.rnp_minutes,
         "p_driv": args.p_driv, "k": sim.k, "geometry": args.geometry,
         "nucleus_radius_um": R_nuc, "start_radius_um": R_cell / 2, "cell_radius_um": R_cell,
         "wedge": args.wedge, "rnps": chosen, "protein": prot}, indent=2))
    print(f"wrote {args.outdir}/photo_overlay.png, photo_overlay_nolabels.png, photo_preview.png, "
          "cover_photo_final.png, cover_photo_final_preview.png, photo_choice.json")


if __name__ == "__main__":
    main()
