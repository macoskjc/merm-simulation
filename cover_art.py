"""
Cover-art foreground for the MERM Perspective (layout after Ludovic Autin's mock-up).

Middle: RNPs from simulation.py (Song Kim's model, unmodified) followed for 20 min,
        with directed runs highlighted, plus a GFP-sized protein diffusing freely for
        1 s (same protein and Stokes-Einstein settings as goodsell.py's Figure 1).
Lower left: goodsell.py's bacterium (Ludovic Autin), seed 29: Figure 1's 5, 10 and 15 ms
        panels, magnified x4, and the 15 ms panel at the same scale as the eukaryotic cell.
All text and the scale bar are white, for laying over a photo.

Everything is seeded, so the same command redraws the same picture:

    python cover_art.py                  # -> cover_output/  (~15 s)
    python cover_art.py --seed 3         # a different (but again reproducible) draw

Directed motion uses the paper's p_driv = 0.004 (0.4% of the time; see sim-paper.pdf,
Appendix; also run.py's default). With 0.004 the simulation reproduces the
Perspective's 16.4% directed / 83.6% motor-enhanced split.

Seed choice (illustration only, like goodsell.py's seed 29): at p_driv = 0.004, 59% of
RNPs started 7.5 um out reach the membrane within 20 min, with ~5.8 directed runs each
(64-RNP ensemble). Of seeds 1-24, seed 17 is closest to both: 5 of 8 exit, 5.9 runs/RNP.

Outputs (all 8.5 x 11 in, 300 dpi, pixel-aligned so they stack):
    cover_mockup.png / .pdf      full composition with labels and a masthead placeholder
                                 on a dark page standing in for the background photo
    cover_foreground.png         transparent background, with labels; the cell is an
                                 opaque off-white disk
    cover_foreground_no_cell.png the same without the cell disk, spokes and nucleus,
                                 so a background photo shows through everywhere
                                 (layers/ has every part separately, e.g. to drop labels)
    layers/*.png                 the same foreground split into transparent layers

Reproducibility notes (simulation.py is not modified):
  * simulation.py never seeds numba's random generator; each RNP here is seeded with
    seed * 1000 + i before it runs.
  * simulation.py draws its trap stiffness k with Python's `random` at import, and numba
    freezes that global into its on-disk cache. This script seeds that draw and compiles
    into a fresh cache directory, so k is set by --seed, not by an old cache.
"""

import os
import tempfile

# Must happen before numba is imported (see module docstring).
os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="merm_cover_numba_")

import argparse
import importlib
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle
from numba import njit

PAGE_IN = (8.5, 11.0)            # US letter, portrait
DPI = 300
UM_PER_IN = 15.02 / (0.396 * 8.5)  # cell radius = 39.6% of page width, as in the mock-up

PAGE = "#2B2E35"                 # review mock-up only: a dark stand-in for the photo
CYTOPLASM = "#FAFAFA"
NUCLEUS = "#DCD8E6"
SPOKE = "#D2E6DA"
RNP = "#3A9E7A"
DIRECTED = "#EBA024"
PROTEIN = "#2A36B8"
TEXT = "#FFFFFF"
CONNECTOR = "#B8BCC4"

LAYERS = ("cell", "rnp_paths", "directed_runs", "protein", "bacterium", "scalebar", "labels")


@njit(cache=False)
def seed_numba(s):
    np.random.seed(s)


def um_to_pt(um):
    return um / UM_PER_IN * 72


# ------------------------------------------------------------------ simulations

def simulate_rnps(sim, n, minutes, seed, rotation, p_driv):
    cfg = sim.SimulationConfig(total_time=int(minutes * 60), n_particles=n, p_driv=p_driv)
    tracks = []
    for i in range(n):
        seed_numba(seed * 1000 + i)
        out = sim.move(cfg, theta=rotation + 2 * np.pi * i / n, stop_on_cell_exit=True)
        tracks.append((out.x * 1e6, out.y * 1e6, out.state_dt.astype(bool), out.exit_time))
    return cfg, tracks


def simulate_protein(D, seconds, start, seed, r_nuc, r_cell, step_um):
    """Free 2D Brownian motion (the x-y projection of 3D motion), reflected at the
    nucleus and the membrane. step_um sets the r.m.s. step per axis."""
    dt = step_um ** 2 / (2 * D)
    n = int(np.ceil(seconds / dt))
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, step_um, size=(n, 2))
    xy = np.empty((n + 1, 2))
    xy[0] = start
    for i in range(n):
        p = xy[i] + steps[i]
        r = np.hypot(*p)
        if r < r_nuc:
            p *= (2 * r_nuc - r) / r
        elif r > r_cell:
            p *= (2 * r_cell - r) / r
        xy[i + 1] = p
    return xy, np.arange(n + 1) * dt


# ------------------------------------------------------------------ drawing

def runs_by_state(x, y, state, every):
    """Split a track (decimated by `every`) into diffusive and directed polylines."""
    x, y, s = x[::every], y[::every], state[::every]
    breaks = np.flatnonzero(np.diff(s.astype(np.int8))) + 1
    diffusive, directed = [], []
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(s)]):
        seg = np.c_[x[a:b + 1], y[a:b + 1]]
        if len(seg) > 1:
            (directed if s[a] else diffusive).append(seg)
    return diffusive, directed


def draw(sim, g, tracks, protein, bact, args):
    W, H = PAGE_IN[0] * UM_PER_IN, PAGE_IN[1] * UM_PER_IN     # page in um
    R_cell, R_nuc = sim.CELL_RADIUS * 1e6, sim.NUCLEUS_RADIUS * 1e6
    c = np.array([0.557 * W, H - 0.537 * H])                   # cell centre (mock-up)

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=PAGE_IN)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    layer = {k: [] for k in LAYERS + ("page",)}

    def add(name, artist):
        layer[name].append(artist)
        return artist

    fs = um_to_pt(0.70)                                        # label font size

    # page + masthead placeholder (review mock-up only)
    add("page", ax.add_patch(Rectangle((0, 0), W, H, fc=PAGE, lw=0, zorder=0)))
    add("page", ax.add_patch(Rectangle((0.003 * W, H - 0.185 * H), 0.994 * W, 0.18 * H,
                                       fc="none", ec="#6C7078", lw=0.6, ls=(0, (3, 3)), zorder=0)))
    add("page", ax.text(W / 2, H - 0.095 * H, "[ journal masthead ]", ha="center", va="center",
                        fontsize=1.4 * fs, color="#8A8E96"))

    # cell, radial spokes (microtubule cue), nucleus
    cell = add("cell", ax.add_patch(Circle(c, R_cell, fc=CYTOPLASM, lw=0, zorder=1)))
    ang = np.linspace(0, 2 * np.pi, args.spokes, endpoint=False) + np.pi / args.spokes
    spokes = [np.array([c + R_nuc * np.r_[np.cos(a), np.sin(a)],
                        c + R_cell * np.r_[np.cos(a), np.sin(a)]]) for a in ang]
    add("cell", ax.add_collection(LineCollection(spokes, colors=SPOKE, linewidths=um_to_pt(0.025),
                                                 zorder=1.5)))
    add("cell", ax.add_patch(Circle(c, R_nuc, fc=NUCLEUS, lw=0, zorder=4)))

    # RNPs: diffusive (motor-enhanced random) motion in green, directed runs in orange
    every = max(1, int(round(args.rnp_sample_ms * 1e-3 / sim.SimulationConfig().dt)))
    diff_all, dir_all = [], []
    for x, y, state, _ in tracks:
        d, r = runs_by_state(x + c[0], y + c[1], state, every)
        diff_all += d
        dir_all += r
    lc = add("rnp_paths", ax.add_collection(LineCollection(
        diff_all, colors=RNP, linewidths=um_to_pt(args.rnp_width_um), zorder=2,
        capstyle="round", joinstyle="round")))
    lc.set_clip_path(cell)
    lc = add("directed_runs", ax.add_collection(LineCollection(
        dir_all, colors=DIRECTED, linewidths=um_to_pt(args.directed_width_um), zorder=3,
        capstyle="round")))
    lc.set_clip_path(cell)

    # protein: its full 1 s path
    rc = g.RenderConfig()
    pxy, _ = protein
    lc = add("protein", ax.add_collection(LineCollection(
        [pxy + c], colors=PROTEIN, linewidths=um_to_pt(args.protein_width_um), zorder=3.5,
        capstyle="round", joinstyle="round")))
    lc.set_clip_path(cell)

    # bacterium from goodsell.py: Figure 1's 5, 10 and 15 ms panels, magnified, plus the
    # 15 ms panel at the same scale as the eukaryotic cell
    gcfg, P, S, t = bact
    a = gcfg.display_pixel_nm * 1e-3                            # um per goodsell display pixel
    Pxy, Pt = g.display_path(P, t, a)
    Sxy, St = g.display_path(S, t, a)
    Rb, hb = gcfg.cell_diameter_um / 2, (gcfg.cell_length_um - gcfg.cell_diameter_um) / 2
    mag = args.bacterium_mag
    gap = 0.15 * 2 * Rb * mag
    y_bot = 1.8
    centres = [np.array([1.0 + Rb * mag + i * (2 * Rb * mag + gap), y_bot + (Rb + hb) * mag])
               for i in range(len(g.PANEL_TIMES_S))]
    small_c = np.array([centres[-1][0] + Rb * mag + 2.4, y_bot + Rb + hb])

    def bacterium(centre, scale, t_now):
        outline = g.capsule_xy(centre[0], centre[1], Rb * scale, hb * scale)
        clearance = np.hypot(*(outline - c).T).min() - R_cell
        if clearance < 0.2:
            raise SystemExit(f"bacterium panel at x = {centre[0]:.1f} um overlaps the cell; "
                             "use a smaller --bacterium-mag")
        body = add("bacterium", ax.add_patch(Polygon(outline, closed=True, fc=CYTOPLASM,
                                                     lw=0, zorder=2)))
        runs = (g.tinted_runs(Sxy, St, t_now, g.FADE_T_S, rc.sugar_color, rc, np.zeros(2))
                + g.tinted_runs(Pxy, Pt, t_now, g.FADE_T_S, rc.protein_color, rc, np.zeros(2)))
        lc = add("bacterium", ax.add_collection(LineCollection(
            [r[0] * a * scale + centre for r in runs], colors=[r[1] for r in runs],
            linewidths=um_to_pt(rc.stroke_px * a * scale), zorder=3,
            capstyle="projecting", joinstyle="miter")))
        lc.set_clip_path(body)

    for centre, t_now in zip(centres, g.PANEL_TIMES_S):
        bacterium(centre, mag, t_now)
    bacterium(small_c, 1.0, g.PANEL_TIMES_S[-1])
    add("bacterium", ax.plot([centres[-1][0] + 1.06 * Rb * mag, small_c[0] - 1.6 * Rb],
                             [small_c[1], small_c[1]], color=CONNECTOR,
                             lw=um_to_pt(0.03), zorder=1.8)[0])

    # scale bar
    x0, yb = 0.771 * W, 0.023 * H
    add("scalebar", ax.plot([x0, x0 + 5], [yb, yb], color=TEXT, lw=um_to_pt(0.08),
                            solid_capstyle="butt", zorder=5)[0])
    add("scalebar", ax.text(x0 + 2.5, yb + 0.35, "5 µm", ha="center", va="bottom",
                            fontsize=fs, color=TEXT))

    # labels (all white; the legend keeps its colour key as short line swatches)
    lab = lambda *p, **k: add("labels", ax.text(*p, fontsize=fs, zorder=6, color=TEXT, **k))
    add("cell", ax.text(*c, "nucleus", ha="center", va="center", fontsize=fs, color=TEXT,
                        zorder=6))                           # only shown with the cell
    x_leg, y_leg = c[0] + 0.43 * R_cell, c[1] + 0.98 * R_cell
    for dy, colour, lw, text in ((0.0, RNP, args.rnp_width_um * 4, f"RNPs, {args.rnp_minutes:g} min"),
                                 (-1.05, DIRECTED, args.directed_width_um, "directed runs")):
        add("labels", ax.plot([x_leg - 1.5, x_leg - 0.4], [y_leg + dy, y_leg + dy], color=colour,
                              lw=um_to_pt(lw), solid_capstyle="round", zorder=6)[0])
        lab(x_leg, y_leg + dy, text, ha="left", va="center")
    end = pxy[np.argmin(pxy[:, 1])] + c
    lab(end[0], end[1] - 0.6, f"protein, {args.protein_seconds:g} s", ha="center", va="top")
    for centre, t_now in zip(centres, g.PANEL_TIMES_S):
        lab(centre[0], y_bot - 0.3, g.label(t_now), ha="center", va="top")
    lab(centres[0][0], centres[0][1] + (Rb + hb) * mag + 0.35, f"×{mag:g}", ha="center",
        va="bottom")
    lab(small_c[0] + 1.0, small_c[1], f"bacterium, {g.label(g.PANEL_TIMES_S[-1])}, same scale",
        ha="left", va="center")
    return fig, layer


def save(fig, layer, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "layers").mkdir(exist_ok=True)

    def only(names):
        for k, arts in layer.items():
            for art in arts:
                art.set_visible(k in names)

    only(set(layer))
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"cover_mockup.{ext}", dpi=DPI,
                    metadata={"Date": None} if ext == "pdf" else None)
    fg = set(LAYERS)
    only(fg)
    fig.savefig(outdir / "cover_foreground.png", dpi=DPI, transparent=True)
    only(fg - {"cell"})
    fig.savefig(outdir / "cover_foreground_no_cell.png", dpi=DPI, transparent=True)
    for name in LAYERS:
        only({name})
        fig.savefig(outdir / "layers" / f"{name}.png", dpi=DPI, transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=17, help="master seed (RNPs, k, protein)")
    ap.add_argument("--p-driv", type=float, default=0.004,
                    help="fraction of time in directed motion (paper: 0.004)")
    ap.add_argument("--n-rnps", type=int, default=8)
    ap.add_argument("--rnp-minutes", type=float, default=20.0)
    ap.add_argument("--rotation-deg", type=float, default=0.0,
                    help="rotate the RNP start positions (evenly spaced, 7.5 um out)")
    ap.add_argument("--protein-seconds", type=float, default=1.0)
    ap.add_argument("--protein-start", type=float, nargs=2, default=(9.8, -1.2),
                    metavar=("X_UM", "Y_UM"), help="relative to the cell centre")
    ap.add_argument("--bacterium-seed", type=int, default=None,
                    help="goodsell.py seed (default: goodsell.SEED = 29, as in Figure 1)")
    ap.add_argument("--bacterium-mag", type=float, default=4.0,
                    help="magnification of the three bacterium panels (5 overlaps the cell)")
    ap.add_argument("--spokes", type=int, default=40)
    ap.add_argument("--rnp-sample-ms", type=float, default=10.0)
    ap.add_argument("--rnp-width-um", type=float, default=0.035)
    ap.add_argument("--directed-width-um", type=float, default=0.12)
    ap.add_argument("--protein-width-um", type=float, default=0.03)
    ap.add_argument("--outdir", type=Path, default=Path("cover_output"))
    args = ap.parse_args()

    random.seed(args.seed)                     # fixes simulation.py's k draw at import
    sim = importlib.import_module("simulation")
    g = importlib.import_module("goodsell")
    print(f"trap stiffness k = {sim.k:.4e} N/m (simulation.py range 1.5e-6 to 2.6e-6)")

    _, tracks = simulate_rnps(sim, args.n_rnps, args.rnp_minutes, args.seed,
                              np.deg2rad(args.rotation_deg), args.p_driv)
    for i, (x, y, s, ex) in enumerate(tracks):
        print(f"  RNP {i}: {len(x) - 1} steps, directed {100 * s.mean():.1f}% of the time, "
              + (f"left the cell at {ex:.0f} s" if ex >= 0 else "stayed in the cell"))

    gcfg = g.GoodsellConfig()
    protein = simulate_protein(gcfg.D_protein, args.protein_seconds, np.array(args.protein_start),
                               args.seed, sim.NUCLEUS_RADIUS * 1e6, sim.CELL_RADIUS * 1e6,
                               step_um=0.01)
    print(f"protein: D = {gcfg.D_protein:.2f} um^2/s, {len(protein[0]) - 1} steps over "
          f"{args.protein_seconds:g} s")

    bseed = g.SEED if args.bacterium_seed is None else args.bacterium_seed
    P, S, t = g.simulate(gcfg, bseed)
    print(f"bacterium: goodsell seed {bseed}, first contact at {t[-1]:.4f} s")

    fig, layer = draw(sim, g, tracks, protein, (gcfg, P, S, t), args)
    save(fig, layer, args.outdir)
    print(f"wrote {args.outdir}/cover_mockup.png/.pdf, cover_foreground.png, layers/*.png")


if __name__ == "__main__":
    main()
