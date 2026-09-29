"""Brownian motion of a sugar and a protein in a bacterium (Figure 1, after D. S. Goodsell).

    python goodsell.py                      # seed 29 -> goodsell_output/goodsell_fig1.png/.svg/.pdf
    python goodsell.py --no-labels          # same figure without text
    python goodsell.py --find-seed          # redo the seed choice (64 runs, a few minutes)

A sugar (glucose, red) and a typical protein (GFP-sized, blue) start at opposite poles
of a bacterium and diffuse until they first touch. The figure shows the paths after
5, 10 and 15 ms and, after a break in time, the complete paths up to the first contact.

Physics (from the Figure 1 caption):
  * cell: capsule 1 um wide, 2 um long; viscosity 20 mPa.s (~30x water); 37 C
  * Stokes radii: protein 2.5 nm (GFP, from D0 = 87 um^2/s in water, Swaminathan
    et al. 1997), glucose 0.36 nm (Ribeiro et al. 2006)
  * D from Stokes-Einstein, D = kT / (6 pi eta r): protein 4.5, sugar 31.6 um^2/s

Simulation: off-lattice 3D Brownian dynamics, Gaussian steps with sigma = sqrt(2 D dt)
per axis. Each molecule's centre is confined to the capsule shrunk by its own radius,
with reflecting walls. The run stops at first contact: the centres come within
r_protein + r_sugar = 2.86 nm at any moment during a step (closest approach of the two
straight-line moves). dt = 14.4 ns keeps the sugar's rms step at 1/3 of the contact
distance. Both molecules start on the long axis, 0.1 um from their pole.

Choices that only affect the drawing:
  * panel times 5, 10, 15 ms: the extents of the red and blue traces in Goodsell's
    drawing (which has no time labels) match the median simulated paths at T/4, T/2,
    3T/4 with T ~ 20 ms. An earlier adapted version of the figure labelled its panels
    0.25-1 s, 30-50x too long for these physics;
  * seed 29: of seeds 0-63, the run whose first contact (1.001 s) is closest to
    Goodsell's "about a second". First-contact times are broadly spread (roughly
    exponential): seeds 0-63 give a median of 0.92 s and a mean of 1.43 s; an
    independent set of 64 runs gave a median of 1.02 s and a mean of 1.45 +- 0.16 s;
  * paths are projected onto the page (depth hidden, so paths can cross on the page
    without the molecules touching) and snapped to a grid 64 pixels across the cell,
    one stored position per ~3.9 us (the sugar moves ~1 pixel rms per axis);
  * older segments are drawn lighter: panels 1-3 fade over a 20 ms scale, the last
    panel over its full duration, drawing oldest segments first.

Requires numpy, matplotlib, numba (as simulation.py).
"""
import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from numba import njit, prange

K_B = 1.380649e-23  # J/K


@dataclass
class GoodsellConfig:
    cell_diameter_um: float = 1.0
    cell_length_um: float = 2.0          # pole to pole
    viscosity_mPas: float = 20.0
    temperature_C: float = 37.0
    protein_radius_nm: float = 2.5
    sugar_radius_nm: float = 0.36
    start_from_pole_um: float = 0.10
    step_fraction: float = 1 / 3         # sugar rms step per axis <= this x contact distance
    pixels_across: int = 64              # display grid
    t_max_s: float = 10.0                # give up if no contact by then

    @property
    def kT(self):
        return K_B * (self.temperature_C + 273.15)

    def stokes_einstein(self, r_nm):
        """D in um^2/s."""
        return self.kT / (6 * math.pi * self.viscosity_mPas * 1e-3 * r_nm * 1e-9) * 1e12

    @property
    def D_protein(self):
        return self.stokes_einstein(self.protein_radius_nm)

    @property
    def D_sugar(self):
        return self.stokes_einstein(self.sugar_radius_nm)

    @property
    def contact_nm(self):
        return self.protein_radius_nm + self.sugar_radius_nm

    @property
    def display_pixel_nm(self):
        return self.cell_diameter_um * 1e3 / self.pixels_across

    @property
    def dt_s(self):
        step_um = self.step_fraction * self.contact_nm * 1e-3
        return step_um ** 2 / (2 * self.D_sugar)

    @property
    def display_dt_s(self):
        a_um = self.display_pixel_nm * 1e-3
        return a_um ** 2 / (2 * self.D_sugar)


@dataclass
class RenderConfig:
    canvas_width_in: float = 16.0        # 4800 px at 300 dpi
    dpi: int = 300
    gap: float = 0.20                    # between cells, x cell width
    panel_margin: float = 0.06
    page_margin: float = 0.06
    brace_width: float = 0.75
    break_gap: float = 0.25              # extra space before the last panel
    page: str = "#FFFFFF"
    panel: str = "#E6E6E8"
    cytoplasm: str = "#FAFAFA"
    protein_color: str = "#2A36B8"
    sugar_color: str = "#D63A2E"
    text_color: str = "#000000"
    stroke_px: float = 0.40              # line width, in display pixels
    fade_tau: float = 0.60               # x fade time scale
    fade_min: float = 0.30
    fade_levels: int = 64
    chunk_s: float = 0.5e-3              # oldest-first ordering in the last panel
    marker_px: float = 4.0               # contact ring radius
    label_size: float = 0.17             # time label cap height, x cell width
    labels: bool = True
    fonts: list = field(default_factory=lambda: ["Helvetica Neue", "Helvetica", "Arial",
                                                 "DejaVu Sans"])


PANEL_TIMES_S = (0.005, 0.010, 0.015)
FADE_T_S = 0.020
SEED = 29


# ------------------------------------------------------------------ simulation

@njit(cache=True)
def _reflect(x, y, z, R, h):
    """Mirror a point back into the capsule of radius R and half-length h (cylinder part)."""
    for _ in range(4):
        if abs(y) <= h:
            rho = math.sqrt(x * x + z * z)
            if rho <= R:
                return x, y, z
            s = (2.0 * R - rho) / rho
            x *= s
            z *= s
        else:
            cy = h if y > 0 else -h
            dy = y - cy
            r = math.sqrt(x * x + dy * dy + z * z)
            if r <= R:
                return x, y, z
            s = (2.0 * R - r) / r
            x *= s
            z *= s
            y = cy + dy * s
    return x, y, z


@njit(cache=True)
def _closest(px, py, pz, pnx, pny, pnz, sx, sy, sz, snx, sny, snz):
    """Minimum distance between two points moving linearly during one step."""
    dx, dy, dz = sx - px, sy - py, sz - pz
    vx = (snx - sx) - (pnx - px)
    vy = (sny - sy) - (pny - py)
    vz = (snz - sz) - (pnz - pz)
    vv = vx * vx + vy * vy + vz * vz
    t = 0.0
    if vv > 0:
        t = min(1.0, max(0.0, -(dx * vx + dy * vy + dz * vz) / vv))
    ex, ey, ez = dx + vx * t, dy + vy * t, dz + vz * t
    return math.sqrt(ex * ex + ey * ey + ez * ez)


@njit(cache=True)
def run_bd(seed, dt, n_max, every, Rp, Rs, h, Dp, Ds, contact, y0, store):
    """Protein starts at (0, +y0, 0), sugar at (0, -y0, 0).
    Returns (stored protein, stored sugar, n_stored, fine steps taken, met)."""
    np.random.seed(seed)
    sp = math.sqrt(2.0 * Dp * dt)
    ss = math.sqrt(2.0 * Ds * dt)
    px, py, pz = 0.0, y0, 0.0
    sx, sy, sz = 0.0, -y0, 0.0
    n_store = (n_max // every + 2) if store else 1
    P = np.empty((n_store, 3))
    S = np.empty((n_store, 3))
    P[0] = (px, py, pz)
    S[0] = (sx, sy, sz)
    k = 1
    for i in range(1, n_max + 1):
        pnx, pny, pnz = _reflect(px + sp * np.random.standard_normal(),
                                 py + sp * np.random.standard_normal(),
                                 pz + sp * np.random.standard_normal(), Rp, h)
        snx, sny, snz = _reflect(sx + ss * np.random.standard_normal(),
                                 sy + ss * np.random.standard_normal(),
                                 sz + ss * np.random.standard_normal(), Rs, h)
        met = _closest(px, py, pz, pnx, pny, pnz, sx, sy, sz, snx, sny, snz) <= contact
        px, py, pz, sx, sy, sz = pnx, pny, pnz, snx, sny, snz
        if store and (i % every == 0 or met):
            P[k] = (px, py, pz)
            S[k] = (sx, sy, sz)
            k += 1
        if met:
            return P, S, k, i, True
    return P, S, k, n_max, False


@njit(parallel=True, cache=True)
def contact_times(seeds, dt, n_max, Rp, Rs, h, Dp, Ds, contact, y0):
    """First-contact time for many independent runs (no storage), in parallel."""
    out = np.empty(len(seeds))
    for j in prange(len(seeds)):
        _, _, _, n, met = run_bd(seeds[j], dt, n_max, n_max, Rp, Rs, h, Dp, Ds,
                                 contact, y0, False)
        out[j] = n * dt if met else np.nan
    return out


def _geometry(cfg):
    Rc = cfg.cell_diameter_um / 2
    h = cfg.cell_length_um / 2 - Rc
    y0 = cfg.cell_length_um / 2 - cfg.start_from_pole_um
    return Rc - cfg.protein_radius_nm * 1e-3, Rc - cfg.sugar_radius_nm * 1e-3, h, y0


def simulate(cfg, seed):
    """Run to first contact. Positions (um) stored every display_dt; last entry = contact."""
    Rp, Rs, h, y0 = _geometry(cfg)
    every = max(1, int(round(cfg.display_dt_s / cfg.dt_s)))
    n = int(np.ceil(cfg.t_max_s / cfg.dt_s / every)) * every
    P, S, k, n_done, met = run_bd(seed, cfg.dt_s, n, every, Rp, Rs, h, cfg.D_protein,
                                  cfg.D_sugar, cfg.contact_nm * 1e-3, y0, True)
    assert met, f"seed {seed}: no contact within {cfg.t_max_s} s"
    t = np.arange(k) * every * cfg.dt_s
    t[-1] = n_done * cfg.dt_s
    return P[:k], S[:k], t


def find_seed(cfg, target_s=1.0, n=64):
    """Seed in 0..n-1 whose first contact time is closest to target_s."""
    Rp, Rs, h, y0 = _geometry(cfg)
    t = contact_times(np.arange(n, dtype=np.int64), cfg.dt_s, int(cfg.t_max_s / cfg.dt_s),
                      Rp, Rs, h, cfg.D_protein, cfg.D_sugar, cfg.contact_nm * 1e-3, y0)
    best = int(np.nanargmin(np.abs(t - target_s)))
    print(f"contact times over seeds 0-{n - 1}: median {np.nanmedian(t):.3f} s, "
          f"mean {np.nanmean(t):.3f} s; closest to {target_s} s: seed {best} "
          f"({t[best]:.4f} s)")
    return best


# ------------------------------------------------------------------ drawing

def capsule_xy(cx, cy, R, h, n=180):
    t = np.linspace(0, np.pi, n)
    top = np.c_[cx + R * np.cos(t), cy + h + R * np.sin(t)]
    bot = np.c_[cx - R * np.cos(t), cy - h - R * np.sin(t)]
    return np.vstack([top, bot])


def display_path(xyz, t, a_um):
    """Project to x-y in display-pixel units, snap to the grid and drop repeats."""
    ij = np.rint(xyz[:, :2] / a_um).astype(int)
    keep = np.r_[True, (np.diff(ij, axis=0) != 0).any(1)]
    return ij[keep].astype(float), t[keep]


def tinted_runs(xy, t, t_now, t_scale, color, rc, off):
    """Path up to t_now as runs of constant tint; tint fades with age / t_scale."""
    k = np.searchsorted(t, t_now, side="right")
    if k < 2:
        return []
    xy, t = xy[:k] + off, t[:k]
    tm = 0.5 * (t[:-1] + t[1:])
    age = (t_now - tm) / t_scale
    inten = rc.fade_min + (1 - rc.fade_min) * np.exp(-age / rc.fade_tau)
    lev = np.clip((inten * (rc.fade_levels - 1)).round().astype(int), 0, rc.fade_levels - 1)
    base, white = np.array(to_rgb(color)), np.array(to_rgb(rc.cytoplasm))
    breaks = np.flatnonzero(np.diff(lev)) + 1
    runs = []
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(tm)]):
        runs.append((xy[a:b + 1],
                     tuple(white + (base - white) * lev[a] / (rc.fade_levels - 1))))
    return runs


def oldest_first(paths, t_now, rc, off):
    """Tinted runs for several paths, faded over [0, t_now] and ordered by time across
    paths (in chunks of rc.chunk_s), so newer segments are drawn on top of older ones."""
    tagged = []
    for xy, t, color in paths:
        k = np.searchsorted(t, t_now, side="right")
        edges = np.searchsorted(t[:k], np.arange(0, t_now, rc.chunk_s))
        for a, b in zip(edges, np.r_[edges[1:], k - 1]):
            if b > a:
                runs = tinted_runs(xy[a:b + 1], t[a:b + 1], t_now, t_now, color, rc, off)
                tagged += [(t[a], r) for r in runs]
    return [r for _, r in sorted(tagged, key=lambda x: x[0])]


def brace(x, y0, y1, w):
    """Curly brace from (x, y0) to (x, y1), tip pointing left by w."""
    ym, d, xm = (y0 + y1) / 2, 0.08 * (y1 - y0), x - w / 2
    v = [(x, y1), (xm, y1), (xm, y1 - d), (xm, ym + d), (xm, ym), (x - w, ym),
         (xm, ym), (xm, ym - d), (xm, y0 + d), (xm, y0), (x, y0)]
    c = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
         MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    return MPath(v, c)


def label(t):
    return f"{t:.3g} s" if t >= 0.1 else f"{t * 1e3:.3g} ms"


def render(cfg, rc, P, S, t, outdir, stem):
    snaps = list(PANEL_TIMES_S) + [float(t[-1])]
    a = cfg.display_pixel_nm * 1e-3
    Pxy, Pt = display_path(P, t, a)
    Sxy, St = display_path(S, t, a)

    R = cfg.pixels_across / 2 + 0.5
    h = (cfg.cell_length_um - cfg.cell_diameter_um) / 2 / a
    cw = 2 * R
    n = len(snaps)
    pad = rc.panel_margin * cw
    brk = rc.break_gap * cw
    panel_w = n * cw + (n - 1) * rc.gap * cw + 2 * pad + brk
    panel_h = 2 * (R + h) + 2 * pad
    pm, bw = rc.page_margin * cw, (rc.brace_width * cw if rc.labels else 0.0)
    total_w, total_h = bw + panel_w + 2 * pm, panel_h + 2 * pm

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": rc.fonts,
                         "svg.hashsalt": "goodsell-fig1", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(rc.canvas_width_in, rc.canvas_width_in * total_h / total_w),
                     facecolor=rc.page)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect("equal")
    ax.axis("off")
    pt = rc.canvas_width_in * 72 / total_w              # points per data unit
    lw = rc.stroke_px * pt

    x_panel = pm + bw
    ax.add_patch(Rectangle((x_panel, pm), panel_w, panel_h, fc=rc.panel, lw=0, zorder=0))
    cy = pm + panel_h / 2
    for i, t_now in enumerate(snaps):
        last = i == n - 1
        cx = x_panel + pad + R + i * (cw + rc.gap * cw) + (brk if last else 0.0)
        cell = Polygon(capsule_xy(cx, cy, R, h), closed=True, fc=rc.cytoplasm, lw=0, zorder=1)
        ax.add_patch(cell)
        off = np.array([cx, cy])
        if last:     # complete paths to contact, faded over their duration, oldest first
            runs = oldest_first(((Sxy, St, rc.sugar_color), (Pxy, Pt, rc.protein_color)),
                                t_now, rc, off)
        else:        # protein drawn over sugar
            runs = (tinted_runs(Sxy, St, t_now, FADE_T_S, rc.sugar_color, rc, off)
                    + tinted_runs(Pxy, Pt, t_now, FADE_T_S, rc.protein_color, rc, off))
        lc = LineCollection([r[0] for r in runs], colors=[r[1] for r in runs],
                            linewidths=lw, capstyle="projecting", joinstyle="miter",
                            zorder=2)
        lc.set_clip_path(cell)
        ax.add_collection(lc)
        y_lab = cy + 0.12 * (R + h)
        if last and abs(Pxy[-1, 1] - 0.08 * cw + cy - y_lab) < 0.3 * cw:
            y_lab = cy - 0.35 * (R + h)                  # keep the label off the contact
        if rc.labels:
            pe = ([patheffects.withStroke(linewidth=0.03 * cw * pt, foreground="white")]
                  if last else None)
            ax.text(cx - R + 0.07 * cw, y_lab, label(t_now), ha="left", va="baseline",
                    fontsize=rc.label_size * cw * pt / 0.72, color=rc.text_color,
                    zorder=5, path_effects=pe)

    # ring where the two met; break in the time axis before the last panel
    halo = [patheffects.withStroke(linewidth=5 * lw, foreground="white")]
    ax.add_patch(Circle(Pxy[-1] + off, rc.marker_px, fc="none", ec=rc.text_color,
                        lw=2 * lw, zorder=4, path_effects=halo))
    if rc.labels:
        ax.text(cx - R - (rc.gap * cw + brk) / 2, cy, "…", ha="center", va="center",
                fontsize=rc.label_size * cw * pt / 0.72, color=rc.text_color)
        ax.text(cx - R + 0.07 * cw, y_lab - 0.09 * cw, "first contact", ha="left",
                va="top", fontsize=0.075 * cw * pt / 0.72, color=rc.text_color,
                zorder=5, path_effects=halo)
        x_b = x_panel + pad - 0.04 * cw
        ax.add_patch(PathPatch(brace(x_b, cy - (R + h) * 0.9, cy + (R + h) * 0.9, 0.28 * cw),
                               fc="none", ec=rc.text_color, lw=0.035 * cw * pt,
                               capstyle="round", zorder=3))
        ax.text(x_b - 0.52 * cw, cy, "Bacterial cell", rotation=90, ha="center",
                va="center", fontsize=0.16 * cw * pt / 0.72, color=rc.text_color)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=rc.dpi, facecolor=rc.page,
                    metadata={"Date": None} if ext in ("svg", "pdf") else None)
    plt.close(fig)
    print(f"wrote {outdir / stem}.png/.svg/.pdf  (panels at "
          + ", ".join(label(x) for x in snaps) + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--find-seed", action="store_true",
                    help="choose the seed (0-63) whose contact is closest to 1 s")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("goodsell_output"))
    a = ap.parse_args()
    cfg = GoodsellConfig()
    print(f"D_protein = {cfg.D_protein:.2f} um^2/s, D_sugar = {cfg.D_sugar:.2f} um^2/s, "
          f"dt = {cfg.dt_s * 1e9:.1f} ns")
    seed = find_seed(cfg) if a.find_seed else a.seed
    P, S, t = simulate(cfg, seed)
    print(f"seed {seed}: first contact at {t[-1]:.4f} s")
    rc = RenderConfig(labels=not a.no_labels)
    render(cfg, rc, P, S, t, a.outdir,
           "goodsell_fig1" + ("_nolabels" if a.no_labels else ""))


if __name__ == "__main__":
    main()
