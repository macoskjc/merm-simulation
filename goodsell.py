"""Brownian motion of a sugar and a protein in a bacterium (Figure 1, after D. S. Goodsell).

    python goodsell.py                      # seed 29 -> goodsell_output/goodsell_fig1.png/.svg/.pdf
    python goodsell.py --no-labels          # same figure without text
    python goodsell.py --find-seed          # redo the seed choice (64 runs, a few minutes)
    python validate_goodsell.py             # numerical validation (see README)

A sugar (glucose, red) and a typical protein (GFP-sized, blue) start at opposite poles
of a bacterium and diffuse until they first touch. The figure shows the paths at three
chosen illustration times (5, 10, 15 ms) and, after a break in time, the complete paths
up to the first encounter.

Model: two spherical particles with constant diffusion coefficients in a homogeneous
medium inside a reflecting capsule. Parameters from the Figure 1 caption: capsule 1 um
wide and 2 um long, viscosity 20 mPa.s (~30x water at 37 C), 37 C. Diffusion coefficients
from Stokes-Einstein, D = kT / (6 pi eta r_h), with hydrodynamic radii of 2.5 nm (protein,
GFP-sized) and 0.36 nm (glucose): D_protein = 4.54, D_sugar = 31.55 um^2/s. They can
also be set directly. Sources for the radii:
  * GFP: D = 87 um^2/s in buffer at room temperature (Swaminathan et al. 1997, Biophys J
    72:1900) gives 2.5-2.8 nm; EGFP D(20 C, water) = 83.4 um^2/s by analytical
    ultracentrifugation (Vamosi et al. 2016, Sci Rep 6:33022) gives 2.57 nm.
  * glucose: 0.36 nm (Ribeiro et al. 2006, J Chem Eng Data 51:1836), itself derived from
    the measured D (0.679e-9 m^2/s at 25 C) by Stokes-Einstein.
Assumptions and limitations:
  * One effective viscosity for both molecules (20 mPa.s, the caption's value; 29x water
    at 37 C) is an assumption, not a measured property of bacterial cytoplasm. For
    comparison, GFP in E. coli diffuses ~11x slower than in water, and no single
    effective viscosity describes all proteins (Elowitz et al. 1999, J Bacteriol
    181:197). Crowding, specific interactions, hydrodynamic coupling and
    orientation-dependent binding are not represented.
  * "Encounter" is geometric: the centres come within the encounter distance
    (default: sum of the steric radii, 2.86 nm). It is not necessarily binding.
  * Steric radii set wall exclusion (each centre is confined to the capsule shrunk by
    its steric radius); they default to the hydrodynamic radii but are separate settings.

Numerics:
  * Free diffusion: independent Gaussian increments, sigma = sqrt(2 D dt) per coordinate,
    base step dt = 14.4 ns (rms relative 3D displacement per step sqrt(6 D_rel dt)
    = 0.62 x encounter distance, D_rel = D_protein + D_sugar).
  * Walls: an end point outside the accessible capsule is mirrored back across the
    surface. This is an approximation on curved walls; validate_goodsell.py checks that a
    uniform distribution stays uniform, including near the wall and the cylinder-cap joins.
  * Encounters: whenever the straight-line relative move of a step comes within
    a + 8 sqrt(2 D_rel dt) of contact, the step is refined by Brownian-bridge bisection:
    both paths are split at the midpoint, drawn exactly from the bridge law (mean = average
    of the end points, variance D dt_sub / 2 per coordinate), recursively down to
    dt / 2^L (default L = 10). Sub-intervals whose straight-line approach stays farther
    than the same 8-sigma margin are skipped; the chance that a bridge deviates that far
    is below 1e-17 per interval. On the finest intervals the first contact is located on
    the straight-line relative move, giving the event time and both positions. The bridge
    draws come from a separate counter-based random stream, so the base trajectory is the
    same for every L. Remaining approximations: straight-line contact on the finest
    intervals (the convergence in L is measured), and free bridges within steps that
    also reflected off a wall (0.2% of refined steps in a test sphere, R = 0.25 um).
  * Censoring: a run with no encounter by t_max (10 s) is reported as censored.

Illustration choices (they do not change the physics):
  * panel times 5, 10, 15 ms are chosen for illustration;
  * seed: of seeds 0-63, the run whose first encounter is closest to Goodsell's "about a
    second" (--find-seed). This choice is for illustration, not evidence;
  * paths are sampled every ~3.9 us, projected onto the page (depth hidden, so paths can
    cross on the page without the molecules touching) and snapped to a grid 64 pixels
    across the cell. They are sampled projections, not molecular-scale resolved paths;
  * older segments are drawn lighter: panels 1-3 fade over a 20 ms scale, the last
    panel over its full duration, drawing oldest segments first.

Requires numpy, matplotlib, numba (as simulation.py).
"""
import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    # hydrodynamic radii: used only for Stokes-Einstein (ignored if D is given)
    protein_radius_nm: float = 2.5
    sugar_radius_nm: float = 0.36
    D_protein_um2s: Optional[float] = None
    D_sugar_um2s: Optional[float] = None
    # steric radii: wall exclusion (default: hydrodynamic radii)
    protein_steric_nm: Optional[float] = None
    sugar_steric_nm: Optional[float] = None
    # encounter distance between centres (default: sum of steric radii)
    encounter_distance_nm: Optional[float] = None
    start_from_pole_um: float = 0.10
    step_fraction: float = 1 / 3         # sets dt = 14.4 ns (see docstring)
    dt_override_s: Optional[float] = None  # set the base step directly
    refine_levels: int = 10              # encounter refinement down to dt / 2^L
    near_sigmas: float = 8.0             # refinement margin, in bridge standard deviations
    pixels_across: int = 64              # display grid
    t_max_s: float = 10.0                # runs without encounter by then are censored

    def __post_init__(self):
        checks = [
            (self.cell_diameter_um > 0, "cell diameter must be > 0"),
            (self.cell_length_um >= self.cell_diameter_um,
             "cell length must be >= cell diameter (capsule)"),
            (self.viscosity_mPas > 0, "viscosity must be > 0"),
            (self.temperature_C > -273.15, "temperature must be above absolute zero"),
            (self.protein_radius_nm > 0 and self.sugar_radius_nm > 0, "radii must be > 0"),
            (self.D_protein > 0 and self.D_sugar > 0, "diffusion coefficients must be > 0"),
            (self.steric_protein_nm >= 0 and self.steric_sugar_nm >= 0,
             "steric radii must be >= 0"),
            (self.contact_nm > 0, "encounter distance must be > 0"),
            (max(self.steric_protein_nm, self.steric_sugar_nm) * 1e-3
             < self.cell_diameter_um / 2, "particles must fit in the cell"),
            (self.step_fraction > 0 and self.refine_levels >= 0 and self.near_sigmas > 0,
             "invalid numerical settings"),
            (self.pixels_across > 0 and self.t_max_s > 0, "invalid display or t_max"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)
        y0 = self.cell_length_um / 2 - self.start_from_pole_um
        Rc = self.cell_diameter_um / 2
        h = self.cell_length_um / 2 - Rc
        for r_nm in (self.steric_protein_nm, self.steric_sugar_nm):
            if not (0 <= y0 <= h + Rc - r_nm * 1e-3):
                raise ValueError("start position lies outside a particle's accessible capsule")
        if 2 * y0 <= self.contact_nm * 1e-3:
            raise ValueError("particles would start in contact")

    @property
    def kT(self):
        return K_B * (self.temperature_C + 273.15)

    def stokes_einstein(self, r_nm):
        """D in um^2/s."""
        return self.kT / (6 * math.pi * self.viscosity_mPas * 1e-3 * r_nm * 1e-9) * 1e12

    @property
    def D_protein(self):
        if self.D_protein_um2s is not None:
            return self.D_protein_um2s
        return self.stokes_einstein(self.protein_radius_nm)

    @property
    def D_sugar(self):
        if self.D_sugar_um2s is not None:
            return self.D_sugar_um2s
        return self.stokes_einstein(self.sugar_radius_nm)

    @property
    def steric_protein_nm(self):
        return self.protein_radius_nm if self.protein_steric_nm is None else self.protein_steric_nm

    @property
    def steric_sugar_nm(self):
        return self.sugar_radius_nm if self.sugar_steric_nm is None else self.sugar_steric_nm

    @property
    def contact_nm(self):
        if self.encounter_distance_nm is not None:
            return self.encounter_distance_nm
        return self.steric_protein_nm + self.steric_sugar_nm

    @property
    def display_pixel_nm(self):
        return self.cell_diameter_um * 1e3 / self.pixels_across

    @property
    def dt_s(self):
        if self.dt_override_s is not None:
            return self.dt_override_s
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


PANEL_TIMES_S = (0.005, 0.010, 0.015)     # illustration times
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
def _first_contact(p0, p1, s0, s1, a):
    """Earliest alpha in [0, 1] with |r0 + alpha dr| = a on the straight-line relative
    move (r = sugar - protein), 0 if already in contact, -1 if none."""
    r0 = s0 - p0
    dr = (s1 - s0) - (p1 - p0)
    C = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2] - a * a
    if C <= 0.0:
        return 0.0
    A = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]
    if A == 0.0:
        return -1.0
    B = 2.0 * (r0[0] * dr[0] + r0[1] * dr[1] + r0[2] * dr[2])
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return -1.0
    alpha = (-B - math.sqrt(disc)) / (2.0 * A)
    return alpha if 0.0 <= alpha <= 1.0 else -1.0


# counter-based random stream for the bridge draws (splitmix64), independent of the
# base-step stream, so the base trajectory does not depend on the refinement depth
_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True)
def _mix(z):
    z = z + _G
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _key(seed, step, node):
    k = _mix(np.uint64(seed))
    k = _mix(k ^ np.uint64(step))
    return _mix(k ^ np.uint64(node))


@njit(cache=True)
def _normals6(key, out):
    """Six independent standard normals from one key (Box-Muller, uniforms in (0, 1))."""
    for j in range(3):
        u1 = (np.float64(_mix(key + np.uint64(2 * j + 1)) >> np.uint64(11)) + 0.5) * 2.0 ** -53
        u2 = (np.float64(_mix(key + np.uint64(2 * j + 2)) >> np.uint64(11)) + 0.5) * 2.0 ** -53
        r = math.sqrt(-2.0 * math.log(u1))
        out[2 * j] = r * math.cos(2.0 * math.pi * u2)
        out[2 * j + 1] = r * math.sin(2.0 * math.pi * u2)


@njit(cache=True)
def _refine(seed, step, L, near, p0, p1, s0, s1, Dp, Ds, dt, a, Rp, Rs, h, walls, out):
    """Earliest encounter within one base step by Brownian-bridge bisection to dt / 2^L.
    Returns the fraction of the step at which it happens (-1: none); out = (P, S) there."""
    Drel = Dp + Ds
    n = 2 * L + 4
    lvl = np.empty(n, np.int64)
    node = np.empty(n, np.int64)
    f0 = np.empty(n)
    E = np.empty((n, 4, 3))                    # p0, p1, s0, s1 of each pending interval
    z = np.empty(6)
    top = 0
    lvl[0], node[0], f0[0] = 0, 0, 0.0
    E[0, 0], E[0, 1], E[0, 2], E[0, 3] = p0, p1, s0, s1
    top = 1
    while top > 0:
        top -= 1
        l, nd, f = lvl[top], node[top], f0[top]
        q0, q1, t0, t1 = E[top, 0].copy(), E[top, 1].copy(), E[top, 2].copy(), E[top, 3].copy()
        span = 0.5 ** l
        if l == L:
            al = _first_contact(q0, q1, t0, t1, a)
            if al >= 0.0:
                out[0:3] = q0 + al * (q1 - q0)
                out[3:6] = t0 + al * (t1 - t0)
                return f + al * span
            continue
        dts = dt * span
        if _closest(q0[0], q0[1], q0[2], q1[0], q1[1], q1[2],
                    t0[0], t0[1], t0[2], t1[0], t1[1], t1[2]) > a + near * math.sqrt(2.0 * Drel * dts):
            continue
        _normals6(_key(seed, step, nd), z)
        sp, ss = math.sqrt(Dp * dts / 2.0), math.sqrt(Ds * dts / 2.0)
        pm = 0.5 * (q0 + q1) + sp * z[0:3]
        sm = 0.5 * (t0 + t1) + ss * z[3:6]
        if walls:
            pm[0], pm[1], pm[2] = _reflect(pm[0], pm[1], pm[2], Rp, h)
            sm[0], sm[1], sm[2] = _reflect(sm[0], sm[1], sm[2], Rs, h)
        # push the later half first, so the earlier half is examined first
        lvl[top], node[top], f0[top] = l + 1, 2 * nd + 2, f + span / 2
        E[top, 0], E[top, 1], E[top, 2], E[top, 3] = pm, q1, sm, t1
        top += 1
        lvl[top], node[top], f0[top] = l + 1, 2 * nd + 1, f
        E[top, 0], E[top, 1], E[top, 2], E[top, 3] = q0, pm, t0, sm
        top += 1
    return -1.0


@njit(cache=True)
def run_bd(seed, dt, n_max, every, Rp, Rs, h, Dp, Ds, a, yp0, ys0, store, L, near, walls):
    """Protein starts at (0, yp0, 0), sugar at (0, ys0, 0); runs to first encounter.
    Returns (stored P, stored S, n_stored, event time, met, near steps, near steps that
    also reflected off a wall). The last stored sample is the event (positions and time)."""
    np.random.seed(seed)
    sp = math.sqrt(2.0 * Dp * dt)
    ss = math.sqrt(2.0 * Ds * dt)
    margin = a + near * math.sqrt(2.0 * (Dp + Ds) * dt)
    p = np.array([0.0, yp0, 0.0])
    s = np.array([0.0, ys0, 0.0])
    pn, sn, ev = np.empty(3), np.empty(3), np.empty(6)
    n_store = (n_max // every + 2) if store else 1
    P = np.empty((n_store, 3))
    S = np.empty((n_store, 3))
    P[0], S[0] = p, s
    k = 1
    n_near, n_near_wall = 0, 0
    for i in range(1, n_max + 1):
        rx, ry, rz = (p[0] + sp * np.random.standard_normal(),
                      p[1] + sp * np.random.standard_normal(),
                      p[2] + sp * np.random.standard_normal())
        qx, qy, qz = (s[0] + ss * np.random.standard_normal(),
                      s[1] + ss * np.random.standard_normal(),
                      s[2] + ss * np.random.standard_normal())
        if walls:
            pn[0], pn[1], pn[2] = _reflect(rx, ry, rz, Rp, h)
            sn[0], sn[1], sn[2] = _reflect(qx, qy, qz, Rs, h)
        else:
            pn[0], pn[1], pn[2] = rx, ry, rz
            sn[0], sn[1], sn[2] = qx, qy, qz
        if _closest(p[0], p[1], p[2], pn[0], pn[1], pn[2],
                    s[0], s[1], s[2], sn[0], sn[1], sn[2]) <= margin:
            n_near += 1
            if pn[0] != rx or pn[1] != ry or pn[2] != rz or \
                    sn[0] != qx or sn[1] != qy or sn[2] != qz:
                n_near_wall += 1
            f = _refine(seed, i, L, near, p, pn, s, sn, Dp, Ds, dt, a, Rp, Rs, h, walls, ev)
            if f >= 0.0:
                if store:
                    P[k], S[k] = ev[0:3], ev[3:6]
                    k += 1
                return P, S, k, (i - 1 + f) * dt, True, n_near, n_near_wall
        p[:] = pn
        s[:] = sn
        if store and i % every == 0:
            P[k], S[k] = p, s
            k += 1
    return P, S, k, n_max * dt, False, n_near, n_near_wall


@njit(cache=True)
def _run_multi(seed, dt, n_max, Rp, Rs, h, Dp, Ds, a, p0, s0, Ls, near, walls, t_out, met_out):
    """One base trajectory; first-encounter time for each refinement depth in Ls (paired)."""
    np.random.seed(seed)
    sp = math.sqrt(2.0 * Dp * dt)
    ss = math.sqrt(2.0 * Ds * dt)
    margin = a + near * math.sqrt(2.0 * (Dp + Ds) * dt)
    p = p0.copy()
    s = s0.copy()
    pn, sn, ev = np.empty(3), np.empty(3), np.empty(6)
    nL = len(Ls)
    for j in range(nL):
        t_out[j], met_out[j] = n_max * dt, False
    left = nL
    for i in range(1, n_max + 1):
        rx, ry, rz = (p[0] + sp * np.random.standard_normal(),
                      p[1] + sp * np.random.standard_normal(),
                      p[2] + sp * np.random.standard_normal())
        qx, qy, qz = (s[0] + ss * np.random.standard_normal(),
                      s[1] + ss * np.random.standard_normal(),
                      s[2] + ss * np.random.standard_normal())
        if walls:
            pn[0], pn[1], pn[2] = _reflect(rx, ry, rz, Rp, h)
            sn[0], sn[1], sn[2] = _reflect(qx, qy, qz, Rs, h)
        else:
            pn[0], pn[1], pn[2] = rx, ry, rz
            sn[0], sn[1], sn[2] = qx, qy, qz
        if _closest(p[0], p[1], p[2], pn[0], pn[1], pn[2],
                    s[0], s[1], s[2], sn[0], sn[1], sn[2]) <= margin:
            for j in range(nL):
                if not met_out[j]:
                    f = _refine(seed, i, Ls[j], near, p, pn, s, sn, Dp, Ds, dt, a,
                                Rp, Rs, h, walls, ev)
                    if f >= 0.0:
                        t_out[j], met_out[j] = (i - 1 + f) * dt, True
                        left -= 1
            if left == 0:
                return
        p[:] = pn
        s[:] = sn


@njit(parallel=True, cache=True)
def encounter_times(seeds, dt, n_max, Rp, Rs, h, Dp, Ds, a, P0, S0, Ls, near, walls):
    """First-encounter times, shape (len(seeds), len(Ls)), and whether each run met
    (False = censored at n_max * dt). All depths share each seed's base trajectory.
    P0, S0: start positions (len(seeds), 3)."""
    T = np.empty((len(seeds), len(Ls)))
    M = np.zeros((len(seeds), len(Ls)), np.bool_)
    for j in prange(len(seeds)):
        _run_multi(seeds[j], dt, n_max, Rp, Rs, h, Dp, Ds, a, P0[j], S0[j], Ls, near, walls,
                   T[j], M[j])
    return T, M


def geometry(cfg):
    """Accessible capsule radii (steric exclusion), half-length h, start offset y0."""
    Rc = cfg.cell_diameter_um / 2
    h = cfg.cell_length_um / 2 - Rc
    y0 = cfg.cell_length_um / 2 - cfg.start_from_pole_um
    return Rc - cfg.steric_protein_nm * 1e-3, Rc - cfg.steric_sugar_nm * 1e-3, h, y0


def simulate(cfg, seed):
    """Run to first encounter. Positions (um) stored every display_dt; the last entry is
    the encounter (exact time and positions within the method)."""
    Rp, Rs, h, y0 = geometry(cfg)
    every = max(1, int(round(cfg.display_dt_s / cfg.dt_s)))
    n = int(np.ceil(cfg.t_max_s / cfg.dt_s / every)) * every
    P, S, k, t_ev, met, n_near, n_wall = run_bd(
        seed, cfg.dt_s, n, every, Rp, Rs, h, cfg.D_protein, cfg.D_sugar,
        cfg.contact_nm * 1e-3, y0, -y0, True, cfg.refine_levels, cfg.near_sigmas, True)
    if not met:
        raise RuntimeError(f"seed {seed}: no encounter within {cfg.t_max_s} s (censored)")
    t = np.arange(k) * every * cfg.dt_s
    t[-1] = t_ev
    return P[:k], S[:k], t


def ensemble(cfg, seeds, Ls=None, walls=True, starts=None, n_max=None):
    """Encounter times for many seeds and refinement depths (paired): (T, met).
    starts = (protein y, sugar y) on the long axis (default: 0.1 um from opposite poles),
    or a pair of (len(seeds), 3) arrays of start points."""
    Rp, Rs, h, y0 = geometry(cfg)
    seeds = np.asarray(seeds, np.int64)
    if starts is None:
        starts = (y0, -y0)
    if np.ndim(starts[0]) == 0:
        P0 = np.zeros((len(seeds), 3)); P0[:, 1] = starts[0]
        S0 = np.zeros((len(seeds), 3)); S0[:, 1] = starts[1]
    else:
        P0, S0 = np.asarray(starts[0], float), np.asarray(starts[1], float)
    Ls = np.asarray([cfg.refine_levels] if Ls is None else Ls, np.int64)
    n_max = int(cfg.t_max_s / cfg.dt_s) if n_max is None else n_max
    return encounter_times(seeds, cfg.dt_s, n_max, Rp, Rs, h,
                           cfg.D_protein, cfg.D_sugar, cfg.contact_nm * 1e-3,
                           P0, S0, Ls, cfg.near_sigmas, walls)


def find_seed(cfg, target_s=1.0, n=64):
    """Seed in 0..n-1 whose first encounter is closest to target_s (illustration only)."""
    T, M = ensemble(cfg, np.arange(n))
    t, met = T[:, 0], M[:, 0]
    best = int(np.argmin(np.where(met, np.abs(t - target_s), np.inf)))
    print(f"seeds 0-{n - 1}: {met.sum()} met, {(~met).sum()} censored at {cfg.t_max_s:g} s; "
          f"closest to {target_s:g} s: seed {best} ({t[best]:.4f} s)")
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
                    help="choose the seed (0-63) whose encounter is closest to 1 s")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--outdir", type=Path, default=Path("goodsell_output"))
    a = ap.parse_args()
    cfg = GoodsellConfig()
    print(f"D_protein = {cfg.D_protein:.2f} um^2/s, D_sugar = {cfg.D_sugar:.2f} um^2/s, "
          f"dt = {cfg.dt_s * 1e9:.1f} ns, refinement to dt/2^{cfg.refine_levels}")
    seed = find_seed(cfg) if a.find_seed else a.seed
    P, S, t = simulate(cfg, seed)
    print(f"seed {seed}: first encounter at {t[-1]:.4f} s")
    if t[-1] <= PANEL_TIMES_S[-1]:
        raise SystemExit(f"encounter at {t[-1]:.4f} s precedes the illustration panels; "
                         "choose another seed")
    rc = RenderConfig(labels=not a.no_labels)
    render(cfg, rc, P, S, t, a.outdir,
           "goodsell_fig1" + ("_nolabels" if a.no_labels else ""))


if __name__ == "__main__":
    main()
