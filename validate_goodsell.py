"""Numerical validation of goodsell.py.

    python validate_goodsell.py                 # all tests (~2 h on 10 cores)
    python validate_goodsell.py rng benchmark   # selected tests
    Tests: rng free walls benchmark sphere_exact split finite_size pairs convergence
           capsule display

Writes goodsell_validation/results.json and prints a summary of each test:
  rng          the bridge random stream: normal moments, bridge midpoint variance
  free         free diffusion: increment variance 2 D dt, MSD 6 D t
  walls        reflecting capsule: containment, and a uniform start stays uniform
               (bins in distance to the wall and along the axis) at dt, dt/4, dt/16
  benchmark    two particles in free space from r0 = 2a, against the exact first-passage
               law P(T <= t) = (a / r0) erfc((r0 - a) / sqrt(4 D_rel t)), for
               refinement depths L = 0 (straight-line test only) to 10
  convergence  the capsule problem, paired over L on the same base trajectories
  sphere_exact fixed target at the centre of a reflecting sphere, against the exact
               mean first-passage time, at dt and dt/4
  split        small sphere: the mean should depend on D_protein + D_sugar only
  finite_size  small sphere, both mobile, uniform starts, encounter distance a and a/2
  pairs        no reactions: close-pair probability stays at its equilibrium value
  capsule      encounter statistics in the capsule (with censoring), against the
               well-mixed small-target estimate V / (4 pi D_rel a): opposite-pole
               starts (two independent sets, one paired over L) and uniform starts
  display      the display grid and storage interval do not change the event
"""
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit, prange

import goodsell as g

OUT = Path("goodsell_validation")
RES = {}


def report(name, d):
    RES[name] = d
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(RES, indent=2, default=float))


# ------------------------------------------------------------------ rng

@njit(cache=True)
def _draw_normals(seed, n):
    out = np.empty(6 * n)
    z = np.empty(6)
    for i in range(n):
        g._normals6(g._key(seed, i, 0), z)
        out[6 * i:6 * i + 6] = z
    return out


@njit(cache=True)
def _levy_increments(seed, n_paths, L, D, dt):
    """Free 1D paths over dt built by Brownian-bridge bisection (goodsell._refine law)."""
    N = 2 ** L
    out = np.empty((n_paths, N))
    z = np.empty(6)
    x = np.empty(N + 1)
    for j in range(n_paths):
        g._normals6(g._key(seed, j, 10 ** 9), z)
        x[0], x[N] = 0.0, math.sqrt(2 * D * dt) * z[0]
        step = N
        lvl = 0
        while step > 1:
            half = step // 2
            dts = dt * 0.5 ** lvl
            for i0 in range(0, N, step):
                g._normals6(g._key(seed * 7919 + j, lvl, i0), z)
                x[i0 + half] = 0.5 * (x[i0] + x[i0 + step]) + math.sqrt(D * dts / 2) * z[0]
            step = half
            lvl += 1
        for i in range(N):
            out[j, i] = x[i + 1] - x[i]
    return out


def test_rng():
    z = _draw_normals(123, 500_000)
    m, v = z.mean(), z.var()
    kurt = ((z - m) ** 4).mean() / v ** 2
    lag1 = np.corrcoef(z[:-1], z[1:])[0, 1]
    # Levy construction with the same bridge law and stream as goodsell._refine: draw
    # the end point of a free path over dt, then bisect to dt / 2^L. The finest
    # increments must be independent N(0, 2 D dt / 2^L).
    cfg = g.GoodsellConfig()
    D, dt, L = cfg.D_sugar, cfg.dt_s, 8
    inc = _levy_increments(7, 4000, L, D, dt)
    ratio = inc.var() / (2 * D * dt / 2 ** L)
    lag1_inc = np.corrcoef(inc[:, :-1].ravel(), inc[:, 1:].ravel())[0, 1]
    d = dict(n=len(z), mean=m, var=v, kurtosis=kurt, lag1_corr=lag1,
             bridge_leaf_var_ratio=ratio, bridge_leaf_lag1_corr=lag1_inc)
    print(f"rng: mean {m:+.4f}, var {v:.4f}, kurtosis {kurt:.3f} (3), lag-1 corr "
          f"{lag1:+.4f}; bridge bisection to dt/2^{L}: leaf increment var / (2 D dt_L) = "
          f"{ratio:.4f}, lag-1 corr {lag1_inc:+.4f}")
    report("rng", d)


# ------------------------------------------------------------------ free diffusion

@njit(parallel=True, cache=True)
def _free_paths(seed, n_part, n_steps, sigma):
    X = np.empty((n_part, n_steps + 1, 3))
    for j in prange(n_part):
        np.random.seed(seed + j)
        X[j, 0] = 0.0
        for i in range(1, n_steps + 1):
            for c in range(3):
                X[j, i, c] = X[j, i - 1, c] + sigma * np.random.standard_normal()
    return X


def test_free():
    cfg = g.GoodsellConfig()
    D, dt = cfg.D_sugar, cfg.dt_s
    X = _free_paths(1, 4000, 200, math.sqrt(2 * D * dt))
    inc = np.diff(X, axis=1)
    var_ratio = inc.var() / (2 * D * dt)
    lags = [1, 10, 100, 200]
    msd = {k: float(((X[:, k] - X[:, 0]) ** 2).sum(1).mean() / (6 * D * k * dt)) for k in lags}
    print(f"free: increment var / (2 D dt) = {var_ratio:.4f}; MSD / (6 D t) at "
          + ", ".join(f"{k} steps: {v:.3f}" for k, v in msd.items()))
    report("free", dict(increment_var_ratio=var_ratio, msd_ratio=msd))


# ------------------------------------------------------------------ walls

@njit(cache=True)
def _inside(x, y, z, R, h):
    if abs(y) <= h:
        return x * x + z * z <= R * R * (1 + 1e-12)
    dy = abs(y) - h
    return x * x + dy * dy + z * z <= R * R * (1 + 1e-12)


@njit(cache=True)
def _dist_to_wall(x, y, z, R, h):
    if abs(y) <= h:
        return R - math.sqrt(x * x + z * z)
    dy = abs(y) - h
    return R - math.sqrt(x * x + dy * dy + z * z)


@njit(parallel=True, cache=True)
def _relax(X0, sigma, n_steps, R, h, seed):
    X = X0.copy()
    bad = np.zeros(len(X), np.int64)
    for j in prange(len(X)):
        np.random.seed(seed + j)
        x, y, z = X[j, 0], X[j, 1], X[j, 2]
        for _ in range(n_steps):
            x, y, z = g._reflect(x + sigma * np.random.standard_normal(),
                                 y + sigma * np.random.standard_normal(),
                                 z + sigma * np.random.standard_normal(), R, h)
            if not _inside(x, y, z, R, h):
                bad[j] += 1
        X[j] = (x, y, z)
    return X, bad


def uniform_in_capsule(n, R, h, rng):
    out = []
    while sum(len(o) for o in out) < n:
        p = rng.uniform([-R, -h - R, -R], [R, h + R, R], size=(n, 3))
        dy = np.maximum(np.abs(p[:, 1]) - h, 0)
        ok = p[:, 0] ** 2 + dy ** 2 + p[:, 2] ** 2 <= R * R
        out.append(p[ok])
    return np.vstack(out)[:n]


def capsule_volume_within(d, R, h):
    """Volume of the capsule (radius R, cylinder half-length h) shrunk by d."""
    r = R - d
    return math.pi * r * r * 2 * h + 4 / 3 * math.pi * r ** 3


def test_walls():
    cfg = g.GoodsellConfig()
    Rp, Rs, h, _ = g.geometry(cfg)
    res = {}
    # sugar (largest steps) at dt, dt/4, dt/16; protein at dt
    for particle, R, D, rng_seed, relax_seed, fs in (
            ("sugar", Rs, cfg.D_sugar, 5, 1000, (1, 4, 16)),
            ("protein", Rp, cfg.D_protein, 9, 4242, (1,))):
        res[particle] = _walls_one(cfg, particle, R, h, D, rng_seed, relax_seed, fs)
    report("walls", res)


def _walls_one(cfg, particle, R, h, D, rng_seed, relax_seed, fs):
    rng = np.random.default_rng(rng_seed)
    n = 400_000
    X0 = uniform_in_capsule(n, R, h, rng)
    edges_d = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 250, R * 1e3]) * 1e-3   # um from wall
    edges_y = np.linspace(-(h + R), h + R, 21)
    out = {}
    for label, f in (("dt", 1), ("dt/4", 4), ("dt/16", 16)):
        if f not in fs:
            continue
        dt = cfg.dt_s / f
        steps = min(20_000 * f, 80_000)          # 0.29 ms at dt and dt/4, 0.07 ms at dt/16
        t_phys = steps * dt
        X, bad = _relax(X0, math.sqrt(2 * D * dt), steps, R, h, relax_seed * f)
        dist = np.array([_dist_to_wall(*p, R, h) for p in X])
        cnt_d, _ = np.histogram(dist, edges_d)
        vol = np.array([capsule_volume_within(a, R, h) - capsule_volume_within(b, R, h)
                        for a, b in zip(edges_d[:-1], edges_d[1:])])
        exp_d = n * vol / capsule_volume_within(0, R, h)
        cnt_y, _ = np.histogram(X[:, 1], edges_y)
        # expected axial occupancy: cross-section area integrated over each bin
        yy = np.linspace(-(h + R), h + R, 200_001)
        area = np.pi * np.clip(R * R - np.maximum(np.abs(yy) - h, 0) ** 2, 0, None)
        cum = np.concatenate([[0], np.cumsum((area[1:] + area[:-1]) / 2 * np.diff(yy))])
        vol_y = np.diff(np.interp(edges_y, yy, cum))
        exp_y = n * vol_y / vol_y.sum()
        z_d = (cnt_d - exp_d) / np.sqrt(exp_d)
        z_y = (cnt_y - exp_y) / np.sqrt(exp_y)
        out[label] = dict(dt_ns=dt * 1e9, steps=steps, t_ms=t_phys * 1e3,
                          escapes=int(bad.sum()),
                          wall_bins_nm=[f"{a * 1e3:g}-{b * 1e3:g}" for a, b in
                                        zip(edges_d[:-1], edges_d[1:])],
                          wall_ratio=(cnt_d / exp_d).round(4).tolist(),
                          wall_z=z_d.round(2).tolist(),
                          axial_ratio=(cnt_y / exp_y).round(4).tolist(),
                          axial_z=z_y.round(2).tolist(),
                          chi2_wall=float((z_d ** 2).sum()), dof_wall=len(z_d),
                          chi2_axial=float((z_y ** 2).sum()), dof_axial=len(z_y))
        print(f"walls, {particle}, {label}: {steps} steps ({t_phys * 1e3:.2f} ms), escapes {bad.sum()}; "
              f"wall-distance bins occupancy/expected "
              + " ".join(f"{r:.3f}" for r in cnt_d / exp_d)
              + f"; chi2 {out[label]['chi2_wall']:.1f}/{len(z_d)}, "
              f"axial chi2 {out[label]['chi2_axial']:.1f}/{len(z_y)}")
    return dict(n=n, runs=out)


# ------------------------------------------------------------------ benchmark

def exact_cdf(t, a, r0, D):
    return a / r0 * np.array([math.erfc((r0 - a) / math.sqrt(4 * D * x)) for x in t])


def test_benchmark(n=100_000, Ls=(0, 2, 4, 6, 8, 10)):
    cfg = g.GoodsellConfig()
    a = cfg.contact_nm * 1e-3
    r0 = 2 * a
    Drel = cfg.D_protein + cfg.D_sugar
    n_steps = 2000
    t0 = time.time()
    T, M = g.ensemble(cfg, np.arange(n) + 10_000_000, Ls=Ls, walls=False,
                      starts=(r0 / 2, -r0 / 2),
                      n_max=n_steps)
    ks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    tt = ks * cfg.dt_s
    ex = exact_cdf(tt, a, r0, Drel)
    rows = {}
    print(f"benchmark: free space, r0 = 2a, {n} runs, {time.time() - t0:.0f} s. "
          "P(T <= t): exact | L=... (binomial SE ~ " f"{math.sqrt(0.25 / n):.4f})")
    print("   t/dt   exact  " + "  ".join(f"L={L:<3d}" for L in Ls))
    for k, e in zip(ks, ex):
        tk = k * cfg.dt_s * (1 + 1e-12)
        p = [float((M[:, j] & (T[:, j] <= tk)).mean()) for j in range(len(Ls))]
        rows[int(k)] = dict(exact=float(e), sim=p)
        print(f"  {k:5d}  {e:.4f}  " + "  ".join(f"{x:.4f}" for x in p))
    report("benchmark", dict(n=n, r0_over_a=2.0, Ls=list(Ls), rows=rows,
                             exact_eventual=0.5))


# ------------------------------------------------------------------ capsule problem

def summarize(t, met, t_max):
    n, nm = len(t), int(met.sum())
    tc = np.where(met, t, t_max)
    # restricted mean up to t_max (area under the Kaplan-Meier curve; no censoring
    # before t_max here, so it is the mean of min(T, t_max))
    rmst = float(tc.mean())
    se = float(tc.std(ddof=1) / math.sqrt(n))
    med = float(np.median(tc)) if nm > n / 2 else float("nan")
    return dict(n=n, met=nm, censored=n - nm, restricted_mean_s=rmst, se_s=se,
                median_s=med, mean_of_met_s=float(t[met].mean()) if nm else float("nan"))


def small_target_estimate(cfg):
    Rp, Rs, h, _ = g.geometry(cfg)
    Rc = cfg.cell_diameter_um / 2
    V = math.pi * Rc ** 2 * (cfg.cell_length_um - 2 * Rc) + 4 / 3 * math.pi * Rc ** 3
    return V / (4 * math.pi * (cfg.D_protein + cfg.D_sugar) * cfg.contact_nm * 1e-3)


def test_capsule(n=512, Ls=(0, 2, 4, 6, 8, 10)):
    """The Figure 1 problem. (A) opposite-pole starts, paired over refinement depth L;
    (B) an independent set of opposite-pole starts; (C) uniform random starts."""
    cfg = g.GoodsellConfig()
    est = small_target_estimate(cfg)
    res = dict(t_max_s=cfg.t_max_s, small_target_estimate_s=est)
    T, M = g.ensemble(cfg, np.arange(n) + 5000, Ls=Ls)
    rows = {}
    print(f"capsule A: {n} runs (seeds 5000-{5000 + n - 1}), opposite poles; "
          f"small-target estimate {est:.3f} s")
    for j, L in enumerate(Ls):
        s = summarize(T[:, j], M[:, j], cfg.t_max_s)
        if j + 1 < len(Ls):
            d = np.where(M[:, j], T[:, j], cfg.t_max_s) - np.where(M[:, j + 1], T[:, j + 1],
                                                                    cfg.t_max_s)
            s["paired_diff_to_next_s"] = float(d.mean())
            s["paired_diff_se_s"] = float(d.std(ddof=1) / math.sqrt(n))
            s["runs_changed_to_next"] = int((np.abs(d) > 1e-6).sum())   # > 1 us
        rows[int(L)] = s
        extra = (f"  next L: {s['paired_diff_to_next_s']:+.4f} +- {s['paired_diff_se_s']:.4f} s, "
                 f"{s['runs_changed_to_next']} runs change by > 1 us" if j + 1 < len(Ls) else "")
        print(f"  L={L:<2d} mean {s['restricted_mean_s']:.4f} +- {s['se_s']:.4f} s, median "
              f"{s['median_s']:.4f}, censored {s['censored']}{extra}")
    res["A_opposite_poles_paired"] = dict(seeds=[5000, 5000 + n - 1], Ls=list(Ls), rows=rows)
    tA = np.where(M[:, -1], T[:, -1], cfg.t_max_s)
    TB, MB = g.ensemble(cfg, np.arange(n) + 200000)
    sB = summarize(TB[:, 0], MB[:, 0], cfg.t_max_s)
    res["B_opposite_poles"] = dict(seeds=[200000, 200000 + n - 1], **sB)
    print(f"capsule B: {n} runs (seeds 200000-), opposite poles: mean {sB['restricted_mean_s']:.4f} "
          f"+- {sB['se_s']:.4f} s")
    Rp, Rs, h, _ = g.geometry(cfg)
    a = cfg.contact_nm * 1e-3
    rng = np.random.default_rng(33)
    P0 = uniform_in_capsule(3 * n, Rp, h, rng)
    S0 = uniform_in_capsule(3 * n, Rs, h, rng)
    ok = np.linalg.norm(P0 - S0, axis=1) > 2 * a
    TC, MC = g.ensemble(cfg, np.arange(n) + 300000, starts=(P0[ok][:n], S0[ok][:n]))
    sC = summarize(TC[:, 0], MC[:, 0], cfg.t_max_s)
    res["C_uniform_starts"] = dict(seeds=[300000, 300000 + n - 1], **sC)
    print(f"capsule C: {n} runs, uniform random starts: mean {sC['restricted_mean_s']:.4f} +- "
          f"{sC['se_s']:.4f} s")
    tAB = np.concatenate([tA, np.where(MB[:, 0], TB[:, 0], cfg.t_max_s)])
    pooled = dict(n=len(tAB), mean_s=float(tAB.mean()),
                  se_s=float(tAB.std(ddof=1) / math.sqrt(len(tAB))),
                  median_s=float(np.median(tAB)))
    res["pooled_opposite_poles"] = pooled
    tf = tAB
    grid = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    res["survival_opposite_poles"] = {x: dict(sim=float((tf > x).mean()),
                                              exp_model=math.exp(-x / est)) for x in grid}
    print(f"capsule pooled A+B (opposite poles, L={Ls[-1]}): mean {pooled['mean_s']:.4f} +- "
          f"{pooled['se_s']:.4f} s, median {pooled['median_s']:.4f} s")
    report("capsule", res)


def test_convergence():
    """Paired refinement convergence on the illustration seeds 0-63 (also used by --find-seed)."""
    cfg = g.GoodsellConfig()
    Ls = (0, 2, 4, 6, 8, 10)
    T, M = g.ensemble(cfg, np.arange(64), Ls=Ls)
    rows = {int(L): summarize(T[:, j], M[:, j], cfg.t_max_s) for j, L in enumerate(Ls)}
    changed = {int(L): int((np.abs(np.where(M[:, j], T[:, j], cfg.t_max_s)
                                   - np.where(M[:, -1], T[:, -1], cfg.t_max_s)) > 1e-6).sum())
               for j, L in enumerate(Ls)}                        # runs that move by > 1 us
    seed29 = {int(L): float(T[29, j]) for j, L in enumerate(Ls)}
    print("convergence (seeds 0-63): restricted mean by L: "
          + ", ".join(f"L={L}: {r['restricted_mean_s']:.4f}" for L, r in rows.items())
          + f"; runs whose time differs from L={Ls[-1]} by > 1 us: {changed}; seed 29: {seed29}")
    report("convergence", dict(Ls=list(Ls), rows=rows, runs_differing_from_finest=changed,
                               seed29=seed29))


# ------------------------------------------------------------------ display independence

def test_display():
    base = g.GoodsellConfig()
    res = {}
    for px in (64, 32, 128):
        cfg = replace(base, pixels_across=px)
        P, S, t = g.simulate(cfg, 29)
        sep = float(np.linalg.norm(S[-1] - P[-1]))
        res[px] = dict(t_event_s=float(t[-1]), final_separation_nm=sep * 1e3)
    a = base.contact_nm
    same = len({round(v["t_event_s"], 15) for v in res.values()}) == 1
    print(f"display: event time for pixels_across 64/32/128: "
          + ", ".join(f"{v['t_event_s']:.9f}" for v in res.values())
          + f" (identical: {same}); final separation "
          + ", ".join(f"{v['final_separation_nm']:.9f}" for v in res.values())
          + f" nm (encounter distance {a} nm)")
    report("display", dict(runs=res, identical=same, encounter_distance_nm=a))


# ------------------------------------------------------------------ confined benchmarks

D_REL = g.GoodsellConfig().D_protein + g.GoodsellConfig().D_sugar


def small_sphere(**kw):
    return replace(g.GoodsellConfig(), cell_diameter_um=0.5, cell_length_um=0.5, **kw)


def _mean_se(T, M, t_max):
    tc = np.where(M, T, t_max)
    return float(tc.mean()), float(tc.std(ddof=1) / math.sqrt(len(tc))), int((~M).sum())


def test_sphere_exact(n=4000):
    """Fixed target at the centre of a reflecting sphere (R = 0.25 um), searcher from
    r0 = 0.15 um: exact T(r0) = (a^2 - r0^2)/(6D) + R^3/(3D) (1/a - 1/r0)."""
    rows = {}
    for f in (1, 4):
        cfg = small_sphere(D_protein_um2s=1e-9, D_sugar_um2s=D_REL, t_max_s=2.0,
                           dt_override_s=12.6e-9 / f)
        Rp, Rs, h, _ = g.geometry(cfg)
        a, D, r0 = cfg.contact_nm * 1e-3, cfg.D_sugar + cfg.D_protein, 0.15
        exact = (a * a - r0 * r0) / (6 * D) + Rs ** 3 / (3 * D) * (1 / a - 1 / r0)
        T, M = g.ensemble(cfg, np.arange(n) + 90000, Ls=[0, 10], starts=(0.0, -r0))
        for j, L in enumerate((0, 10)):
            m, se, c = _mean_se(T[:, j], M[:, j], cfg.t_max_s)
            rows[f"dt/{f} L={L}"] = dict(mean_s=m, se_s=se, censored=c, exact_s=exact,
                                         ratio=m / exact, ratio_se=se / exact)
            print(f"sphere exact, dt/{f}, L={L}: mean {m:.5f} +- {se:.5f} s, exact {exact:.5f} "
                  f"-> ratio {m / exact:.4f} +- {se / exact:.4f}")
    report("sphere_exact", dict(n=n, R_um=0.25, r0_um=0.15, rows=rows))


def test_split(n=2000):
    """Small sphere, starts at +-0.15 um: fixed or mobile protein at the same D_rel."""
    est = (4 / 3 * math.pi * 0.25 ** 3) / (4 * math.pi * D_REL * g.GoodsellConfig().contact_nm * 1e-3)
    base = g.GoodsellConfig()
    cases = [("protein fixed", dict(D_protein_um2s=1e-6, D_sugar_um2s=D_REL - 1e-6), 40000),
             ("real split", dict(D_protein_um2s=base.D_protein, D_sugar_um2s=base.D_sugar), 40000),
             ("equal split", dict(D_protein_um2s=D_REL / 2, D_sugar_um2s=D_REL / 2), 40000),
             ("sugar fixed", dict(D_protein_um2s=D_REL - 1e-6, D_sugar_um2s=1e-6,
                                  dt_override_s=12.6e-9), 40000),
             ("real split, equal steric radii", dict(D_protein_um2s=base.D_protein,
              D_sugar_um2s=base.D_sugar, protein_steric_nm=0.36, sugar_steric_nm=0.36,
              encounter_distance_nm=2.86), 40000),
             ("real split, other seeds", dict(D_protein_um2s=base.D_protein,
              D_sugar_um2s=base.D_sugar), 70000),
             ("real split, dt/4", dict(D_protein_um2s=base.D_protein, D_sugar_um2s=base.D_sugar,
                                       dt_override_s=base.dt_s / 4), 40000)]
    rows = {}
    for name, kw, s0 in cases:
        cfg = small_sphere(t_max_s=2.0, **kw)
        T, M = g.ensemble(cfg, np.arange(n) + s0, Ls=[10], starts=(0.15, -0.15))
        m, se, c = _mean_se(T[:, 0], M[:, 0], cfg.t_max_s)
        rows[name] = dict(D_protein=cfg.D_protein, D_sugar=cfg.D_sugar, mean_s=m, se_s=se,
                          censored=c, ratio=m / est, ratio_se=se / est)
        print(f"split, {name}: mean {m:.4f} +- {se:.4f} s, ratio to V/(4 pi D_rel a) "
              f"{m / est:.3f} +- {se / est:.3f}")
    report("split", dict(n=n, estimate_s=est, rows=rows))


def test_finite_size(n=4000):
    """Small sphere, both mobile, uniform random starts, encounter distance a and a/2:
    a finite-size (a/R) excess over V/(4 pi D_rel a) should shrink with a."""
    base = g.GoodsellConfig()
    rng = np.random.default_rng(21)
    V = 4 / 3 * math.pi * 0.25 ** 3
    rows = {}
    for enc in (2.86, 1.43):
        cfg = small_sphere(encounter_distance_nm=enc, dt_override_s=base.dt_s, t_max_s=4.0)
        Rp, Rs, h, _ = g.geometry(cfg)
        a = enc * 1e-3
        est = V / (4 * math.pi * D_REL * a)
        P0 = uniform_in_capsule(3 * n, Rp, h, rng)
        S0 = uniform_in_capsule(3 * n, Rs, h, rng)
        ok = np.linalg.norm(P0 - S0, axis=1) > 2 * a
        T, M = g.ensemble(cfg, np.arange(n) + 100000, Ls=[10], starts=(P0[ok][:n], S0[ok][:n]))
        m, se, c = _mean_se(T[:, 0], M[:, 0], cfg.t_max_s)
        rows[f"a={enc} nm"] = dict(a_over_R=a / 0.25, mean_s=m, se_s=se, censored=c,
                                   estimate_s=est, excess_pct=100 * (m / est - 1),
                                   excess_se_pct=100 * se / est)
        print(f"finite size, a = {enc} nm (a/R {a / 0.25:.4f}): mean {m:.5f} +- {se:.5f} s, "
              f"excess over estimate {100 * (m / est - 1):+.1f} +- {100 * se / est:.1f} %")
    report("finite_size", dict(n=n, R_um=0.25, rows=rows))


@njit(parallel=True, cache=True)
def _pair_close(P0, S0, sp, ss, n_steps, every, Rp, Rs, h, rc, seed):
    cnt = np.zeros(len(P0), np.int64)
    tot = np.zeros(len(P0), np.int64)
    for j in prange(len(P0)):
        np.random.seed(seed + j)
        px, py, pz = P0[j, 0], P0[j, 1], P0[j, 2]
        sx, sy, sz = S0[j, 0], S0[j, 1], S0[j, 2]
        for i in range(1, n_steps + 1):
            px, py, pz = g._reflect(px + sp * np.random.standard_normal(),
                                    py + sp * np.random.standard_normal(),
                                    pz + sp * np.random.standard_normal(), Rp, h)
            sx, sy, sz = g._reflect(sx + ss * np.random.standard_normal(),
                                    sy + ss * np.random.standard_normal(),
                                    sz + ss * np.random.standard_normal(), Rs, h)
            if i % every == 0:
                tot[j] += 1
                if (px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2 < rc * rc:
                    cnt[j] += 1
    return cnt.sum(), tot.sum()


def test_pairs():
    """No reactions: two walkers from independent uniform starts keep the equilibrium
    probability of being within 20 nm (vs independent uniform sampling)."""
    cfg = small_sphere()
    Rp, Rs, h, _ = g.geometry(cfg)
    rng = np.random.default_rng(12)
    rc = 0.020
    hits = tot = 0
    for _ in range(10):
        A = uniform_in_capsule(4_000_000, Rp, h, rng)
        B = uniform_in_capsule(4_000_000, Rs, h, rng)
        hits += int((((A - B) ** 2).sum(1) < rc * rc).sum())
        tot += len(A)
    ref = hits / tot
    P0 = uniform_in_capsule(16000, Rp, h, rng)
    S0 = uniform_in_capsule(16000, Rs, h, rng)
    rows = {}
    for name, Dp, Ds in (("both mobile", cfg.D_protein, cfg.D_sugar),
                         ("protein fixed", 0.0, D_REL)):
        c, t = _pair_close(P0, S0, math.sqrt(2 * Dp * cfg.dt_s), math.sqrt(2 * Ds * cfg.dt_s),
                           200_000, 100, Rp, Rs, h, rc, 999)
        rows[name] = dict(p=c / t, n=int(c), ratio=c / t / ref)
        print(f"pairs, {name}: P(|r| < 20 nm) {c / t:.4e} vs independent uniform {ref:.4e} "
              f"-> ratio {c / t / ref:.3f}")
    report("pairs", dict(r_nm=20, static_p=ref, static_n=hits, rows=rows))


TESTS = dict(rng=test_rng, free=test_free, walls=test_walls, benchmark=test_benchmark,
             sphere_exact=test_sphere_exact, split=test_split, finite_size=test_finite_size,
             pairs=test_pairs, convergence=test_convergence, capsule=test_capsule,
             display=test_display)

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    f = OUT / "results.json"
    if f.exists():
        RES.update(json.loads(f.read_text()))
    for name in (sys.argv[1:] or TESTS):
        TESTS[name]()
