# Simulation of Motor-Enhanced Random Motion (MERM)

Source code for the simulations in the Perspective *Directed, Brownian, and Motor-Enhanced Random Motion: Three Facets of Intracellular Transport* (Kim, Ysaguirre, Hoffman, Lyles & Macosko, PNAS).

Motor-enhanced random motion is a form of intracellular transport in which (1) particles interact with a mesh of cytoskeletal and other intracellular elements, (2) they move by hydrodynamic coupling with the mesh, by hopping from trap to trap, or by random movement of the traps themselves, and (3) the movement depends on cytoskeletal motors and cellular energy. `simulation.py` is a simple model of particles (here, VSV ribonucleoprotein particles, RNPs) that alternate between motor-enhanced random motion (trap hopping and moving traps) and directed motion (radial runs toward or away from the nucleus, as if along microtubules), in a 2D cell 15 µm in radius with a 5 µm nucleus. See `sim-paper.pdf` for the full description and results.

## Quick start

```
./setup.sh                 # creates venv/ and installs requirements.txt
source venv/bin/activate
python3 run.py --n_particles 20
```

This opens a live window showing the particles moving in the cell, with Play/Pause, Reset, trail and CSV-export controls.

To install by hand instead: `pip install -r requirements.txt` (numpy, scipy, matplotlib, numba, tqdm, PySide6, ipykernel).

## Reproducing the paper's result

The Perspective reports that, for particles modeled on VSV RNPs, directed motion contributes about 16.4% of the total distance traveled and motor-enhanced random motion about 83.6%. That run used 1000 particles started 7.5 µm from the center of the cell, followed for 6000 s:

```
python3 run.py --headless --compute_flux --no_csv --n_particles 1000 --total_time 6000
```

It prints the average driven and diffusive (trap) distances and the driven fraction, and saves a histogram to `sim/`. Expect it to take many hours; fewer particles give the same fraction with a larger uncertainty.

## Files

- `run.py`: command-line entry point (`python3 run.py --help` lists all options). Without `--headless` it opens the visualizer; with `--headless` it writes coordinates to `sim/coords.csv`.
- `simulation.py`: the simulation itself (calculation only; no graphics).
- `visualize.py`: the PySide6 visualizer used by `run.py`.
- `analysis.py`: analysis helpers (flux, trap sizes, displacement vs. time).
- `msd.py`: legacy mean-squared-displacement analysis; it contains hardcoded file paths from the original author's machine, so edit those before use.
- `combine-tifs.ijm`, `combine-mega-tif.ijm`: ImageJ scripts that combine TIF frame stacks. The output TIF files are large, so put all files to be combined into one folder and run `combine-tifs` on that folder; stacks named `{NAME}{NUM}` are combined in `{NUM}` order.
- `goodsell.py`, `validate_goodsell.py`: Figure 1 (see below).
- `sim-paper.pdf`: write-up of the simulation and its results.

## License

MIT; see `LICENSE`.

## `goodsell.py`: diffusion in a bacterium (Figure 1)

`goodsell.py` reproduces Figure 1 of the Perspective, a Brownian dynamics simulation revisiting an in silico experiment by David Goodsell (*The Machinery of Life*, 2nd ed., p. 6).
A sugar (glucose, red) and a typical protein (GFP-sized, blue) start at opposite poles of a 1 × 2 µm bacterium and diffuse until they first meet.

```
python goodsell.py                 # -> goodsell_output/goodsell_fig1.png/.svg/.pdf (~1 min)
python goodsell.py --no-labels     # same figure without text
python goodsell.py --find-seed     # redo the seed choice from 64 runs (~2 min)
python validate_goodsell.py        # numerical validation (~2 h on 10 cores)
```

Requires `numpy`, `matplotlib`, `numba`.

### Model and assumptions

- **Model:** two spherical particles with constant diffusion coefficients in a homogeneous medium inside a reflecting capsule (1 µm wide, 2 µm long).
- **Parameters** (from the Figure 1 caption): viscosity 20 mPa·s (29× water at 37 °C), 37 °C.
- **Diffusion coefficients** come from Stokes–Einstein with hydrodynamic radii of 2.5 nm for the protein and 0.36 nm for glucose. That gives D = 4.54 and 31.55 µm²/s, which can also be set directly.
  - **GFP:** 87 µm²/s at room temperature (Swaminathan et al. 1997, *Biophys J* 72:1900) gives 2.5–2.8 nm. 83.4 µm²/s at 20 °C by analytical ultracentrifugation (Vámosi et al. 2016, *Sci Rep* 6:33022) gives 2.57 nm.
  - **Glucose:** 0.36 nm (Ribeiro et al. 2006, *J Chem Eng Data* 51:1836), derived from its measured D by Stokes–Einstein.
- **One effective viscosity for both molecules is an assumption, not a measured property of bacterial cytoplasm.** GFP in *E. coli* diffuses ~11× slower than in water, and no single effective viscosity describes all proteins (Elowitz et al. 1999, *J Bacteriol* 181:197). Crowding, specific interactions, hydrodynamic coupling and orientation-dependent binding are not represented.
- **"Encounter" is geometric:** the centres come within 2.86 nm (the encounter distance, by default the sum of the steric radii). It is not necessarily binding.
- **Radii are separate settings:**
  - hydrodynamic radii: used for Stokes–Einstein;
  - steric radii: wall exclusion;
  - encounter distance: termination.

### Numerics

- **Free diffusion:** Gaussian steps with σ = √(2 D dt) per coordinate, base step dt = 14.4 ns.
- **Walls:** a step that ends outside the accessible capsule is mirrored back across the surface, an approximation on curved walls.
- **Encounters:** checking only the straight line between steps misses contacts the Brownian path makes within a step. Our first version did this, and missed ~21% of encounters in the benchmark below. Now:
  - Steps whose straight-line approach comes within a + 8 √(2 D_rel dt) of contact are refined by Brownian-bridge bisection down to dt/2¹⁰.
  - Midpoints are drawn exactly from the bridge law, with variance D·Δt/2 per coordinate, from a separate counter-based random stream, so the base trajectory is identical at every refinement depth.
  - The first contact is then located on the finest intervals, which gives the event time and both positions.
  - Remaining approximations:
    - straight-line contact on the finest intervals (measured below);
    - free bridges within the 0.2% of refined steps that also reflect off a wall.
- **Censoring:** runs with no encounter by 10 s are reported as censored (none occurred).

### Validation (`validate_goodsell.py`; output in `goodsell_validation/`)

| Test | Result |
|---|---|
| Bridge random stream | normals: mean −0.001, variance 0.999, kurtosis 2.997; bisection to dt/2⁸ reproduces free increments (variance 0.9999 × 2 D Δt, lag-1 correlation 0.002) |
| Free diffusion | increment variance 1.001 × 2 D dt; MSD 0.99–1.00 × 6 D t |
| Walls (uniform start must stay uniform) | 0 escapes out of 400,000; occupancy by distance to the wall (bins from 1 nm) and along the axis (including the cylinder–cap joins) is uniform within noise for the sugar at dt, dt/4, dt/16 and for the protein at dt (χ² ≈ dof) |
| Free space, exact first passage, r₀ = 2a: P(T ≤ 2000 dt), exact 0.4750 | straight-line test only 0.3773 (−21%); refined, L = 2/4/6/8/10: 0.426/0.451/0.464/0.470/**0.473** (error halves every 2 levels) |
| Sphere R = 0.25 µm, fixed central target, exact mean 0.0492 s | straight-line test only: +26% (dt), +12% (dt/4); refined: **+2.1 ± 1.6%** (dt), **+1.6 ± 1.6%** (dt/4) |
| Split of D_rel (sphere R = 0.25 µm, starts ±0.15 µm on the axis, same D_p + D_s) | one mobile particle: 0.96–0.99 × V/(4π D_rel a); both mobile: 1.05–1.09 ×, unchanged at dt/4 or with equal steric radii. Consistent with the finite-size row below, and with the larger excess for mobile targets suggested by Lawley & Miles 2019, Fig. 1a |
| Finite size (sphere R = 0.25 µm, both mobile, uniform starts) | excess over V/(4π D_rel a): +4.3 ± 1.6% at a/R = 0.011, −0.9 ± 1.5% at a/R = 0.006. It shrinks with a, as expected for a finite-size correction (leading order: Lawley & Miles 2019, *J Nonlinear Sci* 29:2955) |
| Survival curves over refinement depth (capsule set A) | 5 of 512 runs change by more than 1 µs between L = 8 and 10, so the curves agree within 1% at every t |
| Censoring | exercised by the free-space benchmark, where ~52% of runs never meet within the horizon; they are counted, not dropped |
| Display independence | event time and final separation (2.860000 nm) identical for display grids of 32, 64, 128 pixels |

### Mean time to first encounter in the cell

Theory for well-mixed particles (leading order): mean = V / (4π (D_p + D_s) a) = **1.01 s**.
This is also the diffusion-limited estimate in [*Cell Biology by the Numbers*](https://book.bionumbers.org/how-many-reactions-do-enzymes-carry-out-each-second/): ~10⁹ M⁻¹s⁻¹ × ~1 nM ≈ 1 s⁻¹, "they will meet within a second on average". Here k_on = 7.8 × 10⁸ M⁻¹s⁻¹, and one molecule in 1.31 µm³ is 1.27 nM.

| Simulation (L = 10, 512 runs each, none censored) | mean ± SE |
|---|---|
| uniform random starts | **1.05 ± 0.05 s** |
| opposite-pole starts, set A (seeds 5000–5511) | 1.21 ± 0.05 s |
| opposite-pole starts, set B (seeds 200000–200511) | 1.06 ± 0.05 s |
| opposite-pole starts, A + B pooled | **1.14 ± 0.03 s** (median 0.80 s) |
| same as set A, straight-line test only (first version) | 1.39 ± 0.06 s |

- **Uniform starts:** the simulated mean agrees with theory.
- **Opposite-pole starts:** the pooled mean is 0.09 ± 0.06 s (1.5σ) above the uniform-start mean. A start lag of that size is plausible for molecules starting 1.8 µm apart, but it was not measured separately. Sets A and B differ by 2.1σ.
- **Waiting times are broad**, roughly exponential, so the median is ~0.7 × the mean.
- **Refinement convergence on set A:** going from L = 8 to L = 10 changes the mean by +0.007 ± 0.004 s.

### Illustration choices (they do not change the physics)

- The panels show 5, 10 and 15 ms, times chosen for illustration, then the complete paths up to the first encounter.
- **Seed 29:** of seeds 0–63, the run whose first encounter (1.001 s) is closest to Goodsell's "about a second" (`--find-seed`).
  - It is an illustration, not evidence for the mean.
  - The corrected method selects the same seed and draws the same figure.
- Paths are sampled every ~3.9 µs, projected onto the page and snapped to a grid 64 pixels across the cell.
  - They are sampled projections, not molecular-scale resolved paths.
  - Paths can cross on the page without the molecules touching.
