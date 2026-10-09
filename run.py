import math
import sys

import argparse
import csv
import numpy as np
import os
from PySide6.QtWidgets import QApplication
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
import simulation as sim
import visualize as vis
from analysis import plot_displacement_vs_time, plot_trap_size_histogram, plot_trap_distance_histogram, compute_trap_sizes_from_simulation


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Particle Simulation Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        '--total_time', type=int, default=2000,
        help='Maximum simulation time (seconds)'
    )
    sim_group.add_argument(
        '--n_particles', type=int, default=1,
        help='Number of particles to simulate'
    )
    sim_group.add_argument(
        '--p_driv', type=float, default=0.03,
        help='Probability of driven motion (0.0-1.0)'
    )
    sim_group.add_argument(
        '--trap_dist', type=str, default="default",
        help=f'Distance between traps (meters) (default: {sim.TRAP_DIST})'
    )
    sim_group.add_argument(
        '--time_between', type=float, default=sim.TIME_BETWEEN_STATES,
        help=f'Average time between state changes (seconds) (default: {sim.TIME_BETWEEN_STATES})'
    )
    sim_group.add_argument(
        '--dt', type=float, default=0.001,
        help='Length of time step for simulation (seconds)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--dirname', type=str, default='sim',
        help='Directory name for output files'
    )
    output_group.add_argument(
        '--record_frames', action='store_true',
        help='Automatically start recording frames on startup'
    )
    output_group.add_argument(
        '--no_csv', action='store_true',
        help='Skip writing CSV output file in headless mode'
    )

    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument(
        '--width', type=int, default=600,
        help='Canvas width in pixels'
    )
    display_group.add_argument(
        '--height', type=int, default=600,
        help='Canvas height in pixels'
    )

    # Special modes
    mode_group = parser.add_argument_group('Special Modes')
    mode_group.add_argument(
        '--headless', action='store_true',
        help='Run simulation without GUI (exports data only)', default=False
    )
    mode_group.add_argument(
        '--compute_flux', action='store_true',
        help='Compute and plot flux distribution for particles', default=False
    )
    mode_group.add_argument(
        '--plot_trap_distances', action='store_true',
        help='Plot histogram of distances between consecutive traps', default=False
    )
    mode_group.add_argument(
        '--plot_trap_sizes', action='store_true',
        help='Plot histogram of trap sizes', default=False
    )
    mode_group.add_argument(
        '--histogram_bins', type=int, default=20,
        help='Number of bins for histograms (default: 20)'
    )

    # Histogram options for trap sizes
    hist_opts = parser.add_argument_group('Trap Size Histogram Options')
    hist_opts.add_argument(
        '--trap_sizes_log', action='store_true',
        help='Use log scale on x-axis for trap size histogram', default=False
    )
    hist_opts.add_argument(
        '--trap_sizes_robust', action='store_true',
        help='Use percentile-based robust x-range to reduce outlier impact', default=False
    )
    hist_opts.add_argument(
        '--trap_sizes_lower_pct', type=float, default=1.0,
        help='Lower percentile for robust range (default: 1.0)'
    )
    hist_opts.add_argument(
        '--trap_sizes_upper_pct', type=float, default=99.0,
        help='Upper percentile for robust range (default: 99.0)'
    )
    hist_opts.add_argument(
        '--trap_sizes_winsorize', action='store_true',
        help='Winsorize data to the robust percentile range before binning', default=False
    )

    # Help and examples
    parser.epilog = """
Examples:
  %(prog)s                                    # Run with default settings
  %(prog)s --n_particles 5 --total_time 5000 # Simulate 5 particles for 5000 steps
  %(prog)s --p_driv 0.1 --dirname results     # Higher drive probability, custom output
  %(prog)s --width 800 --height 800          # Larger display window
  %(prog)s --headless --dirname run          # Run without GUI
  %(prog)s --headless --plot_trap_distances --no_csv  # Plot trap distances histogram
  %(prog)s --headless --plot_trap_sizes --histogram_bins 30 --no_csv  # Plot trap sizes with 30 bins
    """

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """Validate command line arguments"""
    errors = []

    # Validate ranges
    if args.total_time <= 0:
        errors.append("total_time must be positive")
    if args.n_particles <= 0:
        errors.append("n_particles must be positive")
    if not 0.0 <= args.p_driv <= 1.0:
        errors.append("p_driv must be between 0.0 and 1.0")
    if args.dt <= 0:
        errors.append("dt must be positive")
    if args.width <= 0 or args.height <= 0:
        errors.append("width and height must be positive")

    # Validate optional parameters
    # if args.trap_dist is not None and args.trap_dist <= 0:
    #     errors.append("trap_dist must be positive")
    if args.time_between is not None and args.time_between <= 0:
        errors.append("time_between must be positive")

    # Display errors
    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True

def run_headless_simulation(config: sim.SimulationConfig, write_csv: bool = True, verbose: bool = False) -> bool:
    """Run the simulation without a GUI and export results to CSV."""
    print(f"Running headless simulation with {config.n_particles} particles…")
    if write_csv:
        print(f"Output directory: {config.dirname}")

    try:
        if write_csv:
            os.makedirs(config.dirname, exist_ok=True)

        coords, exit_times = [], []
        max_n_steps = n_exited = 0

        if verbose:
            it = tqdm(range(config.n_particles),
                       desc="Particles",
                       unit="particle")
        else:
            it = range(config.n_particles)
        for i in it:
            x_coords, y_coords, exit_time, *_ = sim.move(
                config,
                theta=i * 2 * np.pi / config.n_particles
            )

            max_n_steps = max(len(x_coords), max_n_steps)
            if exit_time != -1:
                n_exited += 1
                exit_times.append(exit_time)

            coords.append([x_coords, y_coords])

        # Export to CSV
        if write_csv:
            csv_filename = os.path.join(config.dirname, "coords.csv")
            with open(csv_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "particle", "x", "y"])
                for p_idx, (xs, ys) in enumerate(coords):
                    for frame, (x, y) in enumerate(zip(xs, ys)):
                        writer.writerow([frame, p_idx, x, y])

        # Summary
        print("\nSimulation complete!")
        print("======================\n")
        print(f"Particles simulated : {len(coords)}")
        print(f"Particles exited    : {n_exited}")
        print(f"Maximum steps       : {max_n_steps}")
        if exit_times:
            print(f"Average exit time   : {np.mean(exit_times):.2e} s")
        if write_csv:
            print(f"\nData saved -> {csv_filename}")

        return True

    except Exception as e:
        print(f"Error running simulation: {e}")
        return False

def main():
    """Main application entry point"""
    # Parse and validate arguments
    args = parse_arguments()

    if not validate_arguments(args):
        sys.exit(1)

    # Create configuration
    config = sim.SimulationConfig.from_args(args)

    # Print configuration
    print(f"Particle Simulation Visualizer")
    print(f"==============================")
    print(f"Configuration:")
    print(f"  Number of particles: {config.n_particles}")
    print(f"  Total time: {config.total_time} s")
    print(f"  Probability of driven motion: {config.p_driv}")
    print(f"  Output directory: {config.dirname}")
    if args.headless:
        print(f"  Mode: Headless (no GUI)")
    else:
        print(f"  Display: {config.width}x{config.height}")
    print()

    # Run headless simulation if requested
    if args.headless:
        # Skip redundant simulation if only computing flux or histograms
        if (args.compute_flux or args.plot_trap_distances or args.plot_trap_sizes) and args.no_csv:
            if args.compute_flux:
                plot_flux_distribution(config)
            if args.plot_trap_distances:
                plot_trap_distances_from_simulation(config, bins=args.histogram_bins)
            if args.plot_trap_sizes:
                plot_trap_sizes_from_simulation(
                    config,
                    bins=args.histogram_bins,
                    robust=args.trap_sizes_robust,
                    lower_pct=args.trap_sizes_lower_pct,
                    upper_pct=args.trap_sizes_upper_pct,
                    log=args.trap_sizes_log,
                    winsorize=args.trap_sizes_winsorize,
                )
            sys.exit(0)
        else:
            success = run_headless_simulation(config, write_csv=not args.no_csv)
            if args.compute_flux:
                plot_flux_distribution(config)
            if args.plot_trap_distances:
                plot_trap_distances_from_simulation(config, bins=args.histogram_bins)
            if args.plot_trap_sizes:
                plot_trap_sizes_from_simulation(config, bins=args.histogram_bins)
            sys.exit(0 if success else 1)

    # Run GUI application
    app = QApplication(sys.argv)

    simulation = vis.SimulationVis(config)
    simulation.show()
    simulation.run_simulation()

    if args.compute_flux:
        plot_flux_distribution(config)
    if args.plot_trap_distances:
        plot_trap_distances_from_simulation(config, bins=args.histogram_bins)
    if args.plot_trap_sizes:
        plot_trap_sizes_from_simulation(
            config,
            bins=args.histogram_bins,
            robust=args.trap_sizes_robust,
            lower_pct=args.trap_sizes_lower_pct,
            upper_pct=args.trap_sizes_upper_pct,
            log=args.trap_sizes_log,
            winsorize=args.trap_sizes_winsorize,
        )

    sys.exit(app.exec())

def generate_displacement_time_driven_graph(n_particles = 50, dt = 0.001, total_time = 600):
    import numpy as np
    import matplotlib.pyplot as plt
    import simulation as sim

    do_analysis = True

    if do_analysis:

        # Create chart
        plt.figure()

        # Calculate and plot lines for each p_driv value on chart
        plot_displacement_vs_time(
            total_time  = total_time,
            n_particles = n_particles,
            p_driv_vals = [1,.50,.24,.12,.06,.03,0],
        )

        # Add labels, title, legend to chart
        plt.xlabel("time (s)")
        plt.ylabel("Displacement (µm)")
        plt.title(f"Displacement vs. Time for Varying Driven Motion Amounts")
        plt.legend()
        plt.xlim([0, math.ceil(total_time / 1000) * 1000]) # Round total time up to the nearest thousand
        plt.ylim([0, 50])

        # Show chart
        plt.show()


def plot_trap_distances_from_simulation(config, bins: int = 20):
    """
    Run a simulation and plot histogram of distances between consecutive traps.

    Parameters
    ----------
    config : SimulationConfig
        Configuration object with simulation parameters.
    bins : int
        Number of bins for the histogram.
    """
    import matplotlib.pyplot as plt
    print(f"\nRunning simulation to collect trap distance data...")
    print(f"Simulation time: {config.total_time} s")

    # Run simulation
    sim_output = sim.move(config, theta=0.0, stop_on_cell_exit=False)

    # Create histogram
    fig, ax = plot_trap_distance_histogram(sim_output, bins=bins)

    # Save figure
    svg_filename = os.path.join(config.dirname,
                                f"trap_distances_t{config.total_time}s.svg")
    os.makedirs(config.dirname, exist_ok=True)
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    print(f"Plot saved → {svg_filename}\n")

    plt.show()


def plot_trap_sizes_from_simulation(
    config,
    bins: int = 20,
    robust: bool = False,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    log: bool = False,
    winsorize: bool = False,
):
    """
    Run a simulation and plot histogram of trap sizes computed from the data.

    Parameters
    ----------
    config : SimulationConfig
        Configuration object with simulation parameters.
    bins : int
        Number of bins for the histogram.
    """
    import matplotlib.pyplot as plt
    print(f"\nRunning simulation to compute trap sizes...")
    print(f"Simulation time: {config.total_time} s")

    # Run simulation
    sim_output = sim.move(config, theta=0.0, stop_on_cell_exit=False)

    # Compute trap sizes from simulation data
    trap_sizes = compute_trap_sizes_from_simulation(sim_output, config)

    print(f"Computed {len(trap_sizes)} trap sizes")

    # Create histogram
    fig, ax = plot_trap_size_histogram(
        trap_sizes,
        bins=bins if bins > 0 else None,
        robust=robust,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        log=log,
        winsorize=winsorize,
    )

    # Save figure
    svg_filename = os.path.join(config.dirname,
                                f"trap_sizes_t{config.total_time}s.svg")
    os.makedirs(config.dirname, exist_ok=True)
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    print(f"Plot saved → {svg_filename}\n")

    plt.show()


def plot_flux_distribution(config, window_duration: int = 4):
    """
    Plot the distribution of 4‑second flux values in **micrometers** for every particle.

    Parameters
    ----------
    config : SimulationConfig
        Configuration object with simulation parameters (must include n_particles).
    snapshot_interval : float, optional
        Time step (s) between snapshots used when calling `compute_flux_4s`.
    window_duration : int, optional
        Window (s) over which diffusive flux is computed.
    diffusive_threshold : float, optional
        Threshold (m) used to decide whether a step counts toward diffusive flux.

    Notes
    -----
    * All flux values returned by the simulation are assumed to be in **meters**.
      They are converted to **micrometers** (× 1 000 000) before plotting.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from tqdm import tqdm

    import simulation as sim
    from analysis import compute_flux_4s

    # Accumulate flux values from every particle
    all_dist_diffusive, all_dist_driven = [], []
    all_displacements_4s = []  # Straight-line displacement per 4-s window
    total_windows = total_driven_windows = 0

    # Full run statistics
    exit_times = []
    total_displacements = []
    total_dist_driven = []
    total_dist_diffusive = []

    # ---- collect raw (unweighted) per-window distances ----
    for i in tqdm(range(config.n_particles), desc="Calculating 4-s distances", unit="particle"):
        theta = 2 * np.pi * i / config.n_particles
        sim_output = sim.move(config, theta=theta, stop_on_cell_exit=False)

        # Collect full run statistics
        if sim_output.exit_time != -1:
            exit_times.append(sim_output.exit_time)

        # Calculate total displacement
        final_displacement = np.sqrt(sim_output.x[-1]**2 + sim_output.y[-1]**2)
        total_displacements.append(final_displacement * 1e6)  # Convert to µm

        # Total distances in each state
        total_dist_driven.append(np.sum(sim_output.distance_driven) * 1e6)
        total_dist_diffusive.append(np.sum(sim_output.distance_trap) * 1e6)

        # get raw meters per window
        diff_d, driv_d, mask = compute_flux_4s(
            sim_output, config,
            window=window_duration, sample_dt=0.01,
            rate=False, return_mask=True
        )

        # Calculate displacement (straight-line distance) for each 4-s window
        t_end = (len(sim_output.x) - 1) * config.dt
        t_samples = np.round(np.arange(0.0, t_end + 1e-12, 0.01), 2)
        t_orig = np.arange(len(sim_output.x)) * config.dt
        x_samp = np.interp(t_samples, t_orig, sim_output.x)
        y_samp = np.interp(t_samples, t_orig, sim_output.y)

        pts_per_win = int(round(window_duration / 0.01))
        n_windows = (len(t_samples) - 1) // pts_per_win

        # Only include windows before particle exits
        exit_time = sim_output.exit_time

        for w in range(n_windows):
            s_idx = w * pts_per_win
            e_idx = s_idx + pts_per_win

            # Check if this window is before exit
            window_end_time = t_samples[e_idx]
            if exit_time != -1 and window_end_time > exit_time:
                break  # Skip remaining windows after exit

            # Include this window in statistics
            all_dist_diffusive.append(diff_d[w])
            all_dist_driven.append(driv_d[w])
            total_windows += 1
            if mask[w]:
                total_driven_windows += 1

            # Calculate displacement for this window
            dx = x_samp[e_idx] - x_samp[s_idx]
            dy = y_samp[e_idx] - y_samp[s_idx]
            displacement = np.hypot(dx, dy)
            all_displacements_4s.append(displacement)

    if total_windows == 0:
        print("\nNo 4-s windows found.")
        return

    # ---- Print full run statistics ----
    print("\n" + "="*60)
    print("FULL RUN STATISTICS")
    print("="*60)
    print(f"Simulation time          : {config.total_time} s")
    print(f"Number of particles      : {config.n_particles}")
    print(f"Driven motion probability: {config.p_driv}")
    print()
    print(f"Particles exited         : {len(exit_times)} ({100*len(exit_times)/config.n_particles:.1f}%)")
    if exit_times:
        print(f"Average exit time        : {np.mean(exit_times):.2f} s")
        print(f"Median exit time         : {np.median(exit_times):.2f} s")
    print()
    print(f"Average total displacement: {np.mean(total_displacements):.2f} µm")
    print(f"Median total displacement : {np.median(total_displacements):.2f} µm")
    print()
    print(f"Average driven distance   : {np.mean(total_dist_driven):.2f} µm")
    print(f"Average diffusive distance: {np.mean(total_dist_diffusive):.2f} µm")
    total_avg = np.mean(total_dist_driven) + np.mean(total_dist_diffusive)
    print(f"Average total distance    : {total_avg:.2f} µm")
    if total_avg > 0:
        print(f"  → Driven fraction       : {100*np.mean(total_dist_driven)/total_avg:.1f}%")
        print(f"  → Diffusive fraction    : {100*np.mean(total_dist_diffusive)/total_avg:.1f}%")
    print("="*60)
    print()

    # ---- measured mixture weights ----
    driv_weight = total_driven_windows / total_windows
    diff_weight = 1.0 - driv_weight
    print(f"\n4-s windows with ANY driven motion: {100*driv_weight:.2f}%"
          f"  ({total_driven_windows}/{total_windows})")

    # Convert meters -> micrometers and scale sample values ----
    diff_um = np.asarray(all_dist_diffusive, float) * 1e6 * diff_weight
    driv_um = np.asarray(all_dist_driven,    float) * 1e6 * driv_weight

    # ---- report means and weighted contributions (μm per 4 s) ----
    # (means before value scaling)
    mu_diff = (diff_um / max(diff_weight, 1e-12)).mean() if diff_um.size else 0.0
    mu_driv = (driv_um / max(driv_weight, 1e-12)).mean() if driv_um.size else 0.0
    phi_total = diff_weight * mu_diff + driv_weight * mu_driv
    print(f"\nΦ_diffusive (unweighted) ≈ {mu_diff:.3f} μm per 4 s")
    print(f"Φ_directed  (unweighted) ≈ {mu_driv:.3f} μm per 4 s")
    print(f"Φ_total (weighted)       ≈ {phi_total:.3f} μm per 4 s")

    # ---- report displacement statistics ----
    if all_displacements_4s:
        displacements_um = np.array(all_displacements_4s) * 1e6
        avg_displacement = np.mean(displacements_um)
        median_displacement = np.median(displacements_um)
        print(f"\nAverage displacement per 4-s window: {avg_displacement:.3f} μm")
        print(f"Median displacement per 4-s window:  {median_displacement:.3f} μm")

    # ---- prepare data for log-space area normalization ----
    diff_nz = diff_um[diff_um > 0]
    driv_nz = driv_um[driv_um > 0]
    if diff_nz.size == 0 and driv_nz.size == 0:
        print("Warning: No positive distances after scaling; skipping histogram.")
        return

    # Check for data incompatible with log plot
    all_nz = np.concatenate([a for a in (diff_nz, driv_nz) if a.size])
    x_min, x_max = float(all_nz.min()), float(all_nz.max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min <= 0:
        print("Invalid range for log histogram.")
        return

    # Create log-spaced bins
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 60)
    log_step_size = np.diff(np.log(bins))

    def logspace_density_percent(data, bins, log_step_size):
        """Return heights so that graph takes up area of 1"""
        counts, _ = np.histogram(data, bins=bins)
        N = counts.sum() if counts.sum() > 0 else 1
        heights = counts / (N * log_step_size)     # density per log-x unit
        return heights * 100.0                     # convert to percent

    y_diff = logspace_density_percent(diff_nz, bins, log_step_size) if diff_nz.size else None
    y_driv = logspace_density_percent(driv_nz, bins, log_step_size) if driv_nz.size else None

    # ---- plot as bars so area on a log-x axis is height * log_step_size ----
    plt.figure()
    lefts = bins[:-1]
    widths = bins[1:] - bins[:-1]

    if y_diff is not None:
        plt.bar(lefts, y_diff, width=widths, align='edge',
                alpha=0.5, edgecolor='black',
                label=f'Diffusive (values × {diff_weight:.3f})')
    if y_driv is not None:
        plt.bar(lefts, y_driv, width=widths, align='edge',
                alpha=0.5, edgecolor='black',
                label=f'Driven (values × {driv_weight:.3f})')

    plt.xscale('log')
    plt.xlabel(r"Distance per 4-s window ($\mu$m) (weighted by state amount)")
    plt.ylabel("Percentage")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(100))  # our heights are % already
    plt.title(f"Distribution of 4-Second Distances over {config.n_particles} particles\n"
              f"(weights: driven={driv_weight:.3f}, diffusive={diff_weight:.3f})")

    # Add vertical reference lines (darkened versions of bar colors)
    plt.axvline(x=0.035, color='#b2590a', linestyle='--', linewidth=2.5, label='Expected Driven')
    plt.axvline(x=0.77, color='#16537e', linestyle='--', linewidth=2.5, label='Expected Georgian')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the figure as SVG
    svg_filename = os.path.join(config.dirname,
                                f"flux_distribution_n{config.n_particles}_p{config.p_driv:.3f}_t{config.total_time}s.svg")
    os.makedirs(config.dirname, exist_ok=True)
    plt.savefig(svg_filename, format='svg', bbox_inches='tight')
    print(f"\nPlot saved → {svg_filename}\n")

    plt.show()


if __name__ == "__main__":
    main()
    # generate_displacement_time_driven_graph(n_particles=20, total_time=600)
    # generate_displacement_time_driven_graph(n_particles=20, total_time=3600)



