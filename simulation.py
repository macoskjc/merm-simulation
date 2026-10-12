import argparse
import time
from dataclasses import dataclass
from functools import wraps
from typing import NamedTuple

import numpy as np
import random
from matplotlib import pyplot as plt
import scipy
import statistics as st
from numba import njit

eta = 0.006 # Pa-s
R = 8.6e-8 # radius of particle (m)
Kb = 1.38064852e-23 # Boltzmann constant
T = 37 + 273.15 # K
g = 6 * np.pi * eta * R  # viscous drag coefficient
D = Kb * T / g # diffusivity coefficient via Stokes-Einstein equation
k = random.uniform(1.5e-6, 2.6e-6) # spring constant; N/m

CELL_RADIUS = 1.502e-5  # radius of cell (m)
NUCLEUS_RADIUS = 5e-6  # radius of cell nucleus (m)
TIME_BETWEEN_STATES = 0.03 # 0.41  # average time between states (s)
MOTOR_PROTEIN_SPEED = 1e-6  # speed of motor proteins (m/s)

TRAP_DIST_ALPHA = 1.40896797409667
TRAP_DIST_LOC = 0.04298463673860459e-6
TRAP_DIST_SCALE = 0.13924626080385946e-6
TRAP_DIST = scipy.stats.gamma(
    TRAP_DIST_ALPHA,
    loc=TRAP_DIST_LOC,
    scale=TRAP_DIST_SCALE,
)  # distance between traps (m)

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__!r} took {end - start:.4f} seconds")
        return result
    return wrapper


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    total_time: int = 2000  # maximum simulation time (s)
    n_particles: int = 1
    # fraction of time in directed motion; 0.004 reproduces the experimental 3% of 4-s
    # tracks containing directed motion (Holzwarth et al. 2020; sim-paper.pdf appendix)
    p_driv: float = 0.004
    trap_dist = TRAP_DIST  # dist of distance between traps (m)
    time_between: float = TIME_BETWEEN_STATES
    dt: float = 0.001  # time step (s)
    dirname: str = "sim"
    width: int = 600  # canvas width in pixels
    height: int = 600  # canvas height in pixels
    record_frames: bool = False
    end_early: bool = True  # whether to end the simulation early if the particle exits the cell

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SimulationConfig':
        """Create configuration from command line arguments"""
        return cls(
            total_time=args.total_time,
            n_particles=args.n_particles,
            p_driv=args.p_driv,
            # TODO how is one supposed to send a prob dist via command line?
            # trap_dist=TRAP_DIST if args.trap_dist == "default" else args.trap_dist,
            time_between=args.time_between,
            dt=args.dt,
            dirname=args.dirname,
            width=args.width,
            height=args.height,
            record_frames=args.record_frames,
        )


class SimulationOutput(NamedTuple):
    x: np.ndarray  # x-coordinates of the particle
    y: np.ndarray  # y-coordinates of the particle
    exit_time: float  # time at which the particle exits the cell
    distance_trap: np.ndarray  # distances traveled during hopping states
    distance_driven: np.ndarray  # distances traveled during driven states
    velocity_driven: np.ndarray  # velocities during driven states
    velocity_trap: np.ndarray  # velocities during hopping states
    state_dt: np.ndarray          # bool per *dt* step (len == len(x))
    t_01s: np.ndarray             # times sampled every 0.01 s (seconds)
    state_01s: np.ndarray         # bool per 0.01 s frame (len == len(t_01s))
    hop_t: np.ndarray
    hop_x: np.ndarray
    hop_y: np.ndarray


@njit(cache=True)
def truncated_dist(dist, low=-np.inf, high=np.inf):
    """Gaussian sample truncated to [low, high] – numba compatible."""
    while True:
        x = dist.rvs()
        if low <= x <= high:
            return x

@njit(cache=True)
def truncated_gauss(mu, sigma, low=-np.inf, high=np.inf):
    """Gaussian sample truncated to [low, high] – numba compatible."""
    while True:
        x = np.random.normal(mu, sigma)
        if low <= x <= high:
            return x

@njit(cache=True)
def truncated_gamma(shape, scale, low=0.0, high=np.inf):
    """Gamma sample truncated to [low, high] – numba compatible."""
    while True:
        x = np.random.gamma(shape, scale)
        if low <= x <= high:
                return x

@njit(cache=True)
def outside_nucleus(x, y):
    return x * x + y * y > NUCLEUS_RADIUS ** 2

@njit(cache=True)
def calc_directed_step(current_x, current_y, reverse_direction, dt):
    dr = np.random.normal(MOTOR_PROTEIN_SPEED * dt, MOTOR_PROTEIN_SPEED * dt)

    theta = np.arctan2(current_y, current_x)
    if reverse_direction:
        theta += np.pi

    jitter = 2e-9
    while True:
        nx = current_x + dr * np.cos(theta) + np.random.normal(0.0, jitter)
        ny = current_y + dr * np.sin(theta) + np.random.normal(0.0, jitter)
        if outside_nucleus(nx, ny):
            return nx, ny


@njit(cache=True)
def calc_diffusive_step(current_x, current_y, trap_x_center, trap_y_center, dt):
    # 1. Restoring‐force (drift) displacement
    displacement_from_center_x = current_x - trap_x_center
    displacement_from_center_y = current_y - trap_y_center

    trap_force_x = -k * displacement_from_center_x
    trap_force_y = -k * displacement_from_center_y

    drift_disp_x = (trap_force_x / g) * dt
    drift_disp_y = (trap_force_y / g) * dt

    # 2. Thermal‐diffusion displacement
    diffusion_std = np.sqrt(2 * D * dt)

    while True:
        diffusion_disp_x = diffusion_std * np.random.normal(0, 1)
        diffusion_disp_y = diffusion_std * np.random.normal(0, 1)

        # 3. Fine‐scale jitter
        jitter_std = 2e-9  # meters

        jitter_disp_x = np.random.normal(0, jitter_std)
        jitter_disp_y = np.random.normal(0, jitter_std)

        # 4. Sum all contributions into the total step
        dx = drift_disp_x + diffusion_disp_x + jitter_disp_x
        dy = drift_disp_y + diffusion_disp_y + jitter_disp_y

        # 5. Compute the candidate next position
        next_x = current_x + dx
        next_y = current_y + dy

        # 6. Only accept if it's outside the nucleus
        if outside_nucleus(next_x, next_y):
            return next_x, next_y

@njit(cache=True)
def calc_new_trap_position(trap_x, trap_y):
    while True:
        # 1. calculate random direction and distance
        # Sample Gamma(alpha, scale), then add loc to match SciPy's parameterization.
        r = TRAP_DIST_LOC + truncated_gamma(TRAP_DIST_ALPHA,TRAP_DIST_SCALE,0.0, np.inf)
        theta = np.random.uniform(0.0, 2.0 * np.pi)

        # 2. propose new center coordinates
        new_trap_x = trap_x + r * np.cos(theta)
        new_trap_y = trap_y + r * np.sin(theta)

        # 3. only accept if it's outside the nucleus
        if outside_nucleus(new_trap_x, new_trap_y):
            return new_trap_x, new_trap_y

@njit(cache=True)
def generate_state_duration(time_between_state_changes):
    while True:
        dur = np.random.normal(time_between_state_changes, time_between_state_changes)

        if dur > 0.0:
            return dur

@njit(cache=True)
def _move(total_time, dt,theta, stop_on_cell_exit,
          p_driv = 0.004, mean_driven_time = 0.6, std_driven_time  = 0.2):

    hop_mean = 1.0; hop_std = 0.3 # Constants so that particle changes traps ~once per second
    next_hop_time = np.inf       # Initialize hopping as disabled

    # Compute matching diffusive‐state distribution
    if p_driv == 0:
        mean_diffusive = 1 / dt # Particles move between traps once per second
        std_diffusive  = 0
    else:
        mean_diffusive = mean_driven_time * (1.0 - p_driv) / p_driv
        std_diffusive  = std_driven_time  * (1.0 - p_driv) / p_driv

    # Pre-allocate trajectory arrays
    max_states = int(total_time / dt) + 1
    x_history  = np.empty(max_states, dtype=np.float64)
    y_history  = np.empty(max_states, dtype=np.float64)
    state_hist = np.empty(max_states, dtype=np.bool_)
    final_i    = max_states - 1

    # Buffers to store trap centers for analysis
    hop_time_idx = np.empty(max_states, dtype=np.int64)
    hop_pos_x    = np.empty(max_states, dtype=np.float64)
    hop_pos_y    = np.empty(max_states, dtype=np.float64)
    hop_count    = 0

    # Initialize fixed-size numpy buffers
    diffusive_distance  = np.empty(max_states, dtype=np.float64)
    driven_distance     = np.empty_like(diffusive_distance)
    diffusive_velocity  = np.empty_like(diffusive_distance)
    driven_velocity     = np.empty_like(diffusive_distance)

    # Buffer indices we're writing to (replacing expensive .append calls)
    diffusive_distance_index = 0
    driven_distance_index    = 0
    diffusive_velocity_index = 0
    driven_velocity_index    = 0

    # Initial positions (particles evenly spaced out halfway between nucleus and cell membrane)
    start_radius = CELL_RADIUS / 2
    x_history[0] = trap_x = start_radius * np.cos(theta)
    y_history[0] = trap_y = start_radius * np.sin(theta)

    # Record the initial trap as the first "center"
    hop_time_idx[0] = 0
    hop_pos_x[0]    = trap_x
    hop_pos_y[0]    = trap_y
    hop_count       = 1

    # Generate state schedule on the fly
    is_driven = np.random.random() < p_driv
    state_hist[0] = is_driven
    if is_driven:
        next_switch_time = truncated_gauss(mean_driven_time, std_driven_time)
    else:
        next_switch_time = truncated_gauss(mean_diffusive, std_diffusive)

    next_hop_time = np.inf if is_driven else truncated_gauss(hop_mean, hop_std)
    state_start_index = 0
    reverse_next      = False
    exit_time         = -1.0

    # MAIN SIMULATION LOOP
    for i in range(1, max_states):
        state_hist[i] = is_driven
        current_time = i * dt

        # Check for state change
        if current_time >= next_switch_time:
            # Record distance traveled in state
            dx   = x_history[i-1] - x_history[state_start_index]
            dy   = y_history[i-1] - y_history[state_start_index]
            dist = np.hypot(dx, dy)
            if is_driven:
                driven_distance[driven_distance_index] = dist
                driven_distance_index += 1
                next_hop_time = np.inf              # disable hopping
            else:
                diffusive_distance[diffusive_distance_index] = dist
                diffusive_distance_index += 1
                next_hop_time = current_time + truncated_gauss(hop_mean, hop_std)

            # Reset state tracking, decide next state
            state_start_index = i - 1
            was_driven        = is_driven
            is_driven         = not was_driven

            # schedule hopping for the state we just entered
            if is_driven:
                next_hop_time = np.inf                         # disable hopping while driven
            else:
                next_hop_time = current_time + truncated_gauss(hop_mean, hop_std)

            # Log transition
            # print(f"{current_time}s: Just switched from "
            #       f"{'driven to diffusive' if was_driven else 'diffusive to driven'}")

            # Schedule next state change
            if is_driven:
                dur = truncated_gauss(mean_driven_time, std_driven_time)
            else:
                dur = truncated_gauss(mean_diffusive, std_diffusive)
            next_switch_time = current_time + dur
            reverse_next     = np.random.random() < 0.5  # 50% chance to reverse



        # Calculate next position based on current state
        if is_driven:
            new_x, new_y = calc_directed_step(
                x_history[i-1], y_history[i-1], reverse_next, dt
            )
            # During driven motion, trap follows particle
            trap_x, trap_y = new_x, new_y
        else:
            # Handle trap changing:
            if current_time >= next_hop_time:
                trap_x, trap_y = calc_new_trap_position(
                    trap_x, trap_y,
                )
                next_hop_time = current_time + truncated_gauss(hop_mean, hop_std)

                # record trap center for analysis
                hop_time_idx[hop_count] = i
                hop_pos_x[hop_count] = trap_x
                hop_pos_y[hop_count] = trap_y
                hop_count += 1

            # Calc next particle position
            new_x, new_y = calc_diffusive_step(
                x_history[i-1], y_history[i-1], trap_x, trap_y, dt
            )

        # Record position
        x_history[i] = new_x
        y_history[i] = new_y

        # Record velocity by state type
        v_now = np.hypot(new_x - x_history[i-1],
                         new_y - y_history[i-1]) / dt
        if is_driven:
            driven_velocity[driven_velocity_index] = v_now
            driven_velocity_index += 1
        else:
            diffusive_velocity[diffusive_velocity_index] = v_now
            diffusive_velocity_index += 1

        # Check for cell exit
        if np.hypot(new_x, new_y) > CELL_RADIUS:
            # Only record the FIRST exit time
            if exit_time == -1.0:
                exit_time = current_time
            final_i   = i
            if stop_on_cell_exit:
                break

    # Return trimmed arrays
    return (
        x_history[:final_i+1],
        y_history[:final_i+1],
        exit_time,
        diffusive_distance[:diffusive_distance_index].copy(),
        driven_distance[:driven_distance_index].copy(),
        driven_velocity[:driven_velocity_index].copy(),
        diffusive_velocity[:diffusive_velocity_index].copy(),
        state_hist[:final_i+1],
        hop_time_idx[:hop_count].copy(),
        hop_pos_x[:hop_count].copy(),
        hop_pos_y[:hop_count].copy(),
    )


def move(config, theta: float = 0.0, stop_on_cell_exit = True):
    """Acts as a wrapper for _move, passing in arguments
    so the njit doesn't need to handle SimulationConfig objects

    Returns:
        SimulationOutput: A named tuple containing:
            x (np.ndarray): x-coordinates of the particle
            y (np.ndarray): y-coordinates of the particle
            exit_time (float): Time at which particle exits cell (-1 if doesn't exit)
            distance_trap (list): Distances traveled during hopping states
            distance_driven (list): Distances traveled during driven states
            velocity_driven (list): Velocities during driven states
            velocity_trap (list): Velocities during hopping states
    """

    x, y, exit_time, dist_trap, distance_driven, vel_driven, vel_trap, state_dt, hop_idx, hop_x, hop_y = _move(
        config.total_time, config.dt,
        theta=theta, stop_on_cell_exit=stop_on_cell_exit, p_driv=config.p_driv
    )

    # Build 0.01 s (100 Hz) state track
    t_end = (len(x) - 1) * config.dt
    t_01s = np.round(np.arange(0.0, t_end + 1e-12, 0.01), 2)
    idx = np.minimum((t_01s / config.dt).astype(np.int64), len(state_dt) - 1)
    state_01s = state_dt[idx]

    # Convert hop indices to seconds
    hop_t = hop_idx.astype(np.float64) * config.dt

    return SimulationOutput(
        x=np.array(x),
        y=np.array(y),
        exit_time=exit_time,
        distance_trap=np.array(dist_trap),
        distance_driven=np.array(distance_driven),
        velocity_driven=np.array(vel_driven),
        velocity_trap=np.array(vel_trap),
        state_dt=np.array(state_dt),
        t_01s=t_01s,
        state_01s=state_01s,
        hop_t=hop_t,
        hop_x=hop_x,
        hop_y=hop_y,
    )

def graph(
    config: SimulationConfig,
    theta: float = 0,
):
    """Graph the movement of a particle in a cell."""
    sim_output = move(config, theta)
    x = sim_output.x
    y = sim_output.y

    gx = [i * 1e6 for i in x]
    gy = [i * 1e6 for i in y]
    for i in range(config.total_time-1):
        plt.scatter(gx[int(i/config.dt):int((i+1)/config.dt)], gy[int(i/config.dt):int((i+1)/config.dt)])
    centerx = []
    centery = []
    for i in range(config.total_time-1):
        centerx.append(st.mean(gx[int(i/config.dt):int((i+1)/config.dt)]))
        centery.append(st.mean(gy[int(i/config.dt):int((i+1)/config.dt)]))
    plt.plot(centerx, centery)
    plt.show()

