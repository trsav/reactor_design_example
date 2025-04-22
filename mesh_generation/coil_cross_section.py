import sys
import os
import subprocess
import time
import shutil
from tqdm import tqdm

# Configure path for imports
sys.path.insert(1, os.path.join(sys.path[0], ".."))
# sys.path.insert(1, "mesh_generation/classy_blocks/src/")
from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh

# Import data visualisation libraries
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from PIL import Image
import imageio
from matplotlib import rc

# Import JAX libraries for Gaussian processes
import distrax as dx
import jax.numpy as jnp
import jax.random as jr
from jax import jit
import optax as ox
from typing import Dict
import gpjax as gpx


# Import typing and dataclass functionality
from dataclasses import dataclass
from typing import Dict
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import jax
import tensorflow_probability.substrates.jax as tfp
from dataclasses import dataclass
from jaxtyping import Array, Float, install_import_hook

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.kernels.computations import DenseKernelComputation
    from gpjax.parameters import PositiveReal, Static  # Import parameter types


# Helper function (ensure jnp usage)
def angular_distance(x, y, c):
    """Calculate circular distance between angles x and y with period 2c."""
    return jnp.abs((x - y + c) % (c * 2) - c)


@dataclass
class Polar(gpx.kernels.AbstractKernel):
    """
    Custom kernel for polar coordinates (Updated for GPJax >= 0.8.0).
    Based on Padonou & Roustant (2015): https://hal.inria.fr/hal-01119942v1
    """

    # Define parameters with types
    period: Static = Static(jnp.array(2 * jnp.pi))
    tau: PositiveReal = PositiveReal(
        jnp.array(4.0)
    )  # Initial value 4.0, constrained > 0

    def __init__(
        self,
        period: float = 2 * jnp.pi,
        tau: float = 4.0,  # Default initial value
        active_dims: list[int] | slice | None = None,
        n_dims: int | None = None,
        **kwargs,  # Allow passing other args like parameter definitions
    ):
        super().__init__(
            active_dims=active_dims,
            n_dims=n_dims,
            compute_engine=DenseKernelComputation(),
        )

        self.period = kwargs.get("period", Static(jnp.array(period)))
        self.tau = kwargs.get("tau", PositiveReal(jnp.array(tau)))

    def __call__(
        self, x: Float[Array, "1 D"], y: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        """Compute the kernel value between two single points in polar coordinates."""
        c = self.period.value / 2.0  # Access parameter value using .value
        t = angular_distance(
            x.squeeze(), y.squeeze(), c
        )  # Ensure inputs are scalar-like for distance
        # Access parameter value using .value
        tau_val = self.tau.value
        # Clip term (1 - t/c) must be non-negative before exponentiation
        base = jnp.clip(1.0 - t / c, 0.0, jnp.inf)
        K = (1.0 + tau_val * t / c) * base**tau_val
        return K.squeeze()  # Return scalar value


# Updated GP Interpolation Function (aligned with GPJax >= 0.8.0)
def gp_interpolate_polar(
    X: Float[Array, "N 1"],
    y: Float[Array, "N 1"],
    n_interp: int,
    learning_rate: float = 0.05,
    num_iters: int = 500,
    seed: int = 1,
):
    """
    Interpolate polar coordinates using a Gaussian process with a Polar kernel.

    Args:
        X: Input angles (N, 1).
        y: Corresponding radii or values (N, 1).
        n_interp: Number of points to interpolate onto around the circle.
        learning_rate: Learning rate for the AdamW optimizer.
        num_iters: Number of optimization iterations.
        seed: Random seed for reproducibility.

    Returns:
        Tuple[Float[Array, "M 1"], Float[Array, "M 1"]]: Interpolated angles and predicted means.
    """
    key = jr.PRNGKey(seed)

    # Ensure data is in JAX arrays and correct shape/type
    X = jnp.asarray(X, dtype=jnp.float64).reshape(-1, 1)
    y = jnp.asarray(y, dtype=jnp.float64).reshape(-1, 1)

    # Generate target angles for interpolation
    angles = jnp.linspace(0, 2 * jnp.pi, num=n_interp).reshape(-1, 1)

    # Create dataset
    D = gpx.Dataset(X=X, y=y)

    # Define polar Gaussian process components
    PKern = Polar()  # Instantiate the updated Polar kernel
    meanf = gpx.mean_functions.Zero()
    # Use D.n for number of datapoints
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

    # Define the model using the Prior * Likelihood syntax
    prior = gpx.gps.Prior(mean_function=meanf, kernel=PKern)
    posterior = prior * likelihood

    # Define the objective function (negative conjugate Marginal Log-Likelihood)
    # Use gpx.objectives module
    objective = jit(gpx.objectives.conjugate_mll(negative=True))

    # Define the optimizer using Optax
    optim = ox.adamw(learning_rate=learning_rate)

    # Optimise GP's marginal log-likelihood using gpx.fit (for gradient-based optim)
    # gpx.fit returns the optimized model state and the optimization history
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=objective,
        train_data=D,
        optim=optim,
        num_iters=num_iters,
        key=key,
        # verbose=True # Uncomment for optimization progress
    )

    # Generate predictions at the interpolation angles
    # predict gives the latent function distribution
    latent_pred = opt_posterior.predict(angles, train_data=D)
    # Pass the latent distribution through the likelihood to get the predictive distribution
    predictive_dist = opt_posterior.likelihood(latent_pred)

    # Extract the mean of the predictive distribution
    mu = predictive_dist.mean()

    return angles, mu


def rotate_z(x, y, z, r_z):
    """Rotate points around z-axis by r_z radians."""
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    x_new = x * np.cos(r_z) - y * np.sin(r_z)
    y_new = x * np.sin(r_z) + y * np.cos(r_z)
    return x_new, y_new, z


def rotate_x(x0, y0, z0, r_x):
    """Rotate points around x-axis by r_x radians."""
    y = np.array(y0) - np.mean(y0)
    z = np.array(z0) - np.mean(z0)
    y_new = y * np.cos(r_x) - z * np.sin(r_x)
    z_new = y * np.sin(r_x) - z * np.cos(r_x)
    y_new += np.mean(y0)
    z_new += np.mean(z0)
    return x0, y_new, z_new


def rotate_y(x0, y0, z0, r_y):
    """Rotate points around y-axis by r_y radians."""
    x = np.array(x0) - np.mean(x0)
    z = np.array(z0) - np.mean(z0)
    z_new = z * np.cos(r_y) - x * np.sin(r_y)
    x_new = z * np.sin(r_y) - x * np.cos(r_y)
    x_new += np.mean(x0)
    z_new += np.mean(z0)
    return x_new, y0, z_new


def rotate_xyz(x, y, z, t, t_x, c_x, c_y, c_z):
    """Apply combined rotations to points."""
    x -= c_x
    y -= c_y
    x, y, z = rotate_x(x, y, z, t_x)
    x, y, z = rotate_z(x, y, z, t)
    x += c_x
    y += c_y
    x, y, z = rotate_z(x, y, z, 3 * np.pi / 2)
    return x, y, z


def create_center_circle(d, r):
    """Create a circle with centre parameters and radius."""
    c_x, c_y, t, t_x, c_z = d
    alpha = np.linspace(0, 2 * np.pi, 64)
    z = r * np.cos(alpha) + c_z
    x = r * np.sin(alpha) + c_x
    y = np.array([c_y for i in range(len(z))])
    x, y, z = rotate_xyz(x, y, z, t, t_x, c_x, c_y, c_z)
    return x, y, z


def create_circle(d, radius_og):
    """Create a circle with variable radius at different angles."""
    c_x, c_y, t, t_x, c_z = d
    radius = radius_og.copy()
    angles_og = np.linspace(0, np.pi * 2, len(radius), endpoint=False)
    angles = angles_og.copy()
    r_mean = np.mean(radius)
    r_std = np.std(radius)

    # Handle non-uniform radius
    if r_std != 0:
        radius = (radius - r_mean) / r_std
        angles, radius = gp_interpolate_polar(
            angles.reshape(-1, 1), radius.reshape(-1, 1), 64
        )
        angles = angles[:, 0]
        radius = radius * r_std + r_mean
    else:
        angles = np.linspace(0, np.pi * 2, 64)
        radius = np.array([r_mean for i in range(64)])

    # Replace NaN values
    for i in range(len(radius)):
        if radius[i] != radius[i]:
            radius = radius.at[i].set(r_mean)

    # Generate circular coordinates
    z_n = radius * np.cos(angles)
    x_n = radius * np.sin(angles)
    y_n = np.array([c_y for i in range(len(x_n))])
    x, y, z = rotate_xyz(x_n + c_x, y_n, z_n + c_z, t, t_x, c_x, c_y, c_z)

    # Generate original points for comparison
    x_p = np.array(
        [radius_og[i] * np.sin(angles_og[i]) + c_x for i in range(len(radius_og))]
    )
    z_p = np.array(
        [radius_og[i] * np.cos(angles_og[i]) + c_z for i in range(len(radius_og))]
    )
    y_p = np.array([c_y for i in range(len(radius_og))])

    x_p, y_p, z_p = rotate_xyz(x_p, y_p, z_p, t, t_x, c_x, c_y, c_z)
    return x, y, z, x_p, y_p, z_p, x_n, z_n


def cylindrical_convert(r, theta, z):
    """Convert cylindrical to Cartesian coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z


def cartesian_convert(x, y, z):
    """Convert Cartesian to cylindrical coordinates."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # Ensure theta is monotonically increasing
    for i in range(1, len(theta)):
        if theta[i] < theta[i - 1]:
            while theta[i] < theta[i - 1]:
                theta[i] = theta[i] + 2 * np.pi
    return r, theta, z


def interpolate(y, fac_interp, kind):
    """Interpolate data with specified factor and method."""
    x = np.linspace(0, len(y), len(y))
    x_new = np.linspace(0, len(y), len(y) * fac_interp)
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    return y_new


def plot_block(block, ax):
    """Plot a 3D block structure."""
    block = np.array(block)
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    for i in range(len(lines)):
        l = lines[i]
        ax.plot(
            [block[l[0], 0], block[l[1], 0]],
            [block[l[0], 1], block[l[1], 1]],
            [block[l[0], 2], block[l[1], 2]],
            c="tab:blue",
            lw=2,
            alpha=0.25,
        )
    return


def add_start(x, y, z, t, t_x, L):
    """Add starting point to coil path."""
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    rho, theta, z = cartesian_convert(x, y, z)

    # Calculate new starting point
    rho_start = np.sqrt(rho[0] ** 2 + L**2)
    theta_start = theta[0] - np.arctan(L / rho[0])
    z_start = z[0]

    # Prepend new point to arrays
    z = np.append(z_start, z)
    t = np.append(t[0], t)
    t_x = np.append(t_x[0], t_x)
    rho = np.append(rho_start, rho)
    theta = np.append(theta_start, theta)

    x, y, z = cylindrical_convert(rho, theta, z)
    return x, y, z, t, t_x


def interpolate_num_same(x, x_new, y, kind):
    """Interpolate y values to match x_new points."""
    f = interp1d(x, y, kind=kind)
    y_new = f(x_new)
    return y_new, x_new


def interpolate_split(r, theta, z, fid_ax):
    """Interpolate cylindrical coordinates with split points."""
    x, y, z = cylindrical_convert(r, theta, z)

    # Interpolate middle section
    rho_mid = interpolate(r[1:], fid_ax, "quadratic")
    theta_mid = interpolate(theta[1:], fid_ax, "quadratic")
    z_mid = interpolate(z[1:], fid_ax, "quadratic")
    x_m, y_m, z_m = cylindrical_convert(rho_mid, theta_mid, z_mid)

    # Interpolate start section
    x_start = interpolate([x[0], x[1]], int(fid_ax / 2), "linear")
    y_start = interpolate([y[0], y[1]], int(fid_ax / 2), "linear")
    z_start = interpolate([z[0], z[1]], int(fid_ax / 2), "linear")
    x_start = x_start[:-1]
    y_start = y_start[:-1]
    z_start = z_start[:-1]

    # Calculate distance of start section
    d_start = np.sqrt(
        (x_start[0] - x_start[-1]) ** 2
        + (y_start[0] - y_start[-1]) ** 2
        + (z_start[0] - z_start[-1]) ** 2
    )

    # Combine sections
    x = np.append(x_start, x_m)
    y = np.append(y_start, y_m)
    z = np.append(z_start, z_m)

    len_s = len(x_start)

    # Calculate cumulative distances
    d_store = [0]
    for i in range(1, len(x)):
        d = np.sqrt(
            (x[i - 1] - x[i]) ** 2 + (y[i - 1] - y[i]) ** 2 + (z[i - 1] - z[i]) ** 2
        )
        d_store.append(d)
    d_store = np.cumsum(d_store)

    # Smoothen the interpolation around connection point
    s = len_s - int(fid_ax / 2)
    e = len_s + int(fid_ax / 2)

    d_int = [d_store[s + 0], d_store[s + 1], d_store[e - 2], d_store[e - 1]]
    z_int = [z[s + 0], z[s + 1], z[e - 2], z[e - 1]]

    z_new, _ = interpolate_num_same(d_int, d_store[s:e], z_int, "quadratic")

    for i in range(s, e):
        z[i] = z_new[i - s]

    return x, y, z, d_start


def add_end(x, y, z, dx, dy, dz, d_start, fid_ax):
    """Add end point to coil path with proper scaling."""
    d_end = np.sqrt(dx**2 + dy**2 + dz**2)
    factor = d_start / d_end

    # Calculate end point
    x_e = x[-1] + dx * factor
    y_e = y[-1] + dy * factor
    z_e = z[-1] + dz * factor

    # Interpolate to end point
    x_end = interpolate([x[-1], x_e], int(fid_ax / 2), "linear")
    y_end = interpolate([y[-1], y_e], int(fid_ax / 2), "linear")
    z_end = interpolate([z[-1], z_e], int(fid_ax / 2), "linear")

    x_end = x_end[1:]
    y_end = y_end[1:]
    z_end = z_end[1:]

    # Append end section
    x = np.append(x, x_end)
    y = np.append(y, y_end)
    z = np.append(z, z_end)

    return x, y, z


def create_mesh(x, z, path: str):
    """Create mesh for coil cross-section simulation."""
    # Coil parameters
    data = {
        "coils": 2,
        "start_rad": 0.0025,
        "radius_center": 0.00125,
        "length": np.pi * 2 * 0.010391 * 2,
        "inversion_parameter": 0.0,
        "a": 0.0,
        "f": 0.0,
        "re": 50.0,
        "pitch": 0.0104,
        "coil_rad": 0.0125,
        "n_cs": 6,
        "n_l": 6,
    }
    n_s = data["n_cs"]
    n_l = data["n_l"]

    # Process input points for interpolation
    interp_points = []
    x = np.array(x).flatten()
    for i in range(n_l):
        new_points = []
        for j in range(n_s):
            new_points.append(x[i * n_l + j])
        interp_points.append(np.array(new_points))

    # Extract parameters
    coil_rad = data["coil_rad"]
    pitch = data["pitch"]
    length = data["length"]
    r_c = data["radius_center"]
    s_rad = data["start_rad"]

    # Add start and end inlet points
    interp_points.append(np.array([s_rad for i in range(len(interp_points[0]))]))
    interp_points.insert(0, np.array([s_rad for i in range(len(interp_points[0]))]))
    interp_points.insert(0, np.array([s_rad for i in range(len(interp_points[0]))]))

    # Set fidelity parameters
    fid_rad = int(z[1])
    fid_ax = int(z[0])

    # Calculate coil properties
    coils = length / (2 * np.pi * coil_rad)
    h = coils * pitch
    keys = ["x", "y", "t", "t_x", "z"]
    data = {}

    n = len(interp_points) - 1
    t_x = -np.arctan(h / length)
    coil_vals = np.linspace(0, 2 * coils * np.pi, n)

    # Generate coil geometry points
    data["x"] = [(coil_rad * np.cos(x_y)) for x_y in coil_vals]
    data["y"] = [(coil_rad * np.sin(x_y)) for x_y in coil_vals]
    data["t"] = list(coil_vals)
    data["t_x"] = [0] + [t_x for i in range(n - 1)]
    data["z"] = list(np.linspace(0, h, n))

    # Add starting section
    L = coil_rad
    data["x"], data["y"], data["z"], data["t"], data["t_x"] = add_start(
        data["x"], data["y"], data["z"], data["t"], data["t_x"], L
    )

    # Initialize mesh
    mesh = Mesh()

    # Set up figure for visualization
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
    fig.tight_layout()

    axs[0].view_init(0, 270)
    axs[1].view_init(0, 180)
    axs[2].view_init(270, 0)

    # Copy mesh template folder
    try:
        shutil.copytree("mesh_generation/mesh", path)
    except:
        print("Folder exists")

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, top=0.99, bottom=0.01)

    # Generate points for all coil sections
    n = len(data["x"])
    p_list = []
    p_c_list = []
    p_interp = []

    for i in range(n):
        x, y, z, x_p, y_p, z_p, x_n, z_n = create_circle(
            [data[keys[j]][i] for j in range(len(keys))], interp_points[i]
        )

        p_list.append([x, y, z])
        p_interp.append([x_n, z_n])
        if i > 0:
            for ax in axs:
                ax.scatter(x_p, y_p, z_p, c="k", alpha=1, s=10)

    # Configure visualisation axes
    for ax in axs:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Label axes
    axs[0].set_xlabel("x", fontsize=14)
    axs[0].set_zlabel("z", fontsize=14)
    axs[1].set_ylabel("y", fontsize=14)
    axs[1].set_zlabel("z", fontsize=14)
    axs[2].set_ylabel("y", fontsize=14)
    axs[2].set_xlabel("x", fontsize=14)

    # Save initial visualization
    plt.savefig(path + "/points.png", dpi=300)

    # Convert lists to arrays
    p_list = np.asarray(np.array(p_list))
    p_c_list = np.asarray(p_c_list)
    p_interp = np.asarray(p_interp)

    # Plot all coil sections
    for i in range(1, n):
        for ax in axs:
            ax.plot(
                p_list[i, 0, :], p_list[i, 1, :], p_list[i, 2, :], c="k", alpha=1, lw=1
            )

    # Save GP slices visualisation
    fig.savefig(path + "/gp_slices.png", dpi=300)

    # Create polar plots of cross-sections
    figc, axsc = plt.subplots(
        2,
        int((len(interp_points) - 3) / 2),
        figsize=(10, 6),
        subplot_kw=dict(projection="polar"),
        sharey=True,
    )
    figc.tight_layout()
    gridspec = fig.add_gridspec(1, 1)
    angles = np.linspace(0, 2 * np.pi, len(interp_points[0]), endpoint=False)

    # Plot each cross-section
    i = 2
    for ax in axsc.ravel():
        ax.set_yticks(
            [0, 0.001, 0.002, 0.003, 0.004],
            [0, "1E-3", "2E-3", "3E-3", "4E-3"],
            fontsize=8,
        )
        ax.set_xticks(
            np.linspace(0, 2 * np.pi, 8, endpoint=False),
            [
                "0",
                r"$\frac{\pi}{4}$",
                r"$\frac{\pi}{2}$",
                r"$\frac{3\pi}{4}$",
                r"$\pi$",
                r"$\frac{5\pi}{4}$",
                r"$\frac{3\pi}{2}$",
                r"$\frac{7\pi}{4}$",
            ],
        )

        ax.set_ylim(0, 0.004)
        ax.scatter(angles, interp_points[i], alpha=1, c="k")

        # Convert to polar coordinates
        x = p_interp[i, 0, :]
        z = p_interp[i, 1, :]
        r = np.sqrt(x**2 + z**2)
        theta = np.arctan2(x, z)

        ax.plot(theta, r, alpha=1, c="k", lw=2.5)
        i += 1

    figc.savefig(path + "/points_short.png", dpi=300)

    # Convert to cylindrical coordinates
    p_cylindrical_list = []
    for i in range(len(p_list[0, 0, :])):
        r, theta, z = cartesian_convert(
            p_list[:, 0, i], p_list[:, 1, i], p_list[:, 2, i]
        )
        p_cylindrical_list.append([r, theta, z])

    p_cylindrical_list = np.asarray(p_cylindrical_list)
    p_new_list = []

    # Interpolate points
    for i in range(len(p_cylindrical_list[:, 0, 0])):
        x, y, z, d_start = interpolate_split(
            p_cylindrical_list[i, 0, :],
            p_cylindrical_list[i, 1, :],
            p_cylindrical_list[i, 2, :],
            fid_ax,
        )
        p_new_list.append([x, y, z])

    p_new_list = np.asarray(p_new_list)

    # Calculate mean coordinates and displacement
    m_x = np.mean(p_new_list[:, 0, :], axis=0)
    m_y = np.mean(p_new_list[:, 1, :], axis=0)
    m_z = np.mean(p_new_list[:, 2, :], axis=0)
    dx = m_x[-1] - m_x[-2]
    dy = m_y[-1] - m_y[-2]
    dz = m_z[-1] - m_z[-2]

    # Add end section to all points
    p_list = []
    for i in range(len(p_new_list[:, 0, 0])):
        x, y, z = add_end(
            p_new_list[i, 0, :],
            p_new_list[i, 1, :],
            p_new_list[i, 2, :],
            dx,
            dy,
            dz,
            d_start,
            fid_ax,
        )
        p_list.append([x, y, z])

    p_list = np.asarray(p_list)

    # Plot interpolated paths
    for i in range(len(p_list[:, 0, 0])):
        for ax in axs:
            ax.plot(
                p_list[i, 0, :],
                p_list[i, 1, :],
                p_list[i, 2, :],
                c="k",
                alpha=0.5,
                lw=0.5,
            )

    # Save interpolated visualization
    fig.savefig(path + "/interpolated.png", dpi=300)

    # Create pre-render visualization
    fig_i, axs_i = plt.subplots(1, 3, figsize=(10, 3), subplot_kw=dict(projection="3d"))
    fig_i.tight_layout()

    axs_i[0].view_init(0, 270)
    axs_i[1].view_init(0, 180)
    axs_i[2].view_init(270, 0)

    # Plot subset of points for clarity
    for i in np.linspace(0, len(p_list[:, 0, 0]) - 1, 10):
        i = int(i)
        for ax in axs_i:
            ax.plot(
                p_list[i, 0, :],
                p_list[i, 1, :],
                p_list[i, 2, :],
                c="k",
                alpha=0.25,
                lw=0.5,
            )

    for i in range(len(p_list[0, 0, :])):
        for ax in axs_i:
            ax.plot(
                p_list[:, 0, i],
                p_list[:, 1, i],
                p_list[:, 2, i],
                c="k",
                alpha=0.25,
                lw=0.5,
            )

    # Configure visualization
    for ax in axs_i:
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid()

    # Label axes
    axs_i[0].set_xlabel("x", fontsize=14)
    axs_i[0].set_zlabel("z", fontsize=14)
    axs_i[1].set_ylabel("y", fontsize=14)
    axs_i[1].set_zlabel("z", fontsize=14)
    axs_i[2].set_ylabel("y", fontsize=14)
    axs_i[2].set_xlabel("x", fontsize=14)

    plt.savefig(path + "/pre-render.png", dpi=600)

    # Define color for rendering
    col = (212 / 255, 41 / 255, 144 / 255)

    # Transpose points for easier processing
    p_list = p_list.T
    c = np.mean(p_list, axis=2)
    s = 0
    ds = int((len(p_list[0, 0, :])) / 4)
    c_same = np.zeros_like(p_list)

    # Generate center points
    for i in range(len(p_list[0, 0, :])):
        c_same[:, :, i] = c
    inner_p_list = 0.8 * p_list + 0.2 * c_same

    print("Defining blocks...")
    # Create mesh blocks for each quarter of the cross-section
    for i in tqdm(range(4)):
        e = s + ds + 1
        quart = p_list[:, :, s:e]
        inner_quart = inner_p_list[:, :, s:e]

        c_weight = 0.4
        mid_ind = int(len(quart[0, 0, :]) / 2)

        # Define reference points
        p_30 = c
        p_74 = (c_weight * quart[:, :, 0]) + ((1 - c_weight) * p_30)
        p_21 = (c_weight * quart[:, :, -1]) + ((1 - c_weight) * p_30)
        p_65 = 0.8 * (0.5 * p_74 + 0.5 * p_21) + 0.2 * quart[:, :, mid_ind]

        # Create inner blocks
        for k in range(len(quart[:, 0, 0]) - 1):
            block_points = [
                list(p_30[k + 1, :]),
                list(p_21[k + 1, :]),
                list(p_21[k, :]),
                list(p_30[k, :]),
                list(p_74[k + 1, :]),
                list(p_65[k + 1, :]),
                list(p_65[k, :]),
                list(p_74[k, :]),
            ]
            block_edges = []

            block = Block.create_from_points(block_points, block_edges)

            # Set block properties
            block.chop(0, count=fid_rad)
            block.chop(1, count=1)
            block.chop(2, count=fid_rad)

            mesh.add_block(block)
            if k == 0:
                block.set_patch("back", "inlet")
            if k == len(quart[:, 0, 0]) - 2:
                block.set_patch("front", "outlet")

        # Create outer blocks with curves
        for m_in in [0, int(len(quart[0, 0, :]) / 2)]:
            m_out = m_in + mid_ind
            if m_out == len(quart[0, 0, :]):
                m_out -= 1

            # Define reference points
            p_74_ = inner_quart[:, :, m_in]
            p_74_u = quart[:, :, m_in]
            p_65_ = inner_quart[:, :, m_out]
            p_65_u = quart[:, :, m_out]
            p_30_u = p_74_
            p_21_u = p_65_

            if m_in == 0:
                p_30_ = p_74
                p_21_ = p_65
            else:
                p_30_ = p_65
                p_21_ = p_21

            for k in range(len(quart[:, 0, 0]) - 1):
                # Create curved edges
                curve_76 = inner_quart[0, :, m_in:m_out].T
                curve_76_u = quart[0, :, m_in:m_out].T
                curve_76 = list(
                    [
                        list(curve_76[int(i), :])
                        for i in np.linspace(0, len(curve_76[:, 0]) - 1, fid_rad)
                    ]
                )
                curve_76_u = list(
                    [
                        list(curve_76_u[int(i), :])
                        for i in np.linspace(0, len(curve_76_u[:, 0]) - 1, fid_rad)
                    ]
                )
                curve_45 = inner_quart[k + 1, :, m_in:m_out].T
                curve_45_u = quart[k + 1, :, m_in:m_out].T
                curve_45 = list(
                    [
                        list(curve_45[int(i), :])
                        for i in np.linspace(0, len(curve_45[:, 0]) - 1, fid_rad)
                    ]
                )
                curve_45_u = list(
                    [
                        list(curve_45_u[int(i), :])
                        for i in np.linspace(0, len(curve_45_u[:, 0]) - 1, fid_rad)
                    ]
                )

                # Create inner block
                block_points = [
                    list(p_30_[k + 1, :]),
                    list(p_21_[k + 1, :]),
                    list(p_21_[k, :]),
                    list(p_30_[k, :]),
                    list(p_74_[k + 1, :]),
                    list(p_65_[k + 1, :]),
                    list(p_65_[k, :]),
                    list(p_74_[k, :]),
                ]
                block_edges = [Edge(7, 6, curve_76), Edge(4, 5, curve_45)]

                block = Block.create_from_points(block_points, block_edges)
                block.chop(0, count=fid_rad)
                block.chop(1, count=1)
                block.chop(2, count=fid_rad)

                if k == 0:
                    block.set_patch("back", "inlet")
                if k == len(quart[:, 0, 0]) - 2:
                    block.set_patch("front", "outlet")

                mesh.add_block(block)

                # Create outer block
                block_points = [
                    list(p_30_u[k + 1, :]),
                    list(p_21_u[k + 1, :]),
                    list(p_21_u[k, :]),
                    list(p_30_u[k, :]),
                    list(p_74_u[k + 1, :]),
                    list(p_65_u[k + 1, :]),
                    list(p_65_u[k, :]),
                    list(p_74_u[k, :]),
                ]
                block_edges = [
                    Edge(7, 6, curve_76_u),
                    Edge(4, 5, curve_45_u),
                    Edge(0, 1, curve_45),
                    Edge(3, 2, curve_76),
                ]
                block = Block.create_from_points(block_points, block_edges)

                block.chop(0, count=fid_rad)
                block.chop(1, count=1)
                block.chop(2, count=fid_rad)

                if k == 0:
                    block.set_patch("back", "inlet")
                if k == len(quart[:, 0, 0]) - 2:
                    block.set_patch("front", "outlet")

                block.set_patch("top", "wall")
                mesh.add_block(block)
        s += ds

    # Generate mesh file and run blockMesh
    print("Writing geometry...")
    mesh.write(output_path=os.path.join(path, "system", "blockMeshDict"), geometry=None)
    print("Running blockMesh...")
    command = path + " /Allmesh.sh"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return
