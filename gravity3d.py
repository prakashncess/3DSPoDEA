import numpy as np
from numba import njit, prange

# --- Physical constant ---
G = 6.67430e-11  # m^3/kg/s^2
G_mGal = G * 1e5  # convert to mGal


@njit
def prism_gz(xo, yo, zo, x1, x2, y1, y2, z1, z2, drho):
    """
    Compute vertical gravity acceleration (gz) at an observation point (xo, yo, zo)
    due to a rectangular prism defined by its corners and density contrast drho.
    """
    gz = 0.0
    for i in [x1, x2]:
        for j in [y1, y2]:
            for k in [z1, z2]:
                dx = i - xo
                dy = j - yo
                dz = k - zo
                R = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-10)
                sign = (-1)**((i == x2) + (j == y2) + (k == z2))
                gz += sign * np.arctan2(dx * dy, dz * R + 1e-10)
    return G_mGal * drho * gz


@njit(parallel=True)
def compute_gravity(Xobs, Yobs, Zobs, x_edges, y_edges, z_edges, rho_model):
    """
    Compute gz gravity anomaly at all observation points due to a 3D density model.
    """
    nx, ny, nz = rho_model.shape
    nobs_x, nobs_y = Xobs.shape
    gz_obs = np.zeros((nobs_x, nobs_y))

    for i in prange(nobs_x):
        for j in range(nobs_y):
            xo, yo, zo = Xobs[i, j], Yobs[i, j], Zobs[i, j]
            gz = 0.0
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        drho = rho_model[ix, iy, iz]
                        if drho == 0:
                            continue
                        x1, x2 = x_edges[ix], x_edges[ix + 1]
                        y1, y2 = y_edges[iy], y_edges[iy + 1]
                        z1, z2 = z_edges[iz], z_edges[iz + 1]
                        gz += prism_gz(xo, yo, zo, x1, x2, y1, y2, z1, z2, drho)
            gz_obs[i, j] = gz
    return gz_obs
