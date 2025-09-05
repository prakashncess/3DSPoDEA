import numpy as np
from numba import njit, prange

# --- Physical constant ---
G = 6.67430e-11  # m^3/kg/s^2
G_mGal = G * 1e5  # Convert to mGal

@njit
def safe_log(x):
    return np.log(np.abs(x) + 1e-10)

@njit
def safe_arctan(y, x):
    return np.arctan2(y, x + 1e-10)

@njit
def prism_gz(xo, yo, zo, x1, x2, y1, y2, z1, z2, drho):
    """
    Compute gz at observation point (xo, yo, zo) due to a rectangular prism.
    """
    gz = 0.0
    for i in [0, 1]:
        xi = x1 if i == 0 else x2
        sx = -1 if i == 0 else 1
        dx = xi - xo

        for j in [0, 1]:
            yj = y1 if j == 0 else y2
            sy = -1 if j == 0 else 1
            dy = yj - yo

            for k in [0, 1]:
                zk = z1 if k == 0 else z2
                sz = -1 if k == 0 else 1
                dz = zk - zo

                R = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-20)
                sign = sx * sy * sz

                term = (
                    dz * safe_arctan(dx * dy, dz * R) -
                    dx * safe_log(R + dy) -
                    dy * safe_log(R + dx)
                )

                gz += sign * term

    return G_mGal * drho * gz


@njit(parallel=True)
def compute_gravity(Xobs, Yobs, Zobs, x_edges, y_edges, z_edges, rho_model):
    """
    Compute gravity anomaly gz at each observation point due to a 3D density model.
    
    Parameters:
        Xobs, Yobs, Zobs : 2D arrays of observation coordinates
        x_edges, y_edges, z_edges : voxel grid edges
        rho_model : 3D density contrast array (nx, ny, nz) in kg/m³
        
    Returns:
        gz_obs : 2D array (same shape as Xobs) of gravity anomalies in mGal
    """
    nx, ny, nz = rho_model.shape
    nobs_x, nobs_y = Xobs.shape
    gz_obs = np.zeros((nobs_x, nobs_y))

    for i in prange(nobs_x):
        for j in range(nobs_y):
            xo, yo, zo = Xobs[i, j], Yobs[i, j], Zobs[i, j]
            gz = 0.0
            for ix in range(nx):
                x1, x2 = x_edges[ix], x_edges[ix + 1]
                for iy in range(ny):
                    y1, y2 = y_edges[iy], y_edges[iy + 1]
                    for iz in range(nz):
                        drho = rho_model[ix, iy, iz]
                        if drho == 0:
                            continue
                        z1, z2 = z_edges[iz], z_edges[iz + 1]
                        gz += prism_gz(xo, yo, zo, x1, x2, y1, y2, z1, z2, drho)
            gz_obs[i, j] = gz
    return gz_obs
