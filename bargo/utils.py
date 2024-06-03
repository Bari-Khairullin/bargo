import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm

def scatter(arrs, grid=None, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    xmin, ymin = np.inf, np.inf
    xmax, ymax = -np.inf, -np.inf
    for l, arr in arrs:
        ax.scatter(arr[:, 0], arr[:, 1], label=l, s=0.1)
        arr_xmin = arr[:, 0].min()
        arr_ymin = arr[:, 1].min()
        arr_xmax = arr[:, 0].max()
        arr_ymax = arr[:, 1].max()
        xmin, ymin = min(xmin, arr_xmin), min(ymin, arr_ymin)
        xmax, ymax = max(xmax, arr_xmax), max(ymax, arr_ymax)
    if grid:
        for line in grid:
            ax.plot(*zip(*line), color='gray', linewidth=0.2, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.axis('off')
    ax.legend()
    return fig, ax

def unit_circle_projection_loss(y_true, y_pred):
    scalar = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=1)
    norm_pred = tf.norm(y_pred, axis=1)
    cos = scalar / norm_pred
    return tf.reduce_sum(tf.abs(cos - 1)) / 2 / np.pi

def arctan2(arr):
    res = [np.arctan2(y, x) % (2 * np.pi) for x, y in arr]
    return np.array(res)

def norm(arr):
    res = np.linalg.norm(arr, axis=1)
    return res

def dec_to_polar(x, y):
    return (np.sqrt(x**2 + y**2), arctan2(x, y))

def r_of_rectangle(phi, a, b):
    diag = np.arctan2(a, b)
    n = phi // (np.pi/2)
    phi -= n * np.pi/2
    if phi <= diag:
        r = b / 2 / np.cos(phi)
    else:
        r = a / 2 / np.sin(phi)
    return r

def piecewise_linear(x1, y1, x2, y2):
    return lambda x: x * y1 / x1 if x <= x1 else (y2 - y1) / (x2 - x1) * (x - x1) + y1

def angles_distance(x, x0):
    return ((x - x0) % (2 * np.pi) + 2 * np.pi)%(2 * np.pi)

def closest_ids(x, arr):
    distances = [angles_distance(x, x0) for x0 in arr]
    id0 = np.argmin(distances)
    id1 = np.argmax(distances)
    return id0, id1

def r_interp(phi, r0, phi0, r1, phi1):
    m = (r0 * np.sin(phi0) - r1 * np.sin(phi1)) / (r0 * np.cos(phi0) - r1 * np.cos(phi1))
    b = - r0 * (m * np.cos(phi0) - np.sin(phi0))
    r = - b / (m * np.cos(phi) - np.sin(phi))
    return r

class Projection:
    def __init__(self, y_pred) -> None:
        
        self.Rs = norm(y_pred)
        self.Phis = arctan2(y_pred)
        pass

    def scale(self, arr, verbose=0):
        rad_ = norm(arr)
        phi_ = arctan2(arr)
        res = []

        iterator = zip(rad_, phi_)
        if verbose: iterator = tqdm(iterator)
        for rad, phi in iterator:
            if phi in self.Phis:
                R = self.Rs[self.Phis==phi].item()
            else:
                Id0, Id1 = closest_ids(phi, self.Phis)
                R0, R1 = self.Rs[Id0], self.Rs[Id1]
                Phi0, Phi1 = self.Phis[Id0], self.Phis[Id1]
                R = r_interp(phi, R0, Phi0, R1, Phi1)

            Rrec = r_of_rectangle(phi, 1, 1)
            r_res = rad * Rrec / R
            res.append([r_res * np.cos(phi), r_res * np.sin(phi)])

        return np.array(res)