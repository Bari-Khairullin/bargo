import numpy as np
import itertools

def embed(pts, dim):
    res = np.empty(shape=(pts.shape[0], dim))
    res[:, :pts.shape[1]] = pts
    return res


def circle(n: int, R: float):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    res = [[x, y] for (x, y) in zip(R*np.cos(phi), R*np.sin(phi))]
    return np.array(res)

def elephant(n: int, R1: float, R2: float):
    r = (R1 - R2) / 2
    s = R1 + R2 + 2 * r 
    coos = []

    Rs = [R1, r, R2, r]
    Ns = [int(n * R / s) for R in Rs]
    Ns[3] = n - Ns[0] - Ns[1] - Ns[2]
    arcs = [(0, np.pi), (np.pi, 2 * np.pi), (np.pi, 0), (np.pi, 2 * np.pi)]
    DX = [0, -R2-r, 0, R2 + r]

    for R, N, arc, dx in zip(Rs, Ns, arcs, DX):
        phi = np.linspace(arc[0], arc[1], N, endpoint=False)
        for p in phi:
            coos.append((R * np.cos(p) + dx, R * np.sin(p)))
    return np.array(coos)

def flower(n: int, R: float, A:float, k: int):
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rho = [R]*n
    rho = rho + A * np.sin(k*phi)
    res = [[x, y] for (x, y) in zip(rho*np.cos(phi), rho*np.sin(phi))]
    return np.array(res)

def flat_grid(x0, x1, dx, y0, y1, dy):
    X, Y = np.mgrid[x0:x1+dx:dx, y0:y1+dy:dy]
    grid = []
    
    for x_ in X:
        grid.append(list(zip(x_, Y[0])))
    for y_ in Y.T:
        grid.append(list(zip(X.T[0], y_)))
    return grid

def rectangle(n: int, a: float, b: float):
    pts = np.empty((0, 2))
    k = n / (2 * a + 2 * b)
    pts = np.append(pts, [[b/2, y] for y in np.linspace(0, a/2, int(k * a / 2), endpoint=False)], axis=0)
    pts = np.append(pts, [[x, a/2] for x in np.linspace(b/2, -b/2, int(k * b), endpoint=False)], axis=0)
    pts = np.append(pts, [[-b/2, y] for y in np.linspace(a/2, -a/2, int(k * a), endpoint=False)], axis=0)
    pts = np.append(pts, [[x, -a/2] for x in np.linspace(-b/2, b/2, int(k * b), endpoint=False)], axis=0)
    pts = np.append(pts, [[b/2, y] for y in np.linspace(-a/2, 0, n-pts.shape[0], endpoint=False)], axis=0)
    return pts

def hourglass(n: int, a: float, R: float):
    pts = np.empty((0, 2))
    alpha = np.arcsin(a / 2 / R)
    length =  4 * alpha * R + 2 * a
    n_circle = n * 4 * alpha * R / length
    n_square = n * 2 * a / length
    x0 = a / 2 + np.sqrt(R**2 - a**2 / 4)
    upper_left = [
        [x0 + R * np.cos(phi), R * np.sin(phi)]
        for phi in np.linspace(np.pi, np.pi - alpha, int(n_circle / 4), endpoint=False)
    ]
    pts = np.append(pts, upper_left, axis=0)
    upper = [
        [a / 2 - x, a / 2]
        for x in np.linspace(0, a, int(n_square / 2))
    ]
    pts = np.append(pts, upper, axis=0)
    right = [
        [-x0 + R * np.cos(alpha - phi), R * np.sin(alpha - phi)]
        for phi in np.linspace(0, 2 * alpha, int(n_circle / 2))
    ]
    pts = np.append(pts, right, axis=0)
    bottom = [
        [-a / 2 + x, -a / 2]
        for x in np.linspace(0, a, int(n_square / 2))
    ]
    pts = np.append(pts, bottom, axis=0)
    bottom_right = [
        [x0 + R * np.cos(phi), R * np.sin(phi)]
        for phi in np.linspace(np.pi + alpha, np.pi - alpha, n - len(pts), endpoint=False)
    ]
    pts = np.append(pts, bottom_right, axis=0)
    return pts

def mark(pts, id):
    return np.hstack((pts, np.ones((pts.shape[0], 1)) * id))

def square_mesh(a, b, N):
    h = (b - a) / (N - 1)

    mesh_1d = np.linspace(a, b, N, endpoint=True)
    mesh = np.array(list(itertools.product(mesh_1d, mesh_1d)))

    return mesh, h
