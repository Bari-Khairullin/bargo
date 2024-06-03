import os
import sys
import pickle
import multiprocessing

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras import Model
import scipy
import numpy as np
import itertools
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from IPython.display import clear_output
sys.path.append('../')
from bargo.utils import scatter
from tqdm import tqdm

class NN:
    def diffeomorphism(C, n_hidden_layers, fourier_modes=0, **args_hidden):
        dl = 1 / n_hidden_layers if n_hidden_layers != 0 else 1
        finite_difference_coefs = [scipy.special.comb(C, i, exact=True) * (-1)**(C + i + 1) for i in range(C)[::-1]]
        
        x = []
        x.append(Input(2, name='input'))

        if fourier_modes:
            expansion = []
            for mode in range(1, fourier_modes + 1):
                expansion.append(tf.math.sin(mode * np.pi * x[0]))
                expansion.append(tf.math.cos(mode * np.pi * x[0]))
            x.append(tf.keras.layers.Concatenate()([
                x[0], 
                *expansion
            ]))

        if args_hidden['units'] > x[-1].shape[1]:
            x.append(tf.pad(x[-1], [[0, 0], [0, args_hidden['units'] - x[-1].shape[1]]]))
        else:
            args_hidden['units'] = x[-1].shape[1]
            print(f'Hidden kernel is too small, set to {args_hidden["units"]}')

        for n in range(C):
            x.append(tf.identity(x[-1]))

        for n in range(n_hidden_layers + 1):

            if n < n_hidden_layers:
                x.append(Dense(name=f'hidden_{n}', **args_hidden)(x[-1]) * (dl**C))
                for i, coef in enumerate(finite_difference_coefs):
                    x[-1] += coef * x[-i-2]
            else:
                x.append(Dense(
                    name='output',
                    units=2,
                    activation=None,
                    kernel_initializer='Identity',
                    bias_initializer='zeros'
                )(x[-1]))

        model = Model(x[0], x[-1])
        return model


class Geometry:
    def push_forward(X: np.array, path_to_save: str) -> None:
        def process(X, path_to_save):
            run_path = os.path.dirname(path_to_save)
            model_path = os.path.join(run_path, 'model.keras')
            model = tf.keras.models.load_model(model_path)
            
            X = tf.Variable(X, dtype='float32')
            print(X)
            print(model)
            with tf.GradientTape() as g1:
                with tf.GradientTape() as g2:
                    Y = model(X)
                Js = g2.batch_jacobian(Y, X)
            dJs = g1.batch_jacobian(Js, X)

            with open(os.path.join(path_to_save, 'metricized_grid.pkl'), 'wb') as f:
                pickle.dump((Y.numpy(), Js.numpy(), dJs.numpy()), f)

            pass
        p = multiprocessing.Process(target=process, args=(X, path_to_save))
        p.start()
        p.join()
        p.terminate()
        pass

    def contract(A: np.array, B: np.array) -> float:
        return np.dot(A.flatten(), B.flatten())

    def cast_metric(Js, dJs):
        N, dim = Js.shape[0], Js.shape[1]

        Gs_inv = np.zeros_like(Js)
        Ws = np.zeros_like(dJs)
        for b in range(N):
            J = Js[b]
            G_inv = np.linalg.inv(J.T @ J)
            Gs_inv[b] = G_inv
            
            dG = np.zeros_like(dJs[0])
            for i in range(dim):
                dJ = dJs[b, :, :, i]
                dG[:, :, i] = dJ.T @ J + J.T @ dJ
            
            for i, j, k in itertools.product(range(dim), repeat=3):
                Ws[b, i, j, k] = 1/2 * Geometry.contract(
                    G_inv[k, :],
                    dG[j, :, i] + dG[i, :, j] - dG[i, j, :]
                )

        return Gs_inv, Ws


class FDSolver:
    def __init__(self, Y, mask, Gs_inv, Ws, h, f, pars) -> None:
        self.Y = Y
        self.mask = mask
        self.Gs_inv = Gs_inv
        self.Ws = Ws
        self.h = h
        self.f = f
        self.pars = pars
        pass


    def get_laplacian_stencil(self, ind_1d):
        G_inv = self.Gs_inv[ind_1d]
        W = self.Ws[ind_1d]
        h = self.h
        res = np.zeros((3, 3))

        res[0, 0] = G_inv[0, 1] / 2
        res[0, 2] = -G_inv[0, 1] / 2
        res[2, 0] = -G_inv[0, 1] / 2
        res[2, 2] = G_inv[0, 1] / 2

        res[0, 1] = G_inv[0, 0] + h * Geometry.contract(G_inv, W[:, :, 0]) / 2
        res[1, 0] = G_inv[1, 1] + h * Geometry.contract(G_inv, W[:, :, 1]) / 2
        res[1, 2] = G_inv[1, 1] - h * Geometry.contract(G_inv, W[:, :, 1]) / 2
        res[2, 1] = G_inv[0, 0] - h * Geometry.contract(G_inv, W[:, :, 0]) / 2

        res[1, 1] = -2 * (G_inv[0, 0] + G_inv[1, 1])

        res /= h**2

        return res


    def get_helmholtz_stencil(self, ind_1d):
        k = self.pars[0]
        res = self.get_laplacian_stencil(ind_1d)
        res[1, 1] += k**2
        return res
    

    def index_1d(i, j, N):
        res = i * N + j
        return res
    
    def get_fd_matrix(self, get_stencil):
        mask = self.mask

        Nx, Ny = mask.shape
        L_width = mask.sum().astype(int)
        L = np.zeros((L_width, L_width))
        n_eq = 0

        for i in range(Nx):
            for j in range(Ny):
                if not mask[i, j]: continue

                ind_1d = FDSolver.index_1d(i, j, Nx)
                stencil = get_stencil(ind_1d)
                for i_loc in range(3):
                    for j_loc in range(3):
                        i_ngb = i + i_loc - 1
                        j_ngb = j + j_loc - 1
                        if not mask[i_ngb, j_ngb]: continue
                        idx = FDSolver.index_1d(i_ngb, j_ngb, Nx)
                        ind_1d = mask.flatten()[:idx].sum().astype(int)
                        L[n_eq, ind_1d] = stencil[i_loc, j_loc]
                n_eq += 1

        return L
    

    def vectorize_function(self, f):
        Y = self.Y[self.mask.flatten().astype(bool)]

        b = [f(*pt) for pt in Y]
        b = np.reshape(b, (-1, 1))
        return b


    def plot(self, u, kind='heatmap', zlim=None):
        Y = self.Y[self.mask.flatten().astype(bool)]
        fig = plt.figure(figsize=(7, 7))
        if kind=='3d':
            ax = fig.add_subplot(projection='3d')
            if zlim:
                ax.set_zlim(zlim)
                ax.scatter(Y[:, 0], Y[:, 1], u, c=u, vmin=zlim[0], vmax=zlim[1])
            else:
                ax.scatter(Y[:, 0], Y[:, 1], u, c=u)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')

        elif kind=='heatmap':
            ax = fig.add_subplot()
            
            if zlim:
                cs = ax.scatter(Y[:, 0], Y[:, 1], c=u, vmin=zlim[0], vmax=zlim[1])
            else:
                cs = ax.scatter(Y[:, 0], Y[:, 1], c=u)
            fig.colorbar(cs)

            ax.set_xlabel('x')
            ax.set_ylabel('y')

        else:
            raise ValueError('No such kind of graph')

        return fig, ax
    
    
class CustomLoss:
    def __init__(self, boundary_loss, mesh_loss=None, eps=None) -> None:
        self.boundary_loss_fn = getattr(self, boundary_loss)
        self.requires_Js = bool(mesh_loss)
        self.mesh_loss_fn = getattr(self, mesh_loss) if self.requires_Js else None
        self.eps = eps if eps else 1
        pass

    def __call__(self, **args):
        loss = self.boundary_loss_fn(args['Y_true'], args['Y_pred'])
        if self.requires_Js:
            loss += self.mesh_loss_fn(args['Js']) * self.eps
        return loss

    def mse(self, Y_true, Y_pred):
        loss = tf.reduce_mean(tf.square(Y_true - Y_pred)) / 2
        return loss
    
    def abs_distance(self, Y_true, Y_pred):
        squares = tf.square(Y_true - Y_pred)
        dist_squares = tf.reduce_sum(squares, axis=1)
        loss = tf.reduce_mean(tf.sqrt(dist_squares))
        return loss
    
    def mae(self, Y_true, Y_pred):
        loss = tf.reduce_mean(tf.abs(Y_true - Y_pred))
        return loss
    
    # mesh losses
    def tr_over_det(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        Gs_traces = tf.reshape(tf.linalg.trace(Gs), (-1, 1))
        Js_dets = tf.reshape(tf.linalg.det(Js), (-1, 1))
        loss_mesh = tf.reduce_mean(Gs_traces / Js_dets)
        return loss_mesh
    
    def scalar_metric_tensor(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        Gs_dets = tf.reshape(tf.linalg.det(Gs), (-1, 1))
        Gs = tf.reshape(Gs, (-1, 4))
        Conformal = tf.reshape(tf.eye(2, 2, batch_shape=(len(Gs),)), (-1, 4)) * Gs_dets
        loss_mesh = tf.reduce_mean(tf.square(Gs - Conformal))
        return loss_mesh
    
    def area_orthogonality(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        Gs_squared = tf.square(Gs)
        Gs_squared_traces = tf.linalg.trace(Gs_squared)
        loss_mesh = tf.reduce_mean(Gs_squared_traces)
        return loss_mesh
    
    def liao(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        Gs_squared = tf.square(Gs)
        loss_mesh = tf.reduce_mean(Gs_squared) / 4
        return loss_mesh
    
    def winslow(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        Gs_traces = tf.linalg.trace(Gs)
        Js_dets = tf.linalg.det(Js)
        loss_mesh = tf.reduce_mean(Gs_traces / Js_dets)
        return loss_mesh
    
    def orthogonality_1(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        g12s = Gs[:, 0, 1]
        loss_mesh = tf.reduce_mean(tf.square(g12s))
        return loss_mesh
    
    def orthogonality_2(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        loss_mesh = tf.reduce_mean(tf.square(Gs[:, 0, 1]) / Gs[:, 0, 0] / Gs[:, 1, 1])
        return loss_mesh

    def orthogonality_2_std(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        loss_mesh = tf.reduce_mean(tf.square(Gs[:, 0, 1]) / Gs[:, 0, 0] / Gs[:, 1, 1])
        std = tf.math.reduce_std(tf.linalg.det(Gs))
        if std > 1:
            loss_mesh += std * 1e-3
        return loss_mesh
    
    def orthogonality_1_over_det(self, Js):
        Gs = tf.matmul(Js, Js, transpose_a=True)
        g12s = Gs[:, 0, 1]
        loss_mesh = tf.reduce_mean(tf.square(g12s) / tf.linalg.det(Js))
        return loss_mesh


class Trainer:
    def __init__(self, model=None, loss_fn=None, opt=None) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.opt = opt

        self.train_loss = []
        self.val_boundary_loss = []
        self.test_boundary_loss = None
        pass

    def prepare_data(self, X, Y, test_size, val_size, meshX, grid_to_plot=None):
        self.X = X
        self.Y = Y
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, random_state=0)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, shuffle=True, random_state=0)

        self.X_train = tf.Variable(X_train, dtype='float32')
        self.Y_train = Y_train

        self.X_val = tf.Variable(X_val, dtype='float32')
        self.Y_val = Y_val

        self.X_test = tf.Variable(X_test, dtype='float32')
        self.Y_test = Y_test

        self.meshX = tf.Variable(meshX, dtype='float32')
        self.grid_to_plot = grid_to_plot
        pass

    def train_step(self, X_train, Y_train, meshX):
        with tf.GradientTape() as g1:
            if self.loss_fn.requires_Js:
                with tf.GradientTape() as g2:
                    g2.watch(self.meshX)
                    meshY = self.model(meshX)
                Js = g2.batch_jacobian(meshY, meshX)
            else:
                Js = None
            Y_train_pred = self.model(X_train)
            loss = self.loss_fn(Y_true=Y_train, Y_pred=Y_train_pred, Js=Js)

        grad = g1.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return tf.stop_gradient(loss).numpy().item()
    
    def eval_boundary(self, X_val, Y_val):
        Y_val_pred = self.model(X_val)
        loss = self.loss_fn.boundary_loss_fn(Y_true=Y_val, Y_pred=Y_val_pred)
        return tf.stop_gradient(loss).numpy().item()
    
    def eval_mesh(self, meshX):
        if self.loss_fn.requires_Js:
            with tf.GradientTape() as g2:
                g2.watch(meshX)
                meshY = self.model(meshX)
            Js = g2.batch_jacobian(meshY, self.meshX)
        else:
            raise ValueError('No mesh loss function specified')
        loss_mesh = self.mesh_loss_fn(Js)
        return tf.stop_gradient(loss_mesh).numpy().item()
    
    def plot_space(self):
        Y_pred = self.model(self.X).numpy()
        deformed_grid = None
        if self.grid_to_plot:
            deformed_grid = []
            for line in self.grid_to_plot:
                deformed_line = self.model.predict(line)
                deformed_grid.append(deformed_line)
        return scatter([('Physical boundary', self.Y), ('Fitted boundary', Y_pred)], deformed_grid, (5, 5))

    def fit(self, epochs, patience, val_boundary_loss_target=1e-8, decay_patience=None, plot_step=None):
        pbar = tqdm(range(epochs))

        self.train_loss.append([])
        self.val_boundary_loss.append([])
        train_loss = self.train_loss[-1]
        val_boundary_loss = self.val_boundary_loss[-1]

        for epoch in pbar:
            train_loss_value = self.train_step(self.X_train, self.Y_train, self.meshX)
            val_boundary_loss_value = self.eval_boundary(self.X_val, self.Y_val)


            train_loss.append(train_loss_value)
            val_boundary_loss.append(val_boundary_loss_value)

            # printing / plotting / early stopping
            val_patience = len(val_boundary_loss) - np.argmin(val_boundary_loss)
            pbar.set_description(
                f"train_loss: {train_loss_value:.4e} val_boundary_loss: {val_boundary_loss_value:.4e} eps: {self.loss_fn.eps:.0e}"
            )
            
            if val_patience > patience:
                print('NO MORE PATIENCE')
                break
            
            if val_boundary_loss_value < val_boundary_loss_target:
                print('Boundary is good enough')
                break
            
            if decay_patience and (val_patience > decay_patience) and (len(val_boundary_loss) % decay_patience == 0):
                self.loss_fn.eps /= 10

            if plot_step and (len(val_boundary_loss) % plot_step == 0):
                clear_output(wait=False)
                fig, ax = self.plot_space()
                plt.show()
                
        
        self.test_boundary_loss = self.eval_boundary(self.X_test, self.Y_test)
        pass