from abc import abstractmethod
from typing import Iterable, Union

import cvxpy as cp
import numpy as np
from cca_zoo.models import CCA
from cca_zoo.utils import _check_views, _process_parameter
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted


class CCASample(CCA):

    def sample_weights(self, views):
        scores = self.transform(views)
        return np.prod(scores, axis=0)[:, 0] > 0


class CCAMMR(CCASample):
    """
    CCA by Maximum Margin Robots

    Reference
    ---------
    Szedmak, Sandor, Tijl De Bie, and David R. Hardoon. "A metamorphosis of Canonical Correlation Analysis into multivariate maximum margin learning." ESANN. 2007.
    """

    def __init__(
            self,
            scale: bool = True,
            centre=True,
            copy_data=True,
            random_state=None,
            latent_dims=1,
            C=0.0,
            norm='fro',
            normalise_rows=True,
    ):
        """
        Parameters
        ----------
        scale : bool
            Whether to scale the data to unit variance
        centre : bool
            Whether to centre the data
        copy_data : bool
            Whether to copy the data
        random_state : int
            Random state for initialisation
        latent_dims : int
            Number of latent dimensions to use
        C : float
            Regularisation parameter
        norm : str
            Norm to use for regularisation
        """
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
        )
        self.C = C
        self.norm = norm
        self.normalise_rows = normalise_rows

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        if self.normalise_rows:
            views = [view / np.linalg.norm(view, axis=1, keepdims=True) for view in views]
        self.weights, self.psi = self.primal_solve(*views, C=self.C)
        return self

    def primal_solve(self, X, Y, C=0.0):
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]
        W = cp.Variable((p, q))
        psi = cp.Variable(n)
        constraints = [X_ @ W @ Y_ >= 1 - psi_ for X_, Y_, psi_ in zip(X, Y, psi)] + [
            psi >= 0
        ]
        prob = cp.Problem(
            cp.Minimize(cp.norm(W, self.norm) ** 2 / 2 + C * cp.sum(psi)), constraints
        )
        prob.solve(verbose=True)
        return W.value, psi.value

    def sample_weights(self, views):
        z = self.transform(views)
        return np.diag(z[0] @ z[1].T) - 1 > 0

    def transform(self, views: Iterable[np.ndarray], **kwargs):
        views = self._validate_inputs(views)
        if self.normalise_rows:
            views = [view / np.linalg.norm(view, axis=1, keepdims=True) for view in views]
        views[0] = views[0] @ self.weights
        return views


class CCAKMMR(CCAMMR):
    """
    CCA by Kernelized Maximum Margin Robots

    Reference
    ---------
    Szedmak, Sandor, Tijl De Bie, and David R. Hardoon. "A metamorphosis of Canonical Correlation Analysis into multivariate maximum margin learning." ESANN. 2007.

    """

    def __init__(
            self,
            scale: bool = True,
            centre=True,
            copy_data=True,
            random_state=None,
            C=0.0,
            norm='fro',
            kernel: Iterable[Union[float, callable]] = None,
            gamma: Iterable[float] = None,
            degree: Iterable[float] = None,
            coef0: Iterable[float] = None,
            kernel_params: Iterable[dict] = None,
            normalise_rows=True,
    ):
        """

        Parameters
        ----------
        scale : bool
            Whether to scale the data to unit variance
        centre : bool
            Whether to centre the data
        copy_data : bool
            Whether to copy the data
        random_state : int
            Random state for initialisation
        latent_dims : int
            Number of latent dimensions to use
        C : float
            Regularisation parameter
        norm : str
            Norm to use for regularisation
        kernel : Union[float, callable]
            Kernel to use
        gamma : float
            Kernel coefficient for rbf, poly and sigmoid kernels
        degree : float
            Degree for poly kernels
        coef0 : float
        """
        super().__init__(
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            C=C,
            norm=norm,
            normalise_rows=normalise_rows,
        )
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree

    def _check_params(self):
        self.kernel = _process_parameter("kernel", self.kernel, "linear", 1)
        self.gamma = _process_parameter("gamma", self.gamma, None, 1)
        self.coef0 = _process_parameter("coef0", self.coef0, 1, 1)
        self.degree = _process_parameter("degree", self.degree, 1, 1)
        self.c = _process_parameter("c", self.c, 0, 1)

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        if self.normalise_rows:
            views = [view / np.linalg.norm(view, axis=1, keepdims=True) for view in views]
        self._check_params()
        self.train_views = views
        Kx = self._get_kernel(0, self.train_views[0])
        self.weights = self.dual_solve(Kx,self.train_views[1], C=self.C)
        return self

    @staticmethod
    def dual_solve(Kx, y, C=0.0):
        n = Kx.shape[0]
        alpha = cp.Variable(n)
        Q = Kx * (y@y.T)
        constraints = [alpha <= C, alpha >= 0]
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha)), constraints
        )
        prob.solve()
        return alpha.value

    def _get_kernel(self, view, X, Y=None):
        if callable(self.kernel[view]):
            params = self.kernel_params[view] or {}
        else:
            params = {
                "gamma": self.gamma[view],
                "degree": self.degree[view],
                "coef0": self.coef0[view],
            }
        return pairwise_kernels(
            X, Y, metric=self.kernel[view], filter_params=True, **params
        )

    def transform(self, views: np.ndarray, **kwargs):
        check_is_fitted(self, attributes=["weights"])
        views = self._validate_inputs(views)
        if self.normalise_rows:
            views = [view / np.linalg.norm(view, axis=1, keepdims=True) for view in views]
        Kx = self._get_kernel(0, self.train_views[0], Y=views[0])
        y_pred = Kx.T @ (self.train_views[1].T*self.weights).T
        return y_pred, views[1]


class LCA(CCASample):
    """
    CCA by Large Correlation Analysis

    Reference
    ---------
    Chen, Xiaohong, Songcan Chen, and Hui Xue. "Large correlation analysis." Applied mathematics and computation 217.22 (2011): 9041-9052.

    """
    pass
