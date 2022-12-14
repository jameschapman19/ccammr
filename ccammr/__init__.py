from abc import abstractmethod
from typing import Iterable, Union

import cvxpy as cp
import numpy as np
from cca_zoo.models import CCA
from cca_zoo.utils import _check_views
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
    ):
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
        )
        self.C = C
        self.norm = norm

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self.weights, self.psi = self.primal_solve(*views, C=self.C)
        return self

    def primal_solve(self, X, Y, C=0.0):
        X/= np.linalg.norm(X, axis=1, keepdims=True)
        Y/= np.linalg.norm(Y, axis=1, keepdims=True)
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
        views=[view/np.linalg.norm(view, axis=1,keepdims=True) for view in views]
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
    ):
        super().__init__(
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            random_state=random_state,
            C=C,
            norm=norm,
        )
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.coef0 = coef0
        self.kernel = kernel
        self.degree = degree

    def fit(self, views: Iterable[np.ndarray], y=None, **kwargs):
        views = self._validate_inputs(views)
        self.train_views = views
        self.alpha = self.dual_solve(*views, C=self.C)
        return self

    @staticmethod
    def dual_solve(Kx, Ky, C=0.0):
        n = Kx.shape[0]
        alpha = cp.Variable(n)
        Q = Kx * Ky
        constraints = [alpha <= C, alpha >= 0]
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(alpha, Q) - cp.sum(alpha)), constraints
        )
        prob.solve()
        return alpha.value

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, **params
        )

    def transform(self, views: np.ndarray, **kwargs):
        check_is_fitted(self, attributes=["weights"])
        views = _check_views(
            *views, copy=self.copy_data, accept_sparse=self.accept_sparse
        )
        views = self._centre_scale_transform(views)
        Ktest = [
            self._get_kernel(i, self.train_views[i], Y=view)
            for i, view in enumerate(views)
        ]
        transformed_views = [
            kernel.T @ self.weights[i] for i, kernel in enumerate(Ktest)
        ]
        return transformed_views


class LCA(CCASample):
    """
    CCA by Large Correlation Analysis

    Reference
    ---------
    Chen, Xiaohong, Songcan Chen, and Hui Xue. "Large correlation analysis." Applied mathematics and computation 217.22 (2011): 9041-9052.

    """
    pass
