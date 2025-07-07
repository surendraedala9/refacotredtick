"""Simplified pure Python implementation of HawkesADM4.
This version only stores parameters and provided data without performing
any optimization. It is meant for educational or testing purposes and
keeps the public API of the original class minimal.
"""

import numpy as np


class HawkesADM4:
    """Minimal placeholder for the HawkesADM4 learner.

    Parameters
    ----------
    decay : float
        Decay parameter of the exponential kernels.
    """

    def __init__(self, decay, C=1e3, lasso_nuclear_ratio=0.5, max_iter=50,
                 tol=1e-5, n_threads=1, verbose=False, print_every=10,
                 record_every=10, rho=0.1, approx=0, em_max_iter=30,
                 em_tol=None):
        self.decay = decay
        self.C = C
        self.lasso_nuclear_ratio = lasso_nuclear_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads
        self.verbose = verbose
        self.print_every = print_every
        self.record_every = record_every
        self.rho = rho
        self.approx = approx
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol

        self.events = None
        self.end_times = None
        self.baseline = None
        self.adjacency = None

    # ------------------------------------------------------------------
    # Fitting utilities
    # ------------------------------------------------------------------
    def fit(self, events, end_times=None, baseline_start=None,
            adjacency_start=None):
        """Store the provided events and initialise parameters.

        This simplified version does not run the original optimisation
        algorithm. Baseline and adjacency are simply initialised from the
        provided starting points or default values.
        """
        self.events = events
        if end_times is None:
            end_times = [max(map(max, r)) if r else 0.0 for r in events]
        self.end_times = end_times

        n_nodes = len(events[0]) if events else 0
        if baseline_start is None:
            baseline_start = np.ones(n_nodes)
        if adjacency_start is None:
            adjacency_start = np.random.uniform(0.5, 0.9, (n_nodes, n_nodes))

        self.baseline = baseline_start.copy()
        self.adjacency = adjacency_start.copy()
        return self

    # ------------------------------------------------------------------
    # Convenience helpers compatible with the original API
    # ------------------------------------------------------------------
    @property
    def n_nodes(self):
        return len(self.baseline) if self.baseline is not None else 0

    @property
    def coeffs(self):
        if self.baseline is None or self.adjacency is None:
            return None
        return np.hstack((self.baseline, self.adjacency.ravel()))

    def score(self, *args, **kwargs):
        """Placeholder score returning 0.0."""
        return 0.0
