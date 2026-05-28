"""
historical_baseline.py
----------------------
Non-parametric historical day-matching baseline.

For a given query conditioning (cluster_id, day_type, season, temperature),
the baseline retrieves real training profiles that share the same discrete
conditioning and have a temperature within ±temp_tol of the query value,
then returns a random draw from those matching profiles.

This baseline requires NO training — it simply memorises the training set
and retrieves from it at inference time.  It is competitive on small datasets
because it has no inductive bias, but degenerates for unseen combinations.

Public API
----------
HistoricalBaseline
    .fit(xs, c_discrete, c_continuous, dates, meter_ids)
    .sample(c_discrete_query, c_continuous_query, n_samples, temp_tol=1.0)
    .generate(c_batch, key, temp_tol=1.0)   — drop-in replacement for DM sampler
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class HistoricalBaseline:
    """
    Historical day-matching generative model.

    Stores all training profiles and retrieves matching samples at inference.
    No model parameters to train or save.

    Matching criteria (in order of strictness):
      1. Exact match on (cluster_id, day_type, season).
      2. Temperature proximity: |temp_query − temp_stored| ≤ temp_tol.
      3. If fewer than ``min_pool`` candidates remain, progressively relax
         temp_tol until at least ``min_pool`` or the full discrete-match set.
    """

    def __init__(self, min_pool: int = 5, seed: int = 0):
        self.min_pool = min_pool
        self.rng = np.random.default_rng(seed)
        self._fitted = False

    def fit(
        self,
        xs: np.ndarray,              # (N, 24) float32 — normalised profiles
        c_discrete: np.ndarray,      # (N, 3)  int32   [cluster_id, day_type, season]
        c_continuous: np.ndarray,    # (N, 1)  float32 [daily_mean_temp_normed]
        dates: Optional[np.ndarray] = None,      # (N,) — for reference, not used in matching
        meter_ids: Optional[np.ndarray] = None,  # (N,) — for reference
    ) -> "HistoricalBaseline":
        """Store all training instances."""
        self._xs         = np.asarray(xs,          dtype=np.float32)
        self._c_disc     = np.asarray(c_discrete,  dtype=np.int32)
        self._c_cont     = np.asarray(c_continuous, dtype=np.float32)
        self._dates      = dates
        self._meter_ids  = meter_ids
        self._fitted     = True
        return self

    def sample(
        self,
        c_discrete_query: np.ndarray,   # (n_query, 3) int32 or (3,)
        c_continuous_query: np.ndarray, # (n_query, 1) float32 or (1,)
        n_samples: int = 1,
        temp_tol: float = 1.0,          # max |temp - temp_query| for a match
    ) -> np.ndarray:
        """
        Return (n_samples, 24) float32 array of matched historical profiles.

        For each query condition, the method:
          1. Selects stored profiles with matching (cluster_id, day_type, season).
          2. Among those, filters by temperature proximity.
          3. Randomly samples with replacement from the resulting pool.

        If the pool is still smaller than ``min_pool`` after applying temp_tol,
        it doubles temp_tol up to 3 times before falling back to any sample
        matching the discrete conditions only.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .sample()")

        c_disc_q = np.atleast_2d(np.asarray(c_discrete_query, dtype=np.int32))
        c_cont_q = np.atleast_2d(np.asarray(c_continuous_query, dtype=np.float32))

        n_query  = c_disc_q.shape[0]
        out_list = []

        for qi in range(n_query):
            cid, dt, season = int(c_disc_q[qi, 0]), int(c_disc_q[qi, 1]), int(c_disc_q[qi, 2])
            temp_q = float(c_cont_q[qi, 0])

            # Step 1: discrete match
            disc_mask = (
                (self._c_disc[:, 0] == cid) &
                (self._c_disc[:, 1] == dt)  &
                (self._c_disc[:, 2] == season)
            )
            disc_idx = np.where(disc_mask)[0]

            if len(disc_idx) == 0:
                # No discrete match — fall back to any profile
                disc_idx = np.arange(len(self._xs))

            # Step 2: temperature proximity with adaptive relaxation
            current_tol = temp_tol
            for _ in range(4):
                temp_diff = np.abs(self._c_cont[disc_idx, 0] - temp_q)
                cand_idx  = disc_idx[temp_diff <= current_tol]
                if len(cand_idx) >= self.min_pool:
                    break
                current_tol *= 2.0

            if len(cand_idx) == 0:
                cand_idx = disc_idx  # final fallback: discrete match only

            # Step 3: draw with replacement
            chosen = self.rng.choice(len(cand_idx), size=n_samples, replace=True)
            out_list.append(self._xs[cand_idx[chosen]])   # (n_samples, 24)

        # If n_query == 1 and n_samples == 1, return shape (1, 24)
        if n_query == 1:
            return out_list[0]

        # If multiple queries, return (n_query * n_samples, 24) stacked
        return np.concatenate(out_list, axis=0)

    def generate(
        self,
        c_batch: np.ndarray,          # (B, >=2) int32  — first 3 cols used as c_discrete
        key: int = 0,                 # ignored (kept for API compatibility)
        c_continuous_batch: Optional[np.ndarray] = None,  # (B, 1) float32, optional
        temp_tol: float = 1.0,
    ) -> np.ndarray:
        """
        Drop-in replacement for the diffusion/RF sampler callable.

        Parameters
        ----------
        c_batch            : (B, >=3) int32 discrete conditioning
        key                : unused (kept for interface compatibility)
        c_continuous_batch : (B, 1) float32 continuous conditioning (temperature).
                             If None, temperature is assumed 0 (global mean).
        temp_tol           : temperature tolerance for matching

        Returns
        -------
        (B, 24) float32 generated (retrieved) profiles
        """
        B = c_batch.shape[0]
        c_disc_q = np.asarray(c_batch[:, :3], dtype=np.int32)
        if c_continuous_batch is not None:
            c_cont_q = np.asarray(c_continuous_batch, dtype=np.float32)
            if c_cont_q.ndim == 1:
                c_cont_q = c_cont_q[:, None]
        else:
            c_cont_q = np.zeros((B, 1), dtype=np.float32)

        results = []
        for i in range(B):
            prof = self.sample(c_disc_q[i], c_cont_q[i], n_samples=1, temp_tol=temp_tol)
            results.append(prof[0])  # (24,)

        return np.stack(results, axis=0)   # (B, 24)
