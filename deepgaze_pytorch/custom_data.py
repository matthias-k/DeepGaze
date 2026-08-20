"""Helpers for running or adapting DeepGaze models on your own data.

These convert an image directory plus a CSV of fixations into the pysaliency
``FileStimuli`` / ``Fixations`` objects the models and the fine-tuning helper expect, and
fit a center-bias (baseline) log-density over the fixations. They are useful for any DeepGaze
model (II / IIE / III / MSDB), not just MSDB adaptation. If you already have pysaliency
objects, or your own center-bias, skip these and pass your objects directly.
"""
import csv as _csv
import os
from collections import OrderedDict

import numpy as np
import pysaliency
from pysaliency.baseline_utils import (
    BaselineModel,
    CrossvalidatedBaselineModel,
    CrossvalMultipleRegularizations,
    ScikitLearnImageCrossValidationGenerator,
)


def load_fixations_csv(image_dir, csv_path, image_column='image', x_column='x', y_column='y'):
    """Build ``(FileStimuli, Fixations)`` from an image directory and a CSV of fixations.

    Each CSV row references an image filename (relative to ``image_dir``) and a fixation
    location in pixels. Images are indexed in first-seen order; each fixation's ``n`` points
    at its image.

    Args:
        image_dir: directory containing the images.
        csv_path: CSV with a header row and (at least) the image / x / y columns.
        image_column, x_column, y_column: column names in the CSV.

    Returns:
        ``(stimuli, fixations)`` ready for ``fit_centerbias`` / ``finetune_new_dataset`` or any
        DeepGaze model.
    """
    filenames = []
    index_of = {}
    xs, ys, ns = [], [], []
    with open(csv_path, newline='') as f:
        for row in _csv.DictReader(f):
            name = row[image_column]
            if name not in index_of:
                index_of[name] = len(filenames)
                filenames.append(os.path.join(image_dir, name))
            ns.append(index_of[name])
            xs.append(float(row[x_column]))
            ys.append(float(row[y_column]))

    stimuli = pysaliency.FileStimuli(filenames)
    fixations = pysaliency.Fixations.create_without_history(
        x=np.array(xs), y=np.array(ys), n=np.array(ns, dtype=int))
    return stimuli, fixations


def fit_centerbias(stimuli, fixations, bandwidth=None, crossvalidated=True, eps=1e-13,
                   log_bandwidth_bounds=(-2.5, -0.5), fit_uniform_log_weight=-4.0, verbose=False):
    """Fit a Gaussian-KDE center-bias (baseline log-density) over the fixations.

    The center-bias captures where fixations land on average, independent of the image; DeepGaze
    models take it as input. This returns a pysaliency model whose ``log_density(stimulus)`` is a
    normalised log-density and which also provides ``information_gain(...)`` for baseline scoring.

    By default the KDE bandwidth is **fitted** to the data: it is chosen to maximise the mean
    leave-one-image-out crossvalidated log-likelihood, optimised over ``log10(bandwidth)`` with a
    bounded scalar search. Each dataset thus gets its own bandwidth (too-large a bandwidth washes
    out the central-fixation structure; too-small overfits individual fixations).

    Args:
        stimuli, fixations: as returned by ``load_fixations_csv`` (or your own).
        bandwidth: KDE bandwidth in image-diagonal units. If ``None`` (default) it is fitted; pass
            a float to fix it and skip the search.
        crossvalidated: type of the returned model at the chosen bandwidth. If True, a
            ``CrossvalidatedBaselineModel`` (leave-one-image-out; use it as the center-bias for the
            images it was fit on, no leakage); if False, a plain ``BaselineModel`` (uses all images;
            use it to predict on new held-out images).
        eps: regularisation mixed with a uniform density in the returned model.
        log_bandwidth_bounds: ``(low, high)`` bounds for ``log10(bandwidth)`` during the search.
        fit_uniform_log_weight: log10 of the fixed uniform mixture weight used only to keep the CV
            objective finite during the bandwidth search (not the returned model's ``eps``).
        verbose: print the fitted bandwidth and its CV log-likelihood.

    Returns:
        A fitted pysaliency baseline model to pass as the center-bias.
    """
    if bandwidth is None:
        from scipy.optimize import minimize_scalar

        # Fast leave-one-image-out CV objective: CrossvalMultipleRegularizations precomputes the
        # fixations in normalised sklearn form once and scores each bandwidth with an sklearn KDE
        # (resolution-independent), rather than re-running a full-resolution gaussian_filter per
        # image per bandwidth. A tiny fixed uniform mixture keeps the CV log-likelihood finite for
        # outlier fixations; we optimise only the (1-D) bandwidth.
        crossvalidation = ScikitLearnImageCrossValidationGenerator(stimuli, fixations, leave_out_size=1)
        manager = CrossvalMultipleRegularizations(
            stimuli, fixations, OrderedDict([('uniform', pysaliency.UniformModel())]), crossvalidation)

        def neg_cv_score(log_bandwidth):
            return -manager.score(log_bandwidth=float(log_bandwidth), log_uniform=fit_uniform_log_weight)

        result = minimize_scalar(neg_cv_score, bounds=log_bandwidth_bounds,
                                 method='bounded', options={'xatol': 0.02})
        bandwidth = 10 ** result.x
        if verbose:
            print(f"fit_centerbias: bandwidth={bandwidth:.4f} "
                  f"(log10={result.x:.3f}), CV score={-result.fun:.4f} bit/fix")

    cls = CrossvalidatedBaselineModel if crossvalidated else BaselineModel
    return cls(stimuli, fixations, bandwidth=bandwidth, eps=eps)
