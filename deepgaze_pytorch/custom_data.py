"""Helpers for running or adapting DeepGaze models on your own data.

These convert an image directory plus a CSV of fixations into the pysaliency
``FileStimuli`` / ``Fixations`` objects the models and the fine-tuning helper expect, and
fit a center-bias (baseline) log-density over the fixations. They are useful for any DeepGaze
model (II / IIE / III / MSDB), not just MSDB adaptation. If you already have pysaliency
objects, or your own center-bias, skip these and pass your objects directly.
"""
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
        ``(stimuli, fixations)`` ready for ``fit_centerbias`` / ``adapt_dataset_parameters`` or any
        DeepGaze model.
    """
    import pandas as pd

    data = pd.read_csv(csv_path)
    image_names = list(dict.fromkeys(data[image_column]))  # unique, in first-seen order
    index_of = {name: i for i, name in enumerate(image_names)}
    filenames = [os.path.join(image_dir, name) for name in image_names]

    stimuli = pysaliency.FileStimuli(filenames)
    fixations = pysaliency.Fixations.create_without_history(
        x=data[x_column].to_numpy(dtype=float),
        y=data[y_column].to_numpy(dtype=float),
        n=data[image_column].map(index_of).to_numpy(dtype=int),
    )
    return stimuli, fixations


def fit_centerbias(stimuli, fixations, bandwidth=None, crossvalidated=True, eps=1e-3,
                   bandwidth_bounds=(0.005, 0.3), verbose=False):
    """Fit a Gaussian-KDE center-bias (baseline log-density) over the fixations.

    The center-bias captures where fixations land on average, independent of the image; DeepGaze
    models take it as input. This returns a pysaliency model whose ``log_density(stimulus)`` is a
    normalised log-density and which also provides ``information_gain(...)`` for baseline scoring.

    By default the KDE bandwidth is **fitted** to the data: it is chosen to maximise the mean
    leave-one-image-out crossvalidated log-likelihood. Each dataset thus gets its own bandwidth
    (too-large a bandwidth washes out the central-fixation structure; too-small overfits individual
    fixations).

    Args:
        stimuli, fixations: as returned by ``load_fixations_csv`` (or your own).
        bandwidth: KDE bandwidth as a fraction of the image size. If ``None`` (default) it is
            fitted; pass a float to fix it and skip the search.
        crossvalidated: type of the returned model at the chosen bandwidth. If True, a
            ``CrossvalidatedBaselineModel`` (leave-one-image-out; use it as the center-bias for the
            images it was fit on, no leakage); if False, a plain ``BaselineModel`` (uses all images;
            use it to predict on new held-out images).
        eps: weight of a uniform density mixed into the KDE, both for the returned model and to keep
            the CV objective finite for outlier fixations during the search.
        bandwidth_bounds: ``(low, high)`` bounds for the fitted bandwidth (fraction of image size).
        verbose: print the fitted bandwidth and its CV score.

    Returns:
        A fitted pysaliency baseline model to pass as the center-bias.
    """
    if bandwidth is None:
        from scipy.optimize import minimize_scalar

        # Fast leave-one-image-out CV objective: CrossvalMultipleRegularizations precomputes the
        # fixations in normalised sklearn form once and scores each bandwidth with an sklearn KDE
        # (resolution-independent), rather than re-running a full-resolution gaussian_filter per
        # image per bandwidth. The optimiser works on log10(bandwidth) internally.
        crossvalidation = ScikitLearnImageCrossValidationGenerator(stimuli, fixations, leave_out_size=1)
        manager = CrossvalMultipleRegularizations(
            stimuli, fixations, OrderedDict([('uniform', pysaliency.UniformModel())]), crossvalidation)
        log_eps = float(np.log10(eps))

        def neg_cv_score(log_bandwidth):
            return -manager.score(log_bandwidth=float(log_bandwidth), log_uniform=log_eps)

        result = minimize_scalar(
            neg_cv_score, bounds=(np.log10(bandwidth_bounds[0]), np.log10(bandwidth_bounds[1])),
            method='bounded', options={'xatol': 0.02})
        bandwidth = 10 ** result.x
        if verbose:
            print(f"fit_centerbias: bandwidth={bandwidth:.4f}, CV score={-result.fun:.4f} bit/fix")

    cls = CrossvalidatedBaselineModel if crossvalidated else BaselineModel
    return cls(stimuli, fixations, bandwidth=bandwidth, eps=eps)
