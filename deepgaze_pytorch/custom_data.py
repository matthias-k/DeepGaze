"""Helpers for running or adapting DeepGaze models on your own data.

These convert an image directory plus a CSV of fixations into the pysaliency
``FileStimuli`` / ``Fixations`` objects the models and the fine-tuning helper expect, and
fit a center-bias (baseline) log-density over the fixations. They are useful for any DeepGaze
model (II / IIE / III / MSDB), not just MSDB adaptation. If you already have pysaliency
objects, or your own center-bias, skip these and pass your objects directly.
"""
import csv as _csv
import os

import numpy as np
import pysaliency
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel


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


def fit_centerbias(stimuli, fixations, bandwidth=0.1, crossvalidated=True, eps=1e-13):
    """Fit a Gaussian-KDE center-bias (baseline log-density) over the fixations.

    The center-bias captures where fixations land on average, independent of the image; DeepGaze
    models take it as input. This returns a pysaliency model whose ``log_density(stimulus)`` is a
    normalised log-density and which also provides ``information_gain(...)`` for baseline scoring.

    Args:
        stimuli, fixations: as returned by ``load_fixations_csv`` (or your own).
        bandwidth: KDE bandwidth in image-diagonal units. Smaller = sharper; datasets with many
            fixations can afford a smaller value.
        crossvalidated: if True, return a ``CrossvalidatedBaselineModel`` (leave-one-image-out,
            recommended for small datasets); if False, a plain ``BaselineModel``.
        eps: regularisation mixed with a uniform density.

    Returns:
        A fitted pysaliency baseline model to pass as the center-bias.
    """
    cls = CrossvalidatedBaselineModel if crossvalidated else BaselineModel
    return cls(stimuli, fixations, bandwidth=bandwidth, eps=eps)
