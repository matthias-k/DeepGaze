"""Adapt DeepGaze MSDB to a new dataset (dataset-parameter adaptation).

Adaptation adds one per-dataset parameter slot to a pretrained model and trains its 13 scalars
(the multi-scale weights, gaussian sigma, center-bias weight and priority scaling) while the
CLIP+DINOv2 backbone and the saliency network stay frozen. Because the new slot is initialised
from the average of the original datasets, it starts from the model's generalization behaviour
and only needs a little data to specialise. This is the "adapting the dataset-specific
parameters to new data" procedure from the Modeling Saliency Dataset Bias paper (ICCV 2025).

Typical use::

    from deepgaze_pytorch import DeepGazeMSDB
    from deepgaze_pytorch.custom_data import load_fixations_csv, fit_centerbias
    from deepgaze_pytorch.msdb_adaptation import adapt_dataset_parameters

    stimuli, fixations = load_fixations_csv('images/', 'fixations.csv')
    centerbias = fit_centerbias(stimuli, fixations)
    model = DeepGazeMSDB(pretrained=True)
    model, dataset_index = adapt_dataset_parameters(model, stimuli, fixations, centerbias,
                                                    pixel_per_dva=21.75, train_directory='adaptation_run')
    log_density = model(image, centerbias_map, pixel_per_dva=21.75, dataset=dataset_index)
"""
import torch
import torch.nn as nn


class FixedGeometryMSDB(nn.Module):
    """Training-time wrapper that fixes ``pixel_per_dva`` and the dataset index for a whole run.

    The adaptation setting has a single dataset with a single pixels-per-degree value, so neither
    varies per sample. Fixing them lets the wrapper expose the same ``forward`` signature as every
    other DeepGaze model, so the shared training loop can drive MSDB without any dataset plumbing.

    The wrapper is a forward-shim only; it never participates in persistence. ``state_dict`` /
    ``load_state_dict`` delegate to the wrapped model so saved checkpoints match the released
    ``deepgazemsdb.pth`` format instead of gaining a ``model.`` prefix.
    """

    def __init__(self, model, pixel_per_dva, dataset=None):
        super().__init__()
        self.model = model
        self.pixel_per_dva = pixel_per_dva
        self.dataset = dataset

    def forward(self, image, centerbias, x_hist=None, y_hist=None, durations=None, **kwargs):
        return self.model(image, centerbias, pixel_per_dva=self.pixel_per_dva, dataset=self.dataset)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


def adapt_dataset_parameters(model, train_stimuli, train_fixations, centerbias, pixel_per_dva,
                             train_directory, val_stimuli=None, val_fixations=None,
                             dataset_index=None, lr=0.01, milestones=(6, 20, 24, 25, 27),
                             minimum_learning_rate=5e-5, batch_size=4, validation_epochs=1, device=None):
    """Adapt a pretrained ``DeepGazeMSDB`` to a new dataset and return the adapted model.

    Adds a new dataset slot (via ``model.add_dataset()``), trains only its 13 scalars on the given
    training data with the provided center-bias, and returns the (unwrapped) adapted model together
    with the new slot's index. After this call the new dataset is the last slot and ``dataset=None``
    still yields the original generalization average.

    Resumable and re-run-safe (``train_directory`` is required for this): the run is persisted to
    ``train_directory`` (final head-only weights at ``<train_directory>/final.pth``, plus per-epoch
    checkpoints). **Calling this again with the same ``train_directory`` is cheap** -- a completed
    run returns immediately with the adapted weights loaded, and an interrupted one resumes from its
    last checkpoint. So a crash never costs the (potentially hours-long) training, and you can freely
    re-run the call (e.g. a notebook cell) to get the model back without retraining.

    Each call adds a *new* dataset slot, so when re-running pass a freshly constructed
    ``DeepGazeMSDB(pretrained=True)`` (the usual notebook pattern -- construct the model and call
    this in the same cell). Reusing a model that this function already adapted would widen it a
    second time and fail to load the saved (narrower) checkpoint; a clear error is raised in that
    case. To adapt to a *second* real dataset, call this again with a different ``train_directory``.

    **Reloading a saved adapted model:** the adapted per-dataset tensors are one column wider than
    the released model's, so a checkpoint saved from an adapted model (``<train_directory>/final.pth``
    or ``model.head_state_dict()``) can only be loaded into a fresh model that has already had the
    slot added -- call ``model.add_dataset()`` *before* ``load_state_dict(..., strict=False)``.

    Args:
        model: a ``DeepGazeMSDB`` (typically ``pretrained=True``).
        train_stimuli, train_fixations: the new dataset to adapt to.
        centerbias: a center-bias model, e.g. from ``fit_centerbias`` (or your own).
        pixel_per_dva: pixels per degree of visual angle for the dataset's presentation.
        train_directory: directory for checkpoints / logs (persists the run; enables resume).
        val_stimuli, val_fixations: optional held-out data for the validation metric; if omitted,
            the training data is used (fine for this small 13-parameter fit).
        dataset_index: if you already called ``model.add_dataset()`` (e.g. to verify the
            initialisation before training), pass the returned index here so a second slot is not
            added; otherwise a new slot is created automatically.
        lr, milestones, minimum_learning_rate, batch_size, validation_epochs: training schedule.
        device: torch device (defaults to cuda if available).

    Returns:
        ``(model, dataset_index)`` -- the adapted ``DeepGazeMSDB`` (same object as ``model``) and
        the index of the new dataset slot, to pass as ``dataset=`` when using the adapted model.
    """
    import os

    from deepgaze_pytorch.data import ImageDataset, ImageDatasetSampler, FixationMaskTransform
    from deepgaze_pytorch.training import _train

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if val_stimuli is None:
        val_stimuli, val_fixations = train_stimuli, train_fixations

    # add a new dataset slot, unless the caller already added one (e.g. to check the
    # initialisation before training) and passes its index in.
    if dataset_index is None:
        dataset_index = model.add_dataset()
    wrapped = FixedGeometryMSDB(model, pixel_per_dva=pixel_per_dva, dataset=dataset_index).to(device)

    def _loader(st, fx):
        ds = ImageDataset(st, fx, centerbias_model=centerbias,
                          transform=FixationMaskTransform(sparse=False), average='image')
        return torch.utils.data.DataLoader(
            ds, batch_sampler=ImageDatasetSampler(ds, batch_size=batch_size),
            pin_memory=False, num_workers=0)

    train_loader = _loader(train_stimuli, train_fixations)
    val_loader = _loader(val_stimuli, val_fixations)
    train_baseline = centerbias.information_gain(train_stimuli, train_fixations, average='image')
    val_baseline = centerbias.information_gain(val_stimuli, val_fixations, average='image')

    optimizer = torch.optim.Adam(model.dataset_parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(milestones))

    _train(train_directory, wrapped,
           train_loader, train_baseline, val_loader, val_baseline,
           optimizer, lr_scheduler,
           minimum_learning_rate=minimum_learning_rate,
           validation_epochs=validation_epochs,
           state_dict_fn=model.head_state_dict,
           device=device)

    # _train writes final.pth at the end but does not load it back into the model, and it returns
    # early (without training) when final.pth already exists. Load it so the returned model always
    # carries the adapted weights, including on a re-run that skipped a completed training.
    final_path = os.path.join(train_directory, 'final.pth')
    if os.path.exists(final_path):
        try:
            model.load_state_dict(torch.load(final_path, weights_only=True), strict=False)
        except RuntimeError as e:
            # strict=False ignores missing/unexpected keys but NOT shape mismatches: this fires
            # when the model is one slot wider than the saved checkpoint, i.e. an already-adapted
            # model was reused instead of a fresh one.
            raise RuntimeError(
                f"Failed to load adapted weights from {final_path} ({e}). This usually means the "
                f"model was already adapted (its per-dataset tensors are wider than the saved "
                f"checkpoint). adapt_dataset_parameters adds a new slot on every call -- pass a freshly "
                f"constructed DeepGazeMSDB (or the matching dataset_index) when re-running."
            ) from e

    return model, dataset_index
