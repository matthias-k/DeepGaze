"""Fine-tune (adapt) DeepGaze MSDB to a new dataset.

Adaptation adds one per-dataset parameter slot to a pretrained model and trains its 13 scalars
(the multi-scale weights, gaussian sigma, center-bias weight and priority scaling) while the
CLIP+DINOv2 backbone and the saliency network stay frozen. Because the new slot is initialised
from the average of the original datasets, it starts from the model's generalization behaviour
and only needs a little data to specialise.

Typical use::

    from deepgaze_pytorch import DeepGazeMSDB
    from deepgaze_pytorch.custom_data import load_fixations_csv, fit_centerbias
    from deepgaze_pytorch.msdb_finetuning import finetune_new_dataset

    stimuli, fixations = load_fixations_csv('images/', 'fixations.csv')
    centerbias = fit_centerbias(stimuli, fixations)
    model = DeepGazeMSDB(pretrained=True)
    model = finetune_new_dataset(model, stimuli, fixations, centerbias,
                                 pixel_per_dva=21.75, train_directory='adaptation_run')
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


def finetune_new_dataset(model, stimuli, fixations, centerbias, pixel_per_dva,
                         train_directory, train_stimuli=None, train_fixations=None,
                         val_stimuli=None, val_fixations=None,
                         lr=0.01, milestones=(6, 20, 24, 25, 27), minimum_learning_rate=5e-5,
                         batch_size=4, validation_epochs=1, device=None):
    """Adapt a pretrained ``DeepGazeMSDB`` to a new dataset and return the adapted model.

    Adds a new dataset slot (via ``model.add_dataset()``), trains only its 13 scalars against the
    given data with the provided center-bias, and returns the (unwrapped) adapted model. The
    adapted per-dataset index is ``model.add_dataset``'s return value; after this call the new
    dataset is the last slot and ``dataset=None`` still yields the original generalization average.

    Args:
        model: a ``DeepGazeMSDB`` (typically ``pretrained=True``).
        stimuli, fixations: the new dataset (used for both train and val if splits are not given).
        centerbias: a center-bias model, e.g. from ``fit_centerbias`` (or your own).
        pixel_per_dva: pixels per degree of visual angle for the dataset's presentation.
        train_directory: where checkpoints / logs are written (the final head-only weights land in
            ``<train_directory>/final.pth``).
        train_stimuli/train_fixations/val_stimuli/val_fixations: explicit splits; if omitted, the
            same ``stimuli, fixations`` are used for training and validation.
        lr, milestones, minimum_learning_rate, batch_size, validation_epochs: training schedule.
        device: torch device (defaults to cuda if available).

    Returns:
        The adapted ``DeepGazeMSDB`` (same object as ``model``).
    """
    from deepgaze_pytorch.data import ImageDataset, ImageDatasetSampler, FixationMaskTransform
    from deepgaze_pytorch.training import _train

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train_stimuli is None:
        train_stimuli, train_fixations = stimuli, fixations
        val_stimuli, val_fixations = stimuli, fixations

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

    return model
