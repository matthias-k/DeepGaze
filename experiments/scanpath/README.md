# Scanpath experiments

Training and evaluation of `DeepGazeIIIMSDB` (the scanpath part of DeepGaze III on top of DeepGaze MSDB,
see the main README), and a comparison of DeepGaze III trained on stretched vs. original MIT1003 images.

| Script | Purpose |
|---|---|
| `common.py` | datasets, cross-validation splits, center-bias caches |
| `cache_msdb_saliency.py` | precompute the DeepGaze MSDB priority maps of a dataset |
| `train_msdb_scanpath.py` | train the scanpath part on top of the frozen DeepGaze MSDB (MIT1003) |
| `evaluate.py` | score models on held-out data and compare them with paired statistics |
| `fold_check.py` | find the MIT1003 folds the released models did not train on |
| `prepare_centerbias.py` | fill the center-bias caches in advance |
| `train_densenet.py` | DeepGaze III (DenseNet-201) retrained on stretched vs. original MIT1003 |

## Setup

The scripts run in the image from `docker/Dockerfile`, with the repository mounted at `/work` and a data
directory at `/data`:

```bash
docker build -t deepgaze-train docker
docker run --rm --gpus '"device=0"' --user $(id -u):$(id -g) -e HOME=/data/home \
    -v $PWD:/work -v /path/to/data:/data deepgaze-train python experiments/scanpath/evaluate.py ...
```

pysaliency downloads the datasets into `$DEEPGAZE_DATA/pysaliency_datasets` on first use (`DEEPGAZE_DATA`
defaults to `/data`; the MIT1003 fixations are extracted with the dataset's MATLAB code, which the image
runs with Octave). Caches go to `$DEEPGAZE_CACHE` (default `$DEEPGAZE_DATA/cache`), training runs and
results to `$DEEPGAZE_RUNS` (default `$DEEPGAZE_DATA/runs`).

## Evaluation protocol

- **Datasets.** MIT1003 at its original image sizes (not stretched), with scanpaths that start at the
  initial central fixation (`get_mit1003_with_initial_fixation`), 35 pixels per degree. OSIE (Xu et al.
  2014): 700 images of 800 x 600 pixels, 24 pixels per degree.
- **Held-out data.** The MIT1003 folds are those of pysaliency's 10-fold split that the released
  DeepGaze III uses: its component k was trained without fold k (test) and fold k - 1 (validation).
  `fold_check.py` confirms this from the data: on these two folds each component is 0.07-0.10 bit
  below the average of all components, on every other fold 0.01-0.03 bit above it. DeepGaze MSDB was
  trained on 9 of the 10 folds. Relative to component f, which did not see fold f, it is worst on
  fold 0: -0.23 ± 0.02 bit (mean ± standard error), versus -0.04 to -0.13 bit on the other folds. So
  fold 0 is the one it held out. The scanpath part of `DeepGazeIIIMSDB` was trained on folds 1-8
  with fold 9 for validation. MIT1003 fold 0 is therefore held out for DeepGaze MSDB, DeepGaze III
  component 0 and `DeepGazeIIIMSDB`. OSIE was used by none of the models.
- **Scored fixations.** All fixations except the first of each scanpath, each conditioned on the
  true previous fixations (up to four).
- **Metrics.** Log-likelihood in bit per fixation. Information gain (IG) is measured over the
  dataset's center bias: for MIT1003 the leave-one-image-out estimate of `train_deepgaze3.ipynb`, for
  OSIE a cross-validated `fit_centerbias`. AUC and NSS of the conditional density follow the pysaliency
  conventions. All values are averaged per image, then over images. Likelihoods are computed exactly
  at full resolution (`deepgaze_pytorch/scanpath_utils.py`).
- **Comparisons.** Paired over images: bootstrap 95% confidence interval (10,000 resamples), t-test
  and Wilcoxon signed-rank test.
- **Model inputs.** All models get the same center bias. DeepGaze MSDB models get the dataset's pixels
  per degree; on OSIE, which MSDB was not trained on, they use the averaged dataset parameters
  (`dataset=None`). DeepGaze III sees the images at its training resolution of 35 pixels per degree:
  OSIE images are upscaled by 35/24 and its predictions mapped back to the original pixels. At
  OSIE's own resolution it is 0.13 bit worse, so the reported results use the rescaled version.

## Results: DeepGaze III on DeepGaze MSDB

MIT1003 fold 0 (101 images, 10,527 fixations):

| Model | IG | AUC | NSS |
|---|---|---|---|
| center bias | 0 | 0.800 | 1.26 |
| DeepGaze MSDB (no fixation history) | 1.239 | 0.902 | 2.76 |
| DeepGaze III, component 0 | 1.466 | 0.913 | 3.07 |
| DeepGaze III on DeepGaze MSDB | **1.702** | **0.923** | **3.46** |

OSIE (700 images, 87,821 fixations):

| Model | IG | AUC | NSS |
|---|---|---|---|
| center bias | 0 | 0.723 | 0.78 |
| DeepGaze MSDB (no fixation history) | 2.261 | 0.936 | 3.82 |
| DeepGaze III, component 0 | 2.179 | 0.931 | 3.42 |
| DeepGaze III (all 10 components) | 2.210 | 0.932 | 3.46 |
| DeepGaze III on DeepGaze MSDB | **2.675** | **0.948** | **4.55** |

Paired differences, DeepGaze III on DeepGaze MSDB minus the other model:

| Dataset | Other model | IG (95% CI) | AUC | NSS | Images with higher IG |
|---|---|---|---|---|---|
| MIT1003 fold 0 | DeepGaze III, component 0 | +0.236 (0.201 to 0.273) | +0.011 | +0.38 | 94 of 101 |
| MIT1003 fold 0 | DeepGaze MSDB | +0.463 (0.430 to 0.495) | +0.022 | +0.70 | 101 of 101 |
| OSIE | DeepGaze III (all 10 components) | +0.465 (0.448 to 0.481) | +0.016 | +1.09 | 693 of 700 |
| OSIE | DeepGaze III, component 0 | +0.496 (0.479 to 0.513) | +0.017 | +1.13 | 696 of 700 |
| OSIE | DeepGaze MSDB | +0.414 (0.400 to 0.428) | +0.012 | +0.73 | 692 of 700 |

All confidence intervals of the IG, AUC and NSS differences exclude zero, and all p-values are below 1e-14.
On OSIE, DeepGaze MSDB without any fixation history already beats DeepGaze III with history: the full
model by 0.051 bit (95% CI 0.029 to 0.073) and component 0 by 0.082 bit (0.060 to 0.104).

The released scanpath part (`deepgaze_pytorch/weights/deepgaze3_msdb_head.pth`) is the best epoch of
`train_msdb_scanpath.py --fold 0`. That run trained with Adam at a learning rate of 1e-3, divided by 10
whenever the validation log-likelihood had not improved for three epochs, until it fell below 1e-6. It
stopped after 49 epochs; the best epoch was 45. With the priority maps precomputed, training took
2 h 41 min on an RTX 2080 Ti.

## Reproducing the results

1. Compute the DeepGaze MSDB priority maps. On an RTX 2080 Ti this takes about 20 min for MIT1003 and
   16 min for OSIE:

   ```bash
   python experiments/scanpath/cache_msdb_saliency.py mit1003
   python experiments/scanpath/cache_msdb_saliency.py osie
   ```

2. Evaluate and compare the models:

   ```bash
   python experiments/scanpath/evaluate.py mit1003_fold0 --models centerbias msdb_spatial dg3msdb dg3_component0
   python experiments/scanpath/evaluate.py osie --models centerbias msdb_spatial dg3msdb dg3_component0 dg3_mixture
   python experiments/scanpath/evaluate.py osie --compare dg3msdb dg3_mixture
   ```

   The center biases are computed on first use; `prepare_centerbias.py mit1003 osie` computes them in
   advance. Add `dg3_component0_native` or `dg3_mixture_native` to evaluate DeepGaze III at OSIE's
   own resolution.

3. To train the scanpath part yourself, run `train_msdb_scanpath.py --fold 0`. It writes
   `runs/msdb_scanpath/fold0/best.pth`. Evaluate that run with `evaluate.py ... --models dg3msdb --run
   msdb_scanpath/fold0`; its results are stored as `dg3msdb_msdb_scanpath_fold0`.

4. `fold_check.py` reproduces the check of the held-out folds. It requires step 1 for MIT1003.

## Stretched vs. original MIT1003

`train_deepgaze3.ipynb` stretches every MIT1003 image, with its fixations, to 1024 x 768 or 768 x 1024,
whatever its aspect ratio. `train_densenet.py` trains the same DeepGaze III on the stretched images and
on the original ones with an identical recipe, so that the effect of the stretching can be measured on
held-out, unstretched images. The stages follow the notebook:

```bash
# spatial pretraining on SALICON, shared by both variants
# (the notebook's schedule by default; --milestones and --max-epochs shorten it)
python experiments/scanpath/train_densenet.py salicon --salicon-val-images 1000
# for each variant (original, stretched) and test fold k
python experiments/scanpath/train_densenet.py spatial --variant original --fold 0
python experiments/scanpath/train_densenet.py scanpath --variant original --fold 0
# compare on the held-out unstretched images; several folds can be pooled: mit1003_fold0,mit1003_fold1
python experiments/scanpath/evaluate.py mit1003_fold0 --models densenet_original densenet_stretched
python experiments/scanpath/evaluate.py mit1003_fold0 --compare densenet_original densenet_stretched
```

### Results: stretched vs. original MIT1003

The runs used a shortened SALICON pretraining (`--milestones 6 10 --max-epochs 12 --salicon-val-images 1000`,
best epoch 5, shared by both variants). The MIT1003 stages ran with `--min-lr 1e-6 --seed 0`, so with the
same seed both variants start the scanpath part from the same initialization. Both models are scored on
the held-out, unstretched test folds with the protocol above. The table gives paired differences, the
model trained on original images minus the model trained on stretched ones:

| Test data | IG (95% CI) | AUC | NSS | Images with higher IG |
|---|---|---|---|---|
| MIT1003 fold 0 | +0.017 (0.005 to 0.030) | +0.0004 (-0.0002 to 0.0010) | +0.047 | 61 of 101 |
| MIT1003 fold 1 | +0.026 (0.013 to 0.039) | +0.0011 | +0.055 | 68 of 101 |
| MIT1003 fold 2 | +0.030 (0.013 to 0.048) | +0.0008 | +0.061 | 54 of 101 |
| MIT1003 folds 0-2 | **+0.024 (0.016 to 0.033)** | +0.0008 (0.0004 to 0.0012) | +0.055 (0.036 to 0.074) | 183 of 303 |
| OSIE (fold 0 models) | +0.061 (0.055 to 0.068) | +0.0016 | +0.16 | 530 of 700 |

- **Significance.** On the three MIT1003 folds pooled, p = 5e-8 (t-test) and 1e-6 (Wilcoxon).
  Training without stretching is better on every test fold. The effect is small but consistent.
- **Reproduction.** The stretched variant reproduces the released DeepGaze III closely. On MIT1003
  fold 0 its IG is 1.439, against 1.466 for the released component 0. On OSIE it is 2.187, against
  2.179.
- **What the intervals cover.** The confidence intervals cover the variation over images, not between
  training runs. The three folds, each a separate pair of training runs with the same sign, bound
  the latter only indirectly.
- **DeepGaze III on DeepGaze MSDB.** Its scanpath part is trained on the original, unstretched images
  (`train_msdb_scanpath.py`), so this fix is already part of that model.
