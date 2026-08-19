import csv
import numpy as np
from PIL import Image
from deepgaze_pytorch.custom_data import load_fixations_csv, fit_centerbias


def _make_dataset(tmp_path):
    img_dir = tmp_path / 'images'; img_dir.mkdir()
    for name, (w, h) in [('a.png', (40, 30)), ('b.png', (20, 20))]:
        Image.fromarray(np.zeros((h, w, 3), np.uint8)).save(img_dir / name)
    csv_path = tmp_path / 'fix.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['image', 'x', 'y'])
        w.writerow(['a.png', 10, 5]); w.writerow(['a.png', 12, 7]); w.writerow(['b.png', 3, 4])
    return img_dir, csv_path


def test_load_fixations_csv(tmp_path):
    img_dir, csv_path = _make_dataset(tmp_path)
    stimuli, fixations = load_fixations_csv(str(img_dir), str(csv_path))
    assert len(stimuli) == 2
    assert len(fixations) == 3
    assert list(fixations.x) == [10, 12, 3]
    # a.png (index 0) has 2 fixations, b.png (index 1) has 1
    assert sorted(np.bincount(fixations.n).tolist()) == [1, 2]


def test_fit_centerbias_returns_normalized_log_density(tmp_path):
    img_dir, csv_path = _make_dataset(tmp_path)
    stimuli, fixations = load_fixations_csv(str(img_dir), str(csv_path))
    cb = fit_centerbias(stimuli, fixations, bandwidth=0.1, crossvalidated=False)
    ld = cb.log_density(stimuli.stimuli[0])
    assert ld.shape == tuple(stimuli.shapes[0][:2])
    assert np.isclose(np.logaddexp.reduce(ld.ravel()), 0.0, atol=1e-3)  # sums to 1 in prob space
