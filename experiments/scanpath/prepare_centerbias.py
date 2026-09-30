"""Fill the center-bias caches in advance (CPU, one image after another).

    python experiments/scanpath/prepare_centerbias.py mit1003 osie mit1003_stretched
"""
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root: deepgaze_pytorch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

DATASETS = {
    'mit1003': (common.load_mit1003, common.mit1003_centerbias_model),
    'mit1003_stretched': (common.load_mit1003_stretched, common.mit1003_centerbias_model),
    'osie': (common.load_osie, common.osie_centerbias_model),
}


def main():
    for name in sys.argv[1:]:
        load, build = DATASETS[name]
        stimuli, scanpaths = load()
        loader = common.centerbias_cache(name, stimuli, lambda: build(stimuli, scanpaths))
        for n in tqdm(range(len(stimuli)), desc=name, mininterval=30):
            loader(n)


if __name__ == '__main__':
    main()
