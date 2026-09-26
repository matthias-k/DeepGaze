# Training / evaluation environment for DeepGaze (GPU, Turing or newer; driver >= 550).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# gcc/g++: pysaliency builds Cython extensions; octave: pysaliency extracts the MIT1003
# fixations with the dataset's original MATLAB code, which calls nanmedian from the statistics
# package (loaded at startup, pysaliency does not load it); git: CLIP is installed from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ git octave octave-statistics \
    && rm -rf /var/lib/apt/lists/* \
    && echo "pkg load statistics" >> /usr/share/octave/site/m/startup/octaverc

# numpy<2: pysaliency 0.2.22 (PyPI) uses NumPy 1 APIs (np.string_ when storing datasets, np.trapz in
# the ROC code); gdown: pysaliency downloads SALICON from Google Drive; h5py: datasets are stored as HDF5
RUN pip install --no-cache-dir \
        "numpy<2" boltons ipython lmdb matplotlib pandas pillow pysaliency pyyaml \
        scikit-learn scipy tensorboard tqdm pytest gdown h5py \
        "clip @ git+https://github.com/openai/CLIP.git"

WORKDIR /work
