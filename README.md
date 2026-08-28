# pRNN reward-representation analysis

Analysis of predictive RNNs trained to model hippocampal place-cell
representations during reward-directed navigation. Investigates how reward magnitude and repetition shape input-output place-field correlation, spatial reward-cell recruitment, and population dynamics across training.

**Focus of the analysis:**
- Single-field place cells vs. complex (multi-peak) cells
- Population-level analysis
- How the representation shifts over the course of training

This repo is a cleaned-up, standalone, **public** snapshot of my (Hadrien
Padilla's) analysis work from the lab's shared [`dlevenstein/pRNN`](https://github.com/dlevenstein/pRNN) repository — consolidated into one canonical analysis module and four runnable notebooks. The full training framework (network architectures, training loop, environments) lives in the original repo. This repo ships the analysis pipeline, a fixed demo trajectory, and two small trained-network checkpoints so the notebooks are runnable end-to-end.

## Repo structure

```
src/spatial_analysis.py     canonical analysis functions (place fields,
                             cell-type classification, reward-cell
                             classification, reward-proximity geometry,
                             input-output correlation)
notebooks/                  4 notebooks, each runnable top-to-bottom,
                             each demonstrating one analysis against the
                             two included demo networks
data/trajectory/            one fixed exploration trajectory (observations,
                             actions, position, head direction) used to
                             drive every network through the pipeline
nets/                       two small (~34MB each) trained-network
                             checkpoints for the live demos: a no-reward
                             baseline and a reward condition, same seed
figures/example_diagnostics/  example output from the full-scale
                             population-dynamics sweep (notebook 4), one
                             representative image per diagnostic type
```

## Notebooks

1. **`01_place_field_pipeline.ipynb`** — computes input (`obs[2]`, the true grid/place-cell signal) and output (the network's own prediction) place fields
2. **`02_reward_cell_classification.ipynb`** — shuffle-control reward-cell classification (Yaghoubi et al., 2026)
3. **`03_reward_proximity_circles.ipynb`** — what fraction of place-field
peaks land near a reward location
4. **`04_population_dynamics.ipynb`** — population-vector correlation and
single-field cell turnover between the no-reward and reward-trained networks

Each notebook runs the real pipeline against the two included demo networks end-to-end (so it's genuinely reproducible by anyone who clones this repo).

## Setup

This repo depends on the `prnn` package (network architectures, training
harness, environment wrappers) from the lab's
[`dlevenstein/pRNN`](https://github.com/dlevenstein/pRNN) repository — it
isn't vendored here.

```bash
git clone https://github.com/dlevenstein/pRNN.git
cd pRNN && pip install -e .
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

cd /path/to/pRNN-reward-analysis
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:/path/to/pRNN"
jupyter notebook notebooks/
```
