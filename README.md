# Torch-RL Chess
This repo is a chess bot built on top of the Actor-Critic model with ClipPPO loss.

**Note**: The env is copied from the torchrl repo, and modified to accept `san_moves.txt` because the `setup.py` for torchrl v0.7.2 does not ship with the required moves file.

## Getting Started
To train the model, run `train.ipynb`. The model params can be tuned there as well, including the save rate, and the reward function can be found under `env.py`. To simulate a game, run `play.ipynb`. 