# Train Ticket

Gated PRNGs are all you need?  Spoilers: No.

This is a project inspired by the paper "Proving the Lottery Ticket Hypothesis: Pruning is All You Need" (Malach, 2020)
https://proceedings.mlr.press/v119/malach20a/malach20a.pdf

This project provides a high-performance MNIST training script using DeepSpeed and Nvidia DALI.

The goal is to introduce a gated Hadamard product layer on top of the randomly-initialized initial weights of some standard MNIST network architectures.  The idea is that the training process will learn to "prune" the original network weights without modifying them.  This can help to prove the Strong Lottery Ticket Hypothesis described in the paper above, which states that pruning is all you need to learn a good model.

## Results

Okay so I put it into a 4-layer ViT and trained it on MNIST. Observations:

* Baseline converges to 95% accuracy in like 2 epochs.
* Baseline takes 4.8 seconds per epoch (seems a bit long?)
* Train Ticket Linear version takes much longer to converge to 95% (20 epochs).
* Train Ticket Linear version takes 6.6 seconds per epoch.

Similar results from CNN/MLP models:

* Baseline converges to 98.77% accuracy in 38 epochs.
* Train Ticket converges to 98.37% accuracy in 100 epochs.

So the binary model training is much slower, converges much slower, and is not as accurate.

I might revisit this later to see if I can improve the training speed and accuracy, but this seems like a good place to stop for now.

## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

Install CUDA toolkit from https://developer.nvidia.com/cuda-downloads  This also installs the Nvidia kernel drivers, so you may need to uninstall the old drivers and reboot.

Make sure you can use `nvcc` and the version matches `nvidia-smi`.  Version should show 12.3 or newer.

```bash
nvidia-smi
nvcc --version
```

```bash
git clone https://github.com/catid/train_ticket.git
cd train_ticket

conda create -n tt python=3.10 -y && conda activate tt

# Update this from https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu123/torch_stable.html

# Update this from https://github.com/NVIDIA/DALI#installing-dali
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

pip install -U -r requirements.txt
```

## Train

```bash
conda activate tt
./launch_local_train.sh
```

## Evaluate

You can evaluate the model on the test set without using DeepSpeed to validate the results.

```bash
conda activate tt
python evaluate.py
```

## Advanced

Weights & Biases: To enable W&B, run `wandb login` before training to set up your API key. Pass `--wandb` to the training script. You will also need to specify `--project` and `--name` for the experiment as well when using this option, so that it shows up properly in W&B.

I would not recommend using a distributed training setup for this project since the model/dataset are a bit small for that.
