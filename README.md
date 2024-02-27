# train_ticket

Train Ticket: Sigmoid gates are all you need?

## Setup

Install conda: https://docs.conda.io/projects/miniconda/en/latest/index.html

Make sure your Nvidia drivers are installed properly by running `nvidia-smi`.  If not, you may need to run something like `sudo apt install nvidia-driver-535-server` or newer.

Install CUDA toolkit:

```bash
# Here's how to do it on Ubuntu:
sudo apt install nvidia-cuda-toolkit
```

Make sure you can use `nvcc`:

```bash
nvcc --version
```

```bash
git clone https://github.com/catid/train_ticket.git
cd train_ticket

conda create -n tt python=3.10 -y && conda activate tt

# Update this from https://pytorch.org/get-started/locally/
pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu118

# Update this from https://github.com/NVIDIA/DALI#installing-dali
pip install --upgrade nvidia-dali-cuda110 --extra-index-url https://developer.download.nvidia.com/compute/redist

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
