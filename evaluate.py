# Pretty logging

import logging

import colorama
from colorama import Fore, Style

colorama.init()

class ColoredFormatter(logging.Formatter):
    level_colors = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        if not record.exc_info:
            record.msg = f"{ColoredFormatter.level_colors[record.levelno]}{record.msg}{Style.RESET_ALL}"
        return super(ColoredFormatter, self).format(record)

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Model

import torch
import torch.nn as nn

from models.create import create_model

def load_model(args):
    model_path = args.model

    arch = ""
    fp16 = True

    data = torch.load(model_path)
    if 'extra' in data:
        key = data['extra']
        arch = key["arch"]
        fp16 = key["fp16"]
        del data['extra']

    model = create_model(arch)

    model.load_state_dict(data)
    if fp16:
        model.half()
    model.eval()

    for name, param in model.named_parameters():
        logging.info(f"Name: {name}, dtype: {param.dtype}, size: {param.size()}")

    return model, arch, fp16

# Evaluation

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def read_data_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                # The first part is the file path, and the second part is the label
                file_path, label = parts[0], int(parts[1])
                data.append((file_path, label))
    return data

def evaluate(model, dataset_dir, fp16=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    validation_file = os.path.join(dataset_dir, "validation/validation_file_list.txt")

    data = read_data_file(validation_file)

    test_loss = 0
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    batch_input_tensors = []
    batch_label_tensors = []
    batch_size = 256

    for file_path, label in tqdm(data, "Evaluating"):
        file_path = os.path.join(dataset_dir, "validation", file_path)
        input_image = Image.open(file_path).convert("L")

        input_tensor = torch.from_numpy(np.array(input_image)).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimension
        input_tensor = (input_tensor - 128.0) / 128.0  # Normalize to [-1, 1]

        if fp16:
            input_tensor = input_tensor.to(torch.float16)
        else:
            input_tensor = input_tensor.to(torch.float32)

        label_tensor = torch.tensor([label]).to(torch.long).to(device)

        batch_input_tensors.append(input_tensor)
        batch_label_tensors.append(label_tensor)

        if len(batch_input_tensors) == batch_size:
            # Convert lists to tensors and concatenate
            inputs = torch.cat(batch_input_tensors, dim=0).to(device)
            labels = torch.cat(batch_label_tensors, dim=0).to(device)

            with torch.no_grad():
                results = model(inputs)
                loss = criterion(results, labels)
                _, predicted = results.max(1)

            correct += torch.eq(predicted, labels).sum().item()
            test_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            # Clear lists for next batch
            batch_input_tensors = []
            batch_label_tensors = []

    # Process the last batch if it has fewer than batch_size elements
    if batch_input_tensors:
        inputs = torch.cat(batch_input_tensors, dim=0).to(device)
        labels = torch.cat(batch_label_tensors, dim=0).to(device)

        with torch.no_grad():
            results = model(inputs)
            loss = criterion(results, labels)
            _, predicted = results.max(1)

        correct += torch.eq(predicted, labels).sum().item()
        test_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

    logger.info(f"Test loss = {test_loss/total}")
    logger.info(f"Test accuracy: {100.*correct/total}%")

# Entrypoint

import argparse

def main(args):
    # This mutates the args to update e.g. args.fp16 from the .pth file
    model, arch, fp16 = load_model(args)

    num_params = sum(p.numel() for p in model.parameters())

    #for name, param in model.named_parameters():
    #    print(f"{name}: shape={param.shape} numel={param.numel()}")

    logger.info(f"Loaded model with arch={arch} fp16={fp16} model size = {num_params} weights")

    evaluate(model, args.dataset_dir, fp16=fp16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model", type=str, default="model.pth", help="Path to model file")
    parser.add_argument("--dataset-dir", type=str, default=str("mnist"), help="Path to the dataset directory (default: ./mnist/)")

    args = parser.parse_args()

    main(args)
