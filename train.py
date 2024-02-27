import wandb

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from mnist_dali_dataloader import MNISTDataLoader

import argparse, random, os, json, time

import deepspeed
from deepspeed import comm, log_dist

from models.create import create_model

## DeepSpeed

def log_0(msg):
    log_dist(msg, ranks=[0])

def log_all(msg):
    log_dist(msg, ranks=[-1])

def is_main_process():
    return comm.get_rank() == 0

# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

def train_one_epoch(opt_forward_and_loss, criterion, train_loader, model_engine, image_dtype):
    train_loss = 0.0

    model_engine.train()

    with torch.set_grad_enabled(True):
        for batch_idx, (labels, images) in enumerate(train_loader):
            images = images.to(image_dtype)
            labels = labels.squeeze().to(torch.long)

            #log_all(f"train_one_epoch: batch_idx = {batch_idx} labels[:4] = {labels[:4]}")

            labels, images = labels.to(model_engine.local_rank), images.to(model_engine.local_rank)

            loss, _ = opt_forward_and_loss(criterion, images, labels, model_engine)

            model_engine.backward(loss)
            model_engine.step()

            train_loss += loss.item()

    return train_loss

def validation_one_epoch(opt_forward_and_loss, criterion, val_loader, model_engine, image_dtype):
    val_loss = 0.0
    correct = 0
    total = 0

    model_engine.eval()

    with torch.set_grad_enabled(False):
        for batch_idx, (labels, images) in enumerate(val_loader):
            images = images.to(image_dtype)
            labels = labels.squeeze().to(torch.long)

            #log_all(f"validation_one_epoch: batch_idx = {batch_idx} labels[:4] = {labels[:4]}")

            labels, images = labels.to(model_engine.local_rank), images.to(model_engine.local_rank)

            loss, predicted = opt_forward_and_loss(criterion, images, labels, model_engine)

            val_loss += loss.item()

            correct += torch.eq(predicted, labels).sum().item()
            total += predicted.size(0)

            if batch_idx == 0:
                test_images = images[:2]
                output_labels = model_engine(test_images)
                examples = (test_images, labels[:2], output_labels[:2])

    return val_loss, correct, total, examples

from torch.optim.lr_scheduler import (SequentialLR, LinearLR, CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, StepLR, MultiStepLR,
                                      ExponentialLR, OneCycleLR)

def build_lr_scheduler(optimizer, scheduler_type, warmup_epochs, total_epochs, **kwargs):
    warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

    if scheduler_type == "StepLR":
        scheduler = StepLR(optimizer, step_size=kwargs.get('step_size', 50), gamma=kwargs.get('gamma', 0.5))
    elif scheduler_type == "MultiStepLR":
        scheduler = MultiStepLR(optimizer, milestones=kwargs.get('milestones', [30, 60]), gamma=kwargs.get('gamma', 0.1))
    elif scheduler_type == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=kwargs.get('gamma', 0.9))
    elif scheduler_type == "OneCycleLR":
        scheduler = OneCycleLR(optimizer, max_lr=kwargs.get('max_lr', 0.01), total_steps=total_epochs+1)
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=kwargs.get('T_0', total_epochs - warmup_epochs), T_mult=kwargs.get('T_mult', 1))
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    combined_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, scheduler], milestones=[warmup_epochs])

    return combined_scheduler

def synchronize_seed(args, rank, shard_id):
    if args.seed < 0:
        seed = get_true_random_32bit_positive_integer()
    else:
        seed = args.seed

    if shard_id == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.long)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long)  # A tensor to receive the value

    seed_tensor = seed_tensor.cuda(rank)

    comm.broadcast(tensor=seed_tensor, src=0)

    seed = int(seed_tensor.item()) + shard_id
    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={shard_id}")
    return seed

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F # Clear high bit
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

def get_absolute_path(relative_path):
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path
    absolute_path = os.path.abspath(os.path.join(script_dir, relative_path))

    return absolute_path

def main(args):
    t0 = time.time()

    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    log_0(f"args: {args}")

    model = create_model(args.arch)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        warmup_epochs=args.warmup,
        total_epochs=args.max_epochs)

    # Modify deepspeed configuration programmatically
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)

    ds_config["fp16"]["enabled"] = not args.fp32

    # Remove deepspeed_config from the args (we pass a dict into deepspeed.initialize)
    args.deepspeed_config = None

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
        model_parameters=model.parameters())

    comm.barrier()

    fp16 = model_engine.fp16_enabled()
    log_0(f'model_engine.fp16_enabled={fp16}')

    if fp16:
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32

    rank = model_engine.local_rank
    shard_id = model_engine.global_rank
    num_gpus = model_engine.world_size
    train_batch_size = model_engine.train_batch_size()
    data_loader_batch_size = model_engine.train_micro_batch_size_per_gpu()
    steps_per_print = model_engine.steps_per_print()

    seed = synchronize_seed(args, rank, shard_id)

    log_all(f"rank = {rank}, num_shards = {num_gpus}, shard_id={shard_id}, train_batch_size = {train_batch_size}, data_loader_batch_size = {data_loader_batch_size}, steps_per_print = {steps_per_print}, seed={seed}")

    # Weights & Biases
    if args.wandb and is_main_process():
        if not args.name or not args.project:
            raise "The --name and --project argument is required when using --wandb"
        wandb.init(project=args.project, name=args.name, config=args)
        wandb.run.log_code = False

    num_loader_threads = os.cpu_count()//2

    dataset_dir = get_absolute_path(args.dataset_dir)
    log_all(f"Loading dataset from: {dataset_dir}")

    train_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        device_id=rank,
        num_threads=num_loader_threads,
        seed=seed,
        file_list=os.path.join(dataset_dir, "train/training_file_list.txt"),
        mode='training',
        shard_id=shard_id,
        num_shards=num_gpus
    )

    val_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        device_id=rank,
        num_threads=num_loader_threads,
        seed=seed,
        file_list=os.path.join(dataset_dir, "validation/validation_file_list.txt"),
        mode='validation',
        shard_id=shard_id,
        num_shards=num_gpus
    )

    criterion = nn.CrossEntropyLoss()
    criterion.cuda(rank)

    def ref_forward_and_loss(criterion, data, labels, model_engine):
        # DeepSpeed: forward + backward + optimize
        outputs = model_engine(data)
        _, predicted = outputs.max(1)
        return criterion(outputs, labels), predicted

    forward_and_loss = ref_forward_and_loss

    # The time this takes to compile sadly offsets the time it saves
    # Without dynamic=True it actually takes *longer* overall
    if not args.nocompile:
        forward_and_loss = torch.compile(forward_and_loss, dynamic=True, fullgraph=False)

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    avg_val_loss = float("inf")
    start_epoch = 0
    end_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.max_epochs):
        end_epoch = epoch
        start_time = time.time()

        train_loss = train_one_epoch(forward_and_loss, criterion, train_loader, model_engine, image_dtype)

        val_loss, correct, total, examples = validation_one_epoch(forward_and_loss, criterion, val_loader, model_engine, image_dtype)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Sync variables between machines
        sum_train_loss = torch.tensor(train_loss).cuda(rank)
        sum_val_loss = torch.tensor(val_loss).cuda(rank)
        sum_correct = torch.tensor(correct).cuda(rank)
        sum_total = torch.tensor(total).cuda(rank)
        comm.all_reduce(tensor=sum_train_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_val_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_correct, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_total, op=comm.ReduceOp.SUM)

        total_train_items = len(train_loader) * num_gpus
        total_val_items = len(val_loader) * num_gpus
        comm.barrier()
        avg_train_loss = sum_train_loss.item() / total_train_items
        avg_val_loss = sum_val_loss.item() / total_val_items
        val_acc = 100. * sum_correct / sum_total

        if is_main_process():
            log_0(f"Epoch {epoch + 1} - TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={val_acc:.2f}%, Time={epoch_time:.2f} sec")

            if args.wandb:
                lr = optimizer.param_groups[0]['lr']
                wandb.log({"avg_train_loss": avg_train_loss, "val_acc": val_acc, "avg_val_loss": avg_val_loss, "epoch": epoch, "wallclock_time": epoch_time, "lr": lr})

        # Check if validation loss has improved
        if val_acc > best_val_acc:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_train_loss = avg_train_loss
            epochs_without_improvement = 0

            log_0(f'New best validation loss: {best_val_loss:.4f}  Validation accuracy: {best_val_acc:.2f}%')

            client_state = {
                'train_version': 1,
                'best_train_loss': best_train_loss,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
                'fp16': fp16,
                'arch': args.arch,
            }
            model_engine.save_checkpoint(save_dir=args.output_dir, client_state=client_state)

            if is_main_process():
                # Write output .pth file
                saved_state_dict = model_engine.state_dict()
                fixed_state_dict = {key.replace("module.", ""): value for key, value in saved_state_dict.items()}
                fixed_state_dict['extra'] = {
                    'fp16': fp16,
                    'arch': args.arch,
                }
                torch.save(fixed_state_dict, args.output_model)
                log_0(f"Wrote model to {args.output_model} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= args.patience:
                log_0(f"Early stopping at epoch {epoch} due to epochs_without_improvement={epochs_without_improvement}")
                break

    if is_main_process():
        log_0(f'Training complete.  Best model was written to {args.output_model}  Final best validation loss: {best_val_loss}, best validation accuracy: {best_val_acc:.2f}%')

        t1 = time.time()
        dt = t1 - t0

        num_params = sum(p.numel() for p in model.parameters())

        if args.wandb:
            wandb.log({"best_val_loss": best_val_loss, "best_val_acc": best_val_acc, "num_params": num_params, "duration": dt, "end_epoch": end_epoch, 'arch': args.arch, 'fp16': fp16})
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MNIST Train Ticket')
    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='Random seed (default: 42)')
    parser.add_argument('--fp32', action='store_true', default=False,
                        help='FP32')
    parser.add_argument("--nocompile", action="store_true", help="Disable torch.compile")
    parser.add_argument('--arch', type=str, default="simple_cnn", metavar='N',
                        help='Model architecture (default: simple_cnn)')

    parser.add_argument("--dataset-dir", type=str, default=str("mnist"), help="Path to the dataset directory")

    parser.add_argument("--output-dir", type=str, default="output_model", help="Path to the output trained model")
    parser.add_argument("--output-model", type=str, default="model.pth", help="Output model file name")

    parser.add_argument("--wandb", action="store_true", help="WanDB logging")
    parser.add_argument('--name', type=str, default=None, metavar='N',
                        help='Name of the run (default: None)')
    parser.add_argument('--project', type=str, default=None, metavar='N',
                        help='Project name (default: None)')

    # Hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--warmup', type=int, default=3, metavar='N',
                        help='warmup epochs (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingWarmRestarts", help="LR scheduler to use for training")
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum epochs to train")
    parser.add_argument("--patience", type=int, default=10, help="Patience for validation loss not decreasing before early stopping")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    main(args)
