import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm

import os
import logging
import warnings
import random
import numpy as np
import matplotlib.pyplot as plt
from parse_args import parse_arguments

from dataset import PACS

from models.as_module import ActivationShapingModule
from models.resnet import BaseResNet18
from models.ras_resnet import RASResNet18
from models.da_resnet import DAResNet18
from models.load import load_model

from checkpoints import load_epoch_from_checkpoint
from checkpoints import save_checkpoint

from globals import CONFIG
from globals import update_config


def unpack_batch_to_device(batch):
    x = batch[0]
    y = batch[1]
    return x.to(CONFIG.device), y.to(CONFIG.device)


@torch.no_grad()
def evaluate(model, data):
    model.eval()

    acc_meter = Accuracy(task="multiclass", num_classes=CONFIG.num_classes)
    acc_meter = acc_meter.to(CONFIG.device)

    loss = [0.0, 0]
    for x, y in tqdm(data):
        with torch.autocast(
            device_type=CONFIG.device, dtype=torch.float16, enabled=True
        ):
            x, y = x.to(CONFIG.device), y.to(CONFIG.device)
            logits = model(x)
            acc_meter.update(logits, y)
            loss[0] += F.cross_entropy(logits, y).item()
            loss[1] += x.size(0)

    accuracy = acc_meter.compute()
    loss = loss[0] / loss[1]
    logging.info(f"Accuracy: {100 * accuracy:.2f} - Loss: {loss}")


def train(model, data):

    # Create optimizers & schedulers
    optimizer = torch.optim.SGD(
        model.parameters(), weight_decay=0.0005, momentum=0.9, nesterov=True, lr=0.001
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(CONFIG.epochs * 0.8), gamma=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Load checkpoint (if it exists)
    cur_epoch = load_epoch_from_checkpoint(model, scheduler, optimizer)

    # Optimization loop
    for epoch in range(cur_epoch, CONFIG.epochs):
        for batch_idx, batch in enumerate(tqdm(data["train"])):

            if CONFIG.experiment in ["domain_adapt"]:
                with torch.autocast(
                    device_type=CONFIG.device, dtype=torch.float16, enabled=True
                ):
                    # Eval mode to compute activation maps for the target domain without updating the weights
                    model.eval()
                    targ_x = batch[2].to(CONFIG.device)
                    model.store_activation_maps(targ_x)

            model.train()

            # Compute loss
            with torch.autocast(
                device_type=CONFIG.device, dtype=torch.float16, enabled=True
            ):
                if CONFIG.experiment in ["baseline"]:
                    x, y = unpack_batch_to_device(batch)
                    loss = F.cross_entropy(model(x), y)
                elif CONFIG.experiment in ["random_maps"]:
                    x, y = unpack_batch_to_device(batch)
                    model.register_random_shaping_hooks()
                    loss = F.cross_entropy(model(x), y)
                    model.remove_hooks()
                elif CONFIG.experiment in ["domain_adapt"]:
                    src_x, src_y = x, y = unpack_batch_to_device(batch)
                    model.register_shaping_hooks()
                    loss = F.cross_entropy(model(src_x), src_y)
                    model.remove_shaping_hooks()

            # Optimization step
            scaler.scale(loss / CONFIG.grad_accum_steps).backward()
            if ((batch_idx + 1) % CONFIG.grad_accum_steps == 0) or (
                batch_idx + 1 == len(data["train"])
            ):
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()

        scheduler.step()

        # Test current epoch
        logging.info(f"[TEST @ Epoch={epoch}]")
        evaluate(model, data["test"])

        # Save checkpoint
        save_checkpoint(epoch, model, scheduler, optimizer)


def visualize_activations(activations):
    num_visualizations = len(activations)

    fig, axes = plt.subplots(
        num_visualizations, 3, figsize=(15, 5 * num_visualizations)
    )

    if num_visualizations == 1:
        axes = [axes]

    for idx, activation in enumerate(activations):
        source = activation["source"].cpu().numpy()
        mask = activation["M"].cpu().numpy()
        shaped = activation["shaped"].cpu().numpy()

        source = (source - np.min(source)) / (np.max(source) - np.min(source))
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        shaped = (shaped - np.min(shaped)) / (np.max(shaped) - np.min(shaped))

        axes[idx][0].imshow(np.transpose(source[0], (1, 2, 0)), cmap="viridis")
        axes[idx][0].set_title("Source Activation")
        axes[idx][0].axis("off")

        axes[idx][1].imshow(np.transpose(mask[0], (1, 2, 0)), cmap="viridis")
        axes[idx][1].set_title("Mask")
        axes[idx][1].axis("off")

        axes[idx][2].imshow(np.transpose(shaped[0], (1, 2, 0)), cmap="viridis")
        axes[idx][2].set_title("Shaped Activation")
        axes[idx][2].axis("off")

    plt.tight_layout()
    plt.show()


def visualize(model, batches):
    model.visualize()
    for batch in batches:
        sx, sy, tx = batch
        model.store_activation_maps(tx)
        model.register_shaping_hooks()
        model(sx)
        model.remove_shaping_hooks
        visualize_activations(model.to_visualize)


def main():
    # Load dataset
    data = PACS.load_data()

    model = load_model(CONFIG.experiment)
    model.to(CONFIG.device)

    if not CONFIG.test_only:
        train(model, data)
        # TODO: only for domain_adapt
        visualize(model, data["test"][0:3])
    else:
        evaluate(model, data["test"])


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # Parse arguments
    args = parse_arguments()
    update_config(args)

    # Setup output directory
    CONFIG.save_dir = os.path.join("record", CONFIG.experiment_name)
    os.makedirs(CONFIG.save_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(CONFIG.save_dir, "log.txt"),
        format="%(message)s",
        level=logging.INFO,
        filemode="a",
    )

    # Set experiment's device & deterministic behavior
    if CONFIG.cpu:
        CONFIG.device = torch.device("cpu")

    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    main()
