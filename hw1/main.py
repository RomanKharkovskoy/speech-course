import logging
import os
from datetime import datetime

import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from melbanks import LogMelFilterBanks
from dataset import AudioDataset
from model import CNN, count_parameters, count_flops
from train import train_model, evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    filename=f"logs/log_{int(datetime.now().timestamp())}"
)
logger = logging.getLogger(__name__)

IMAGES_DIR = "images"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3


def save_fig(name):
    path = os.path.join(IMAGES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def verify_melbanks():
    logger.info("Verifying LogMelFilterBanks against torchaudio.transforms.MelSpectrogram")

    sr = 16000
    t = torch.linspace(0, 1, sr)
    signal = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

    melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
    logmelbanks = LogMelFilterBanks()(signal)
    reference = torch.log(melspec + 1e-6)

    assert reference.shape == logmelbanks.shape, (
        f"Shape mismatch: {logmelbanks.shape} vs {reference.shape}"
    )
    assert torch.allclose(reference, logmelbanks), (
        f"Value mismatch, max diff: {(reference - logmelbanks).abs().max().item()}"
    )
    logger.info("LogMelFilterBanks verification passed")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].imshow(logmelbanks[0].detach().numpy(), aspect="auto", origin="lower")
    axes[0].set_title("LogMelFilterBanks (melbanks.py)")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Mel bin")
    axes[1].imshow(reference[0].detach().numpy(), aspect="auto", origin="lower")
    axes[1].set_title("log(MelSpectrogram) (torchaudio)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel bin")
    plt.tight_layout()
    save_fig("melbanks_comparison.png")


def experiment_n_mels(train_loader, val_loader, test_loader, device):
    logger.info("Starting n_mels experiment")

    n_mels_list = [20, 40, 80]
    results = {}

    for n_mels in n_mels_list:
        logger.info(f"n_mels={n_mels}")
        model = CNN(n_mels=n_mels, groups=1)
        logger.info(f"Parameters: {count_parameters(model)}")

        history = train_model(model, train_loader, val_loader, NUM_EPOCHS, LR, device)
        test_acc = evaluate(model, test_loader, device)
        logger.info(f"Test accuracy: {test_acc:.4f}")

        results[n_mels] = {"history": history, "test_acc": test_acc}

    plt.figure(figsize=(10, 5))
    for n_mels, res in results.items():
        plt.plot(res["history"]["train_loss"], label=f"n_mels={n_mels}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss vs Epoch (varying n_mels)")
    plt.legend()
    plt.grid(True)
    save_fig("nmels_train_loss.png")

    plt.figure(figsize=(8, 5))
    keys = list(results.keys())
    accs = [results[k]["test_acc"] for k in keys]
    plt.bar([str(k) for k in keys], accs, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.xlabel("n_mels")
    plt.ylabel("Test Accuracy")
    plt.title("n_mels vs Test Accuracy")
    plt.ylim(min(accs) - 0.05, 1.0)
    plt.grid(axis="y")
    save_fig("nmels_test_accuracy.png")

    return results


def experiment_groups(train_loader, val_loader, test_loader, device, n_mels=80):
    logger.info("Starting groups experiment")

    groups_list = [1, 2, 4, 8, 16]
    results = {}

    for g in groups_list:
        logger.info(f"groups={g}")
        model = CNN(n_mels=n_mels, groups=g)
        params = count_parameters(model)
        flops = count_flops(model)
        logger.info(f"params: {params}, FLOPs: {flops}")

        history = train_model(model, train_loader, val_loader, NUM_EPOCHS, LR, device)
        test_acc = evaluate(model, test_loader, device)
        logger.info(f"test accuracy: {test_acc:.4f}")

        results[g] = {
            "history": history,
            "test_acc": test_acc,
            "params": params,
            "flops": flops,
            "avg_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        }

    g_keys = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar([str(g) for g in g_keys], [results[g]["avg_epoch_time"] for g in g_keys], color="#4C72B0")
    axes[0].set_xlabel("groups")
    axes[0].set_ylabel("Avg Epoch Time (s)")
    axes[0].set_title("Epoch Time vs Groups")

    axes[1].bar([str(g) for g in g_keys], [results[g]["params"] for g in g_keys], color="#55A868")
    axes[1].set_xlabel("groups")
    axes[1].set_ylabel("Parameters")
    axes[1].set_title("Parameters vs Groups")

    flops_vals = [results[g]["flops"] for g in g_keys]
    axes[2].bar([str(g) for g in g_keys], flops_vals, color="#C44E52")
    axes[2].set_xlabel("groups")
    axes[2].set_ylabel("FLOPs")
    axes[2].set_title("FLOPs vs Groups")

    plt.tight_layout()
    save_fig("groups_metrics.png")

    plt.figure(figsize=(10, 5))
    for g, res in results.items():
        plt.plot(res["history"]["train_loss"], label=f"groups={g}")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss vs Epoch (varying groups)")
    plt.legend()
    plt.grid(True)
    save_fig("groups_train_loss.png")

    plt.figure(figsize=(8, 5))
    accs = [results[g]["test_acc"] for g in g_keys]
    plt.bar([str(g) for g in g_keys], accs, color="#DD8452")
    plt.xlabel("groups")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Groups")
    plt.ylim(min(accs) - 0.05, 1.0)
    plt.grid(axis="y")
    save_fig("groups_test_accuracy.png")

    return results


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    verify_melbanks()

    logger.info("Loading datasets")
    train_loader = DataLoader(AudioDataset(subset="training"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioDataset(subset="validation"), batch_size=BATCH_SIZE)
    test_loader = DataLoader(AudioDataset(subset="testing"), batch_size=BATCH_SIZE)

    model = CNN(n_mels=80, groups=1)
    params = count_parameters(model)
    logger.info(f"Baseline model parameters: {params}")
    assert params <= 100_000, f"Too many parameters: {params}"

    nmels_results = experiment_n_mels(train_loader, val_loader, test_loader, device)

    groups_results = experiment_groups(train_loader, val_loader, test_loader, device)

    logger.info("n_mels summary")
    for n, res in nmels_results.items():
        logger.info(f"n_mels={n:3d}, test acc={res['test_acc']:.4f}")

    logger.info("groups summary")
    for g, res in groups_results.items():
        flops_str = f"{res['flops']:,.0f}"
        logger.info(
            f"groups={g:2d}, "
            f"params: {res['params']}, "
            f"FLOPs: {flops_str}, "
            f"time: {res['avg_epoch_time']:.1f}, "
            f"test acc: {res['test_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
