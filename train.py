"""Fine-tune a ResNet18 freshness classifier from a folder-based image dataset."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from freshness.config import CLASS_LABELS, MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"), help="Dataset directory with train/ and val/ folders.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (0 is Windows-friendly).")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/use ImageNet pretrained weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir, val_dir = args.data / "train", args.data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit(f"Expected dataset folders: {train_dir} and {val_dir}")

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)
    expected = set(CLASS_LABELS)
    if set(train_data.classes) != expected or set(val_data.classes) != expected:
        raise SystemExit("Both train/ and val/ must contain exactly: aging/, fresh/, spoiled/.")
    if train_data.class_to_idx != val_data.class_to_idx:
        raise SystemExit("Train and validation folders must use the same class folders.")
    if min(len(train_data), len(val_data)) == 0:
        raise SystemExit("Train and validation datasets must both contain images.")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = None if args.no_pretrained else models.ResNet18_Weights.DEFAULT
    network = models.resnet18(weights=weights)

    # Transfer learning: train the final residual block and classification head.
    for parameter in network.parameters():
        parameter.requires_grad = False
    for parameter in network.layer4.parameters():
        parameter.requires_grad = True
    network.fc = nn.Linear(network.fc.in_features, len(train_data.classes))
    network.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW((p for p in network.parameters() if p.requires_grad), lr=args.learning_rate)
    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        network.train()
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = network(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        network.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (network(images).argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train accuracy {train_accuracy:.2%} | validation accuracy {val_accuracy:.2%}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "state_dict": network.cpu().state_dict(),
                    "classes": train_data.classes,
                    "val_accuracy": val_accuracy,
                },
                args.output,
            )
            network.to(device)

    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Saved model checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
