import os
from argparse import ArgumentParser, Namespace

import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import v2

transforms_train = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation((-30, 30)),
        v2.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

transforms_test = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Paths
    parser.add_argument(
        "--data_dir",
        type=str,
        # ../data
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data")),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        # ../models
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "models")),
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        # ../logs
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "logs")),
    )

    # Model parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    return parser.parse_args()


def main(args: Namespace) -> None:
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Load dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, transform=transforms_train, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, transform=transforms_test, download=True
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Create vgg19 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(train_dataset.classes)
    model = torchvision.models.vgg19_bn(num_classes=num_classes).to(device)

    # Create loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5, verbose=True
    )
    # optimizer = torch.optim.Adam(
    #     params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )

    # Start training
    best_val_acc = None
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for epoch in range(args.num_epochs):
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0

        # Train
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_train_acc += (outputs.argmax(dim=1) == labels).sum().item()
            epoch_train_loss += loss.item()

        epoch_train_acc /= len(train_dataset)
        epoch_train_loss /= len(train_loader)

        train_acc.append(epoch_train_acc * 100)
        train_loss.append(epoch_train_loss)

        # Validate
        with torch.no_grad():
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0

            model.eval()
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                epoch_val_acc += (outputs.argmax(dim=1) == labels).sum().item()
                epoch_val_loss += loss.item()

            epoch_val_acc /= len(test_dataset)
            epoch_val_loss /= len(test_loader)

            # Decrease learning rate by a factor of 10 if validation accuracy plateaus
            scheduler.step(epoch_val_acc)

            val_acc.append(epoch_val_acc * 100)
            val_loss.append(epoch_val_loss)

            # Save model
            if best_val_acc is None or epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save(
                    model.state_dict(), os.path.join(args.save_dir, "vgg19_bn.pth")
                )
                print(f"Saved model with validation accuracy: {best_val_acc:.4f}")

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f}"
        )

    # Save loss and accuracy plots
    _, (ax1, ax2) = plt.subplots(2, 1)
    epochs = range(1, args.num_epochs + 1)

    # Plot loss
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, train_acc, label="train_acc")
    ax2.plot(epochs, val_acc, label="val_acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy(%)")
    ax2.set_ylim(0, 100)
    ax2.set_yticks(range(0, 101, 10))
    ax2.set_title("Accuracy")
    ax2.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, "loss_and_acc.png"))
    plt.close()


if __name__ == "__main__":
    main(parse_args())
