import numpy as np
import tqdm
from collections import OrderedDict
# from matplotlib import pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn as nn
import torchvision
import torch
from torchvision import transforms
import csv

EPOCHS = 200
LEARNING_RATE = 0.01
STEP_LR = 5
LR_SCALER = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist(batch_size=64, new_transforms=[]):
    all_transforms = [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    new_transforms.extend(all_transforms)
    transform = transforms.Compose(all_transforms)

    # create train and test sets
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders for the MNIST dataset
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar10(batch_size=64, new_transforms=[]):
    all_transforms = [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    all_transforms.extend(new_transforms)
    transform = transforms.Compose(all_transforms)

    # create train and test sets
    cifar10_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders for the MNIST dataset
    train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar100(batch_size=64, new_transforms=[]):
    all_transforms = [
        # transforms.Resize(224),  # ResNet18 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[2,2,2]),
    ]

    new_transforms.extend(all_transforms)
    transform = transforms.Compose(all_transforms)

    # create train and test sets
    cifar_100_train = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    cifar_100_test = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders for the MNIST dataset
    train_loader = DataLoader(cifar_100_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(cifar_100_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    parellel=True,
    path=None,
):
    # Train the model
    learning_rate = LEARNING_RATE  # LEARNING_RATE
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if parellel:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    else:
        device = torch.device("cpu")

    model.to(device)

    train_losses = []
    test_losses = []
    prev_acc = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=LR_SCALER, patience=STEP_LR
    )

    for epoch in range(epochs):
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Epoch: {} Loss: {:.6f}\r".format(epoch + 1, loss.item()), end="")

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader.dataset)
        acc, test_loss = test(model, test_loader, parellel=parellel)
        train_losses.append(epoch_loss)
        test_losses.append(test_loss)
        print(
            "Epoch: {} Loss: {:.6f} Test Loss: {:.6f} Test Accuracy: {:.2f}%".format(
                epoch + 1, epoch_loss, test_loss, acc
            )
        )

        scheduler.step(test_loss)

        if path is not None and acc - prev_acc > 1:
            print(
                f"Accuracy improved from {prev_acc:.2f}% to {acc:.2f}%. Saving model."
            )
            save_model(model, path)

        prev_acc = acc

    return model


def test(model, test_loader, parellel=False):
    # Test the model
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            if parellel:
                data = data.to(device)
                target = target.to(device)
            else:
                data = data.to("cpu")
                target = target.to("cpu")

            output = model(data)

            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total

    torch.cuda.empty_cache()

    return accuracy, test_loss


def test_and_dump(model, test_loader, parellel=False, name="unnamed_test"):
    # Test the model
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    logits_all = []

    with torch.no_grad():
        for data, target in tqdm.tqdm(test_loader):
            if parellel:
                data = data.to(device)
                target = target.to(device)
            else:
                data = data.to("cpu")
                target = target.to("cpu")

            output, logits = model(data)
            logits_all.extend(logits)

            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Open a CSV file to write the logits
    with open(f"{name}.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["logits"])
        for logit in logits_all:
            writer.writerow(logit.cpu().numpy().tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total

    torch.cuda.empty_cache()

    return accuracy, test_loss


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device=device, parellel=True):
    if parellel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def augment_img(img: torch.Tensor, noise: float):
    img += np.random.randn(*img.shape) * noise * 2 - noise
    return np.array(np.clip(img, 0, 1), dtype=np.float32)


def augment_set(x, y, length, noise=0.002):
    x_aug, y_aug = [], []

    for i in range(len(x)):
        for j in range(length):
            x_aug.append(augment_img(x[i], noise))
            y_aug.append(y[i])

    return np.array(x_aug), np.array(y_aug)


def remove_data_parallel(model):
    old_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    for k, v in old_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    return model


def loader_to_xy(loader):
    x, y = [], []
    for data, target in loader:
        x.extend(data)
        y.extend(target)
    return torch.stack(x), torch.stack(y)


def xy_to_loader(x, y, batch_size=64):
    dataset = torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
