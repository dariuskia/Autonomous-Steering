import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.utils.data import DataLoader

import einops
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from dataclasses import dataclass
from tqdm import tqdm

device = ('cuda' if torch.cuda.is_available else 'cpu')

class DrivingImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.tensor(self.img_labels.iloc[idx, -1], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class SteeringModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), # 3 channels for RGB
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        conv = self.conv_layers(x)
        flattened = self.flatten(conv)
        out = self.linear_layers(flattened)
        return out

def get_steering_data():
    SIZE = (66, 200)
    img_transforms = v2.Compose([
        v2.Resize(SIZE, antialias=True),
        v2.ToDtype(torch.float32)
    ])
    # label_transform = lambda x: x.to(dtype=torch.float32)
    steering_trainset = DrivingImageDataset("Data/labelsB_train.csv", "Data/trainB", transform = img_transforms)
    steering_testset = DrivingImageDataset("Data/labelsB_val.csv", "Data/valB", transform = img_transforms)
    return steering_trainset, steering_testset

@dataclass
class SteeringTrainingArgs():
    batch_size: int = 256
    num_epochs: int = 1000
    learning_rate: float = 0.001
    optimizer = torch.optim.Adam
    loss_fn = nn.MSELoss()
    logger: str = "results/results.log"

class SteeringTrainer():
    def __init__(self, args: SteeringTrainingArgs):
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.loss_fn = args.loss_fn
        self.model = SteeringModel().to(device)
        self.optimizer = args.optimizer(self.model.parameters(), lr=args.learning_rate)
        self.steering_trainset, self.steering_testset = get_steering_data()
        self.steering_trainloader = DataLoader(self.steering_trainset, batch_size=args.batch_size, shuffle=True)
        self.steering_testloader = DataLoader(self.steering_testset, batch_size=args.batch_size, shuffle=False)
        # self.loss_list = []
        # self.accuracy_list = []
        self.epsilons = [0.1, 0.2, 0.5, 1, 2, 5]
        self.num_train_batches = int(len(self.steering_trainset) / self.batch_size)
        self.num_test_batches = int(len(self.steering_testset) / self.batch_size)
        self.logger = args.logger

    def train(self):
        for i in range(self.num_epochs):
            self.model.train()
            train_epoch_loss = 0
            train_acc = [0 for _ in range(len(self.epsilons))]
            test_acc = [0 for _ in range(len(self.epsilons))]
            for imgs, labels in tqdm(self.steering_trainloader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                Y = self.model(imgs).squeeze()
                loss = self.loss_fn(Y, labels)
                loss.backward()
                train_epoch_loss += loss.item()
                for idx, eps in enumerate(self.epsilons):
                    train_acc[idx] += ((Y - labels).abs() < eps).sum()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.loss_list.append(loss.item())
            self.model.eval()
            test_epoch_loss = 0
            for imgs, labels in tqdm(self.steering_testloader):
                imgs = imgs.to(device)
                labels = labels.to(device)
                with torch.inference_mode():
                    Y = self.model(imgs).squeeze()
                    test_epoch_loss += self.loss_fn(Y, labels).sum()
                    for idx, eps in enumerate(self.epsilons):
                        test_acc[idx] += ((Y - labels).abs() < eps).sum()
            train_acc = [acc / len(self.steering_trainset) for acc in train_acc]
            test_acc = [acc / len(self.steering_testset) for acc in test_acc]
            avg_eval_loss = test_epoch_loss / len(self.steering_testset)
            # self.accuracy_list.append(avg_eval_loss)
            # print(f"Evaluation loss at epoch {i}: {avg_eval_loss}")
            info = {
                "epoch": f"[{i}/{self.num_epochs}]",
                "training loss": train_epoch_loss/self.num_train_batches,
                "train accuracy": torch.mean(torch.stack(train_acc), dtype=torch.float32).item(),
                "test loss": test_epoch_loss/self.num_test_batches,
                "test accuracy": torch.mean(torch.stack(test_acc), dtype=torch.float32).item(),
            }
            with open(self.logger, "a") as f:
                f.write(", ".join([f"{k}: {v}" for k, v in info.items()]))
                f.write("\n")
            print(", ".join([f"{k}: {v}" for k, v in info.items()]))
        path = "./results/model.pth"
        print(f"Saving model to {path}.")
        torch.save(self.model.state_dict(), "./results/model.pth")
    
# args = SteeringTrainingArgs()
# trainer = SteeringTrainer(args)
# trainer.train()

# model = SteeringModel().to(device)
model = net_nvidia_pytorch().to(device)
model.load_state_dict(torch.load("../Multi_Perturbation_Robustness/results/train_results/trainB_/model-final_0.pth"))
# model.load_state_dict(torch.load("./results/model.pth"))
steering_trainset, steering_testset = get_steering_data()
# print(model(einops.rearrange(steering_testset[0][0].to(device), "c h w -> 1 c h w")), steering_testset[0][1])
steering_testloader  = DataLoader(steering_testset, batch_size=256, shuffle=False)
total_loss = torch.zeros(1).to(device)
eps = 0.5
acc = 0
for imgs, labels in steering_testloader:
    imgs = imgs.to(device)
    labels = labels.to(device)
    # out = model(imgs).squeeze()
    out = model(imgs)[0].squeeze()
    # print(pred)
    acc += ((out - labels).abs() < eps).sum()
    # total_loss += F.mse_loss(out, labels).sum()

# print(f"MSE: {total_loss.item()}, {total_loss.item()/len(steering_testset)}")
print(f"Accuracy: {acc / len(steering_testset)}")