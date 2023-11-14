import os
import argparse
from dataclasses import dataclass
from typing import List, Type, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from datatypes import get_steering_data
from model import SteeringModel

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip("py"),
        help="the name of this experiment")
    args = parser.parse_args()
    return args

@dataclass
class SteeringTrainingArgs():
    batch_size: int
    num_epochs: int
    learning_rate: float
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    results_dir: str
    exp_name: str
    epsilons: List[float]
    with_tau: bool

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
        self.epsilons = args.epsilons
        self.num_train_batches = int(len(self.steering_trainset) / self.batch_size)
        self.num_test_batches = int(len(self.steering_testset) / self.batch_size)
        self.results_dir = args.results_dir
        self.exp_name = args.exp_name

    def train(self):
        with open(os.path.join(self.results_dir, f"results_{self.exp_name}.log"), "a") as f:
            f.write("epoch,training loss,train accuracy,test loss,test accuracy\n")
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
                "epoch": i,
                "training loss": train_epoch_loss/self.num_train_batches,
                "train accuracy": torch.mean(torch.stack(train_acc), dtype=torch.float32).item(),
                "test loss": (test_epoch_loss/self.num_test_batches).item(),
                "test accuracy": torch.mean(torch.stack(test_acc), dtype=torch.float32).item(),
            }
            with open(os.path.join(self.results_dir, f"results_{self.exp_name}.log"), "a") as f:
                f.write(",".join(map(str, info.values())))
                f.write("\n")
            print(", ".join([f"{k}: {v}" for k, v in info.items()]))
        path = os.path.join(self.results_dir, f"model_{self.exp_name}.pth")
        print(f"Saving model to {path}.")
        torch.save(self.model.state_dict(), path)
    
    def evaluate(self):
        test_epoch_loss = 0
        for imgs, labels in tqdm(self.steering_testloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            with torch.inference_mode():
                Y = self.model(imgs).squeeze()
                test_epoch_loss += self.loss_fn(Y, labels).sum()
        test_acc = [acc / len(self.steering_testset) for acc in test_acc]
        avg_test_acc = torch.mean(torch.stack(test_acc), dtype=torch.float32).item()
        # test_loss = (test_epoch_loss/self.num_test_batches).item()
        return avg_test_acc

    
# args = SteeringTrainingArgs()
# trainer = SteeringTrainer(args)
# trainer.train()

if __name__ == "__main__":
    command_args = parse_args()

    training_args = SteeringTrainingArgs(
        batch_size = 256, 
        num_epochs = 1000, 
        learning_rate = 0.001, 
        optimizer = torch.optim.Adam, 
        loss_fn = nn.MSELoss(), 
        results_dir = "results/",
        exp_name = command_args.exp_name,
        epsilons = [0.1, 0.2, 0.5, 1, 2, 5]
    )

    trainer = SteeringTrainer(training_args)
    trainer.train()

    # model = SteeringModel().to(device)
    # # model.load_state_dict(torch.load("../Multi_Perturbation_Robustness/results/train_results/trainB_/model-final_0.pth"))
    # model.load_state_dict(torch.load("./results/model.pth"))
    # steering_trainset, steering_testset = get_steering_data()
    # # print(model(einops.rearrange(steering_testset[0][0].to(device), "c h w -> 1 c h w")), steering_testset[0][1])
    # steering_testloader  = DataLoader(steering_testset, batch_size=256, shuffle=False)
    # total_loss = torch.zeros(1).to(device)
    # eps = 0.5
    # acc = 0
    
    # for imgs, labels in steering_testloader:
    #     imgs = imgs.to(device)
    #     labels = labels.to(device)
    #     # out = model(imgs).squeeze()
    #     out = model(imgs)[0].squeeze()
    #     # print(pred)
    #     acc += ((out - labels).abs() < eps).sum()
    #     # total_loss += F.mse_loss(out, labels).sum()

    # print(f"MSE: {total_loss.item()}, {total_loss.item()/len(steering_testset)}")
    # print(f"Accuracy: {acc / len(steering_testset)}")