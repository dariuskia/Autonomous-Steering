import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from ax.service.ax_client import AxClient, ObjectiveProperties

from train import SteeringTrainer, SteeringTrainingArgs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", help="the name of the experiment")
    args = parser.parse_args()
    return args

class MSEWithTau(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
    
    def forward(self, output, target):
        mse = F.mse_loss(output, target, reduction='none')
        loss = torch.mean(mse * torch.exp(self.tau * output))
        return loss


if __name__ == "__main__":
    args = parse_args()

    def bo_trial(params):
        training_args = SteeringTrainingArgs(
            batch_size = 256, 
            num_epochs = 50, 
            learning_rate = 0.001, 
            optimizer = torch.optim.Adam, 
            loss_fn = MSEWithTau(params['tau']),
            results_dir = "results/",
            exp_name = f"optimize_{args.exp_name}",
            epsilons = [0.1, 0.2, 0.5, 1, 2, 5]
        )
        trainer = SteeringTrainer(training_args)
        trainer.train()
        return trainer.evaluate()

    ax_client = AxClient()
    ax_client.create_experiment(
        name="dynamic_tau",
        parameters=[
            {
                "name": "tau",
                "type": "range",
                "bounds": [0., 2.],
                "value_type": "float",
            }
        ],
        objectives={"accuracy": ObjectiveProperties(minimize=False)}
    )
    ax_client.attach_trial(
        parameters={"tau": 0.}
    )
    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=bo_trial(baseline_parameters))

    for i in range(3):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=bo_trial(parameters))

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters, values)
    with open(f"results/bo_{args.exp_name}.log", "a") as f:
        f.write(best_parameters, values)
    ax_client.save_to_json_file(f"results/ax_client_{args.exp_name}.json")

# TODO: model results should be optimizer_1epoch_0.log ... optimize_1epoch_24.log for each trial
# how do i add parameters without optimizing them?
# TODO: somehow getting decent accuracy with tao ~ 30
# perhaps limit it within [0., 2.]