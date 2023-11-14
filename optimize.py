import torch
import torch.nn as nn
import torch.nn.functional as F

from ax.service.ax_client import AxClient, ObjectiveProperties

from train import SteeringTrainer, SteeringTrainingArgs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MSEWithTau(nn.Module):
    def __init__(self, tau):
        self.tau = tau
    
    def forward(self, output, target):
        mse = F.mse_loss(output, target, reduction='none')
        loss = torch.mean(mse * torch.exp(self.tau * output))
        return loss

def bo_trial(tau):
    training_args = SteeringTrainingArgs(
        batch_size = 256, 
        num_epochs = 1, 
        learning_rate = 0.001, 
        optimizer = torch.optim.Adam, 
        loss_fn = MSEWithTau(tau),
        results_dir = "results/",
        exp_name = "bo_1",
        epsilons = [0.1, 0.2, 0.5, 1, 2, 5]
    )
    trainer = SteeringTrainer(training_args)
    trainer.train()
    return trainer.evaluate()

if __name__ == "__main__":
    ax_client = AxClient()

    ax_client.create_experiment(
        name="dynamic_tau",
        parameters=[
            {
                "name": "tau",
                "type": "range",
                "bounds": [0., 1e2],
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

    for i in range(25):
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=bo_trial(parameters))

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters, values)
    with open("results/bo_1.log", "a") as f:
        f.write(best_parameters, values)
