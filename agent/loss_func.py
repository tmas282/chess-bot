import torch


class A2CombinedLoss(torch.nn.Module):
    def __init__(self, entropy_coeff, beta_critic_weight):
        super().__init__()
        self.entropy_coeff = entropy_coeff
        self.beta_critic_weight = beta_critic_weight
    def forward(self, prob_log, Qvals, values, entropy_term=0):
        advantage = Qvals - values

        critic_loss = torch.mean(torch.pow(advantage, 2))
        
        mean_entropy = entropy_term / len(prob_log)
        actor_loss = -prob_log * advantage.detach() + self.entropy_coeff * mean_entropy
        actor_loss = torch.mean(actor_loss)

        return actor_loss + self.beta_critic_weight * critic_loss