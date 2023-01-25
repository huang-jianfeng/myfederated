from typing import OrderedDict

import torch

from .fedavg import FedAvgClient
from config.utils import trainable_params


class FedDynClient(FedAvgClient):
    def __init__(self, model, args, logger):
        super().__init__(model, args, logger)

        self.nabla = self.vectorize(self.model).detach().clone().zero_()
        self.vectorized_global_params: torch.Tensor = None
        self.vectorized_curr_params: torch.Tensor = None

    def train(
        self,
        client_id: int,
        new_parameters: OrderedDict[str, torch.nn.Parameter],
        evaluate=False,
        verbose=False,
    ):
        self.vectorized_global_params = self.vectorize(new_parameters).detach().clone()
        res = super().train(client_id, new_parameters, False, evaluate, verbose)
        with torch.no_grad():
            self.nabla = self.nabla - self.args.alpha * (
                self.vectorized_curr_params - self.vectorized_global_params
            )
        return res

    def _train(self):
        self.model.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) > 1:
                    x, y = x.to(self.device), y.to(self.device)
                    logit = self.model(x)
                    loss = self.criterion(logit, y)
                    self.vectorized_curr_params = self.vectorize(self.model)
                    loss -= -torch.dot(self.nabla, self.vectorized_global_params)
                    loss += (self.args.alpha / 2) * torch.norm(
                        self.vectorized_curr_params - self.vectorized_global_params
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

    def vectorize(self, src):
        return torch.cat([param.flatten() for param in trainable_params(src)]).to(
            self.device
        )
