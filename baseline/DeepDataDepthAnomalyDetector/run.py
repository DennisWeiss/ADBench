import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader
from baseline.DeepDataDepthAnomalyDetector.models.DeepDataDepthEncoder import DeepDataDepthEncoder
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DeepDataDepthAnomalyDetector:
    Method = Enum('Method', [
        'VarianceMaximization',
        'KlDivFit'
    ])

    def __init__(self,
                 seed: int = 0,
                 model_name: str = 'DeepDataDepthAnomalyDetector',
                 method: Method = Method.KlDivFit,
                 mem_in_mb: int = 8000,
                 encoder_lr: float = 1e-4,
                 halfspace_optim_lr: float = 1e+3,
                 weight_decay: float = 0,
                 temp: float = 1e-1,
                 representation_dim: int = 32,
                 target_dist: Callable[[float], float] = lambda x: 8 * x,
                 iter: int = 10,
                 data_depth_iter: int = 10,
                 data_depth_computations: int = 10
                 ):
        """
        :param method (DeepDataDepthAnomalyDetector.Method, optional): Method to use for optimizing the data depth distribution (default: ``DeepDataDepthAnomalyDetector.Method.KLDivFit``)
        :param mem_in_mb (int, optional): (GPU) Memory available in MB (default: ``8000``)
        :param encoder_lr (float, optional): Learning rate for encoder (default: ``1e-3``)
        :param halfspace_optim_lr (float, optional): Learning rate of the halfspace optimizer (default: ``1e+3``)
        :param weight_decay (float, optional): Weight decay (default: ``0``)
        :param temp (float, optional): Temperature value (default: ``1e-1``)
        :param representation_dim (int, optional): Representation dimension (default: ``32``)
        :param target_dist (Callable[[float], float], optional): Target distribution. Used only for KL divergence fit (default: ``f(x) = 8x``)
        :param iter (int, optional): Number of iterations of training the encoder (default: ``40``)
        :param data_depth_iter (int, optional): Number of iterations of the halfspace optimizer (default: ``20``)
        :param data_depth_computations (int, optional): Number of re-computations of the data depth value (default: ``20``)
        """
        self.X_train = None
        self.method = method
        self.mem_in_mb = mem_in_mb
        self.encoder_lr = encoder_lr
        self.halfspace_optim_lr = halfspace_optim_lr
        self.weight_decay = weight_decay
        self.temp = temp
        self.representation_dim = representation_dim
        self.target_dist = target_dist
        self.iter = iter
        self.data_depth_iter = data_depth_iter
        self.data_depth_computations = data_depth_computations

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.log(logging.INFO, f'Using device {self.device}')
        if torch.cuda.is_available():
            logging.log(logging.INFO, f'The following GPU will be used: {torch.cuda.get_device_name()}')

    def soft_tukey_depth(self, X_: torch.Tensor, X: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        X_new = X.repeat(X_.size(dim=0), 1, 1)
        X_new_tr = X_.repeat(X.size(dim=0), 1, 1).transpose(0, 1)
        X_diff = X_new - X_new_tr
        dot_products = X_diff.mul(Z.repeat(X.size(dim=0), 1, 1).transpose(0, 1)).sum(dim=2)
        dot_products_normalized = dot_products.transpose(0, 1).divide(self.temp * Z.norm(dim=1))
        return torch.sigmoid(dot_products_normalized).sum(dim=0).divide(X.size(dim=0))

    def get_kl_divergence(self, soft_tukey_depths: torch.Tensor, f: Callable[[float], float], kernel_bandwidth: float = 0.05,
                          epsilon: float = 1e-3) -> torch.Tensor:
        delta = 0.005
        kl_divergence = torch.tensor(0)
        for x in torch.arange(0, 0.5, delta):
            val = torch.exp(torch.square(soft_tukey_depths - x).divide(
                torch.tensor(-2 * kernel_bandwidth * kernel_bandwidth))).mean()
            f_val = f(x)
            kl_divergence = kl_divergence.subtract(f_val * delta * torch.log(val.divide(f_val + epsilon)))
        return kl_divergence

    def draw_histogram(self, X: torch.Tensor, X_: torch.Tensor, z: torch.Tensor, bins=20):
        soft_tukey_depths = self.soft_tukey_depth(X, X_, z)
        tukey_depth_histogram = plt.figure()
        plt.hist(soft_tukey_depths.detach().cpu().numpy(), bins=bins)
        tukey_depth_histogram.show()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        :param X_train: Training data
        :param y_train: Training labels
        :return: self
        """
        X_train = X_train[y_train == 0]

        # self.representation_dim = X_train.shape[1]
        self.representation_dim = min(self.representation_dim, X_train.shape[1])

        logging.log(logging.INFO, f'Training set size is {X_train.shape[0]}.')

        self.X_train = X_train

        batch_size = min(int((15_000 * self.mem_in_mb / X_train.shape[0]) // self.representation_dim), X_train.shape[0])
        if batch_size < X_train.shape[0]:
            batches = (X_train.shape[0] // batch_size) + 1
            batch_size = int(math.ceil((X_train.shape[0] / batches)))

        logging.log(logging.INFO, f'Batch size is {batch_size}.')

        train_tensor = TensorDataset(torch.from_numpy(X_train).float())
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)
        train_loader_full = DataLoader(train_tensor, batch_size=X_train.shape[0], shuffle=False)

        n = X_train.shape[0]
        input_dim = X_train.shape[1]

        hidden_layer_dim = min(max(input_dim, self.representation_dim), 2 * self.representation_dim)

        self.encoder = DeepDataDepthEncoder(input_dim, hidden_layer_dim, self.representation_dim).to(self.device)
        self.encoder.train()

        optimizer_encoder = Adam(self.encoder.parameters(), lr=self.encoder_lr, weight_decay=self.weight_decay)

        best_z = 2 * torch.rand(n, self.representation_dim).to(self.device) - 1

        i = 0

        # for epoch in range(int(math.ceil(self.iter / int(math.ceil(X_train.shape[0] / batch_size))))):
        for epoch in range(self.iter):
            for step_type in ['optimize_halfspace', 'optimize_encoder']:
                for step, (x_train,) in enumerate(train_loader):
                    if step_type == 'optimize_encoder' and (i + 1) % 1 == 0:
                        logging.log(logging.INFO, f'Iteration {int(i + 1)}/{self.iter}')

                    x = x_train.to(self.device)
                    y = self.encoder(x)
                    y_detached = y.detach()

                    for step2, (x_full,) in enumerate(train_loader_full):
                        x_full = x_full.to(self.device)
                        y_full = self.encoder(x_full)
                        y_full_detached = y_full.detach()

                        if step_type == 'optimize_halfspace':
                            tukey_depths = self.soft_tukey_depth(
                                y_detached,
                                y_full_detached,
                                best_z[(step * batch_size):((step + 1) * batch_size)]
                            )

                            for k in range(self.data_depth_computations):
                                z = nn.Parameter(2 * torch.rand(y_detached.size(dim=0), self.representation_dim).to(self.device) - 1)
                                optimizer_z = SGD([z], lr=self.halfspace_optim_lr)
                                for l in range(self.data_depth_iter):
                                    optimizer_z.zero_grad()
                                    current_tukey_depths = self.soft_tukey_depth(y_detached, y_full_detached, z)
                                    current_tukey_depths.sum().backward()
                                    optimizer_z.step()

                                current_tukey_depths = self.soft_tukey_depth(y_detached, y_full_detached, z)

                                for l in range(tukey_depths.size(dim=0)):
                                    if current_tukey_depths[l] < tukey_depths[l]:
                                        tukey_depths[l] = current_tukey_depths[l].detach()
                                        best_z[(step * batch_size) + l] = z[l].detach()
                        elif step_type == 'optimize_encoder':
                            tukey_depths = self.soft_tukey_depth(y_full, y, best_z)

                            optimizer_encoder.zero_grad()
                            if self.method == DeepDataDepthAnomalyDetector.Method.VarianceMaximization:
                                loss = torch.var(tukey_depths)
                            elif self.method == DeepDataDepthAnomalyDetector.Method.KlDivFit:
                                loss = self.get_kl_divergence(tukey_depths, self.target_dist)
                            logging.log(logging.INFO, f'Loss: {loss.item()}')
                            logging.log(logging.INFO, f'Mean: {torch.mean(tukey_depths).item()}')
                            loss.backward()
                            optimizer_encoder.step()
                            if int(i + 1) % 1 == -1:
                                self.draw_histogram(
                                    y_detached,
                                    y_full_detached,
                                    best_z[(step * batch_size):((step + 1) * batch_size)].detach()
                                )
                    if step_type == 'optimize_encoder':
                        i += 1

        return self

    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Data
        :return: Anomaly scores
        """
        if self.X_train is None:
            raise Exception('Model not fitted yet')

        batch_size = min(int((2_000 * self.mem_in_mb / x.shape[0]) // self.representation_dim), x.shape[0])

        self.encoder.eval()

        train_data = TensorDataset(torch.from_numpy(self.X_train).float())
        train_loader = DataLoader(train_data, batch_size=self.X_train.shape[0], shuffle=False)

        test_data = TensorDataset(torch.from_numpy(x).float())
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

        scores = np.zeros(0)

        for step, (x_train,) in enumerate(train_loader):
            x_train = x_train.to(self.device)
            y_train = self.encoder(x_train)
            y_train_detached = y_train.detach()

            for step2, (x_test,) in enumerate(test_loader):
                x_test = x_test.to(self.device)
                y_test = self.encoder(x_test)
                y_test_detached = y_test.detach()

                best_z = 2 * torch.rand(x_test.shape[0], self.representation_dim).to(self.device) - 1
                tukey_depths = self.soft_tukey_depth(y_test_detached, y_train_detached, best_z)

                for i in range(2 * self.data_depth_computations):
                    z = nn.Parameter(2 * torch.rand(x_test.shape[0], self.representation_dim).to(self.device) - 1)
                    optimizer_z = SGD([z], lr=self.halfspace_optim_lr)
                    for j in range(2 * self.data_depth_iter):
                        optimizer_z.zero_grad()
                        current_tukey_depths = self.soft_tukey_depth(y_test_detached, y_train_detached, z)
                        current_tukey_depths.sum().backward()
                        optimizer_z.step()

                    current_tukey_depths = self.soft_tukey_depth(y_test_detached, y_train_detached, z)

                    for j in range(current_tukey_depths.size(dim=0)):
                        if current_tukey_depths[j] < tukey_depths[j]:
                            tukey_depths[j] = current_tukey_depths[j].detach()
                            best_z[j] = z[j].detach()

                scores = np.concatenate((scores, (0.5 - tukey_depths).cpu().numpy()))

        return scores
