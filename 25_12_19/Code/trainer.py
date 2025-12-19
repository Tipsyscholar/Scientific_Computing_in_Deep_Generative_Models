from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Optional, List, Type, Tuple, Dict
from path import GaussianConditionalProbabilityPath, ConditionalVectorField

# Try to use notebook version of tqdm if available
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


MiB = 1024 ** 2


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        print(f'Starting training for {num_epochs} epochs...')
        print('-' * 60)
        
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Print periodic updates
            if (epoch + 1) % max(1, num_epochs // 10) == 0 or epoch == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

        # Finish
        print('-' * 60)
        print('✓ Training completed!')
        self.model.eval()


class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)

        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) < self.eta
        y[mask] = 10
        
        # Step 3: Sample t and x
        t = torch.rand(size=(batch_size, 1, 1, 1), device=z.device)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress and output loss
        u1 = self.path.conditional_vector_field(x, z, t)
        u2 = self.model(x, t, y)  # model doesn't know z
        mse = torch.sum((u1 - u2) ** 2) / batch_size
        return mse
