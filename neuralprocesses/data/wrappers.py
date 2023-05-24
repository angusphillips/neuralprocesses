import lab as B
import torch
from neuralprocesses.data.data import DataGenerator

__all__ = ["CSVGenerator"]

class CSVGenerator(DataGenerator):
    def __init__(
        self,
        dataloader,
        dtype,
        seed,
        batch_size,
        device='cpu'
    ): 
        self.dataloader = dataloader
        num_tasks = self.data.shape[0]
        super().__init__(dtype, seed, num_tasks, batch_size, device)

    def __len__(self):
        return len(self.dataloader)
    
    def generate_batch(self):

        _, context = next(self.dataloader)

        xc = context[4]
        yc = context[5]
        x = context[2]
        y = context[3]

        x = B.to_active_device(torch.tensor(x, dtype=self.dtype))
        y = B.to_active_device(torch.tensor(y, dtype=self.dtype))
        xc = B.to_active_device(torch.tensor(xc, dtype=self.dtype))
        yc = B.to_active_device(torch.tensor(yc, dtype=self.dtype))

        batch={}
        batch["contexts"] = [(xc,yc)]
        batch["xt"] = x
        batch["yt"] = y
        
        return batch
    

class GPPosteriorGenerator(DataGenerator):
    def __init__(
        self,
        dataloader,
        dtype,
        seed,
        batch_size,
        num_batches,
        loglik_eval,
        device='cpu'
    ): 
        self.dataloader = dataloader
        num_tasks = num_batches*batch_size
        super().__init__(dtype, seed, num_tasks, batch_size, device)
        self.loglik_eval = loglik_eval
        
    def __len__(self):
        return len(self.dataloader)
    
    def generate_batch(self):

        _, context = next(self.dataloader)

        xc = context[4]
        yc = context[5]
        x = context[2]
        y = context[3]

        x = B.to_active_device(torch.tensor(x, dtype=self.dtype))
        y = B.to_active_device(torch.tensor(y, dtype=self.dtype))
        xc = B.to_active_device(torch.tensor(xc, dtype=self.dtype))
        yc = B.to_active_device(torch.tensor(yc, dtype=self.dtype))

        batch={}
        batch["contexts"] = [(xc,yc)]
        batch["xt"] = x
        batch["yt"] = y
        
        return batch
