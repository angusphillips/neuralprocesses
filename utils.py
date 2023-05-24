from utils.wrapper import AbstractModelWrapper
import neuralprocesses.torch as nps
import torch
import lab as B

class ModelWrapper(AbstractModelWrapper):
    def __init__(
            self,
            model
    ):
        self.model = model

    def sample(self, rng, shape, context):
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
        
        with torch.no_grad():
            _,_,samples, _ = nps.predict(self.model, batch['contexts'], batch['xt'], num_samples=1)
        return samples.cpu().numpy()[:,0,:,:]