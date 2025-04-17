import torch
import copy

class EMA:
    def __init__(self,model,decay=0.9999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self,model):
        model_state = model.state_dict()
        ema_state = self.ema_model.state_dict()
        
        for key in model_state.keys():
            if model_state[key].dtype in (torch.float32,torch.float16):
                ema_state[key].mul_(self.decay).add_(model_state[key], alpha=1 - self.decay)

    def get_model(self):
        return self.ema_model