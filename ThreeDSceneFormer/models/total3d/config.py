from .training import Trainer
from .dataloader import Total3D_dataloader

def get_trainer(cfg, net, optimizer, device=None):
    return Trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)

def get_tester(cfg, net, device=None):
    # Dataset label mappings are only needed for evaluation, not API inference.
    from .testing import Tester
    return Tester(cfg=cfg, net=net, device=device)

def get_dataloader(config, mode):
    return Total3D_dataloader(config=config, mode=mode)
