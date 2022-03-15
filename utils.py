import torch
def list_to_device(lis, device : str): return [x.to(device) for x in lis]