import torch
def list_to_device(lis, device : str): return [x.to(device) for x in lis]


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self): self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_acc(logits, labels):
    return sum(logits.argmax(dim=1) == labels) / logits.shape[0]