import torch


def get_torch_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        n_gpu = torch.cuda.device_count()
        print(f"Found device: {device_name}, n_gpu: {n_gpu}")
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')
    return device


device = get_torch_device()
