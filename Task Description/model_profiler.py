import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from torch import Tensor

def get_avg_flops(model:nn.Module, input_data:Tensor)->float:
    """
    Estimates the average FLOPs per sample for a model using PyTorch Profiler.
    
    Args:
        model (torch.nn.Module): The neural network model to profile.
        input_data (torch.Tensor): Input tensor for the model (must include batch dimension).
    
    Returns:
        float: Average Mega FLOPs per sample in the batch.
    
    Raises:
        RuntimeError: If no CUDA device is available or input batch size is 0.
    """
    
    # Ensure batch dimension exists
    if input_data.dim() == 0 or input_data.size(0) == 0:
        raise RuntimeError("Input data must have a non-zero batch dimension")
    
    batch_size = input_data.size(0)
    
    # Evaluation mode, improved inference and freeze norm layers
    model = model.eval().cpu()
    input_data = input_data.cpu()
    
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False
        ) as prof:
            model(input_data)
    # Calculate total FLOPs
    total_flops = sum(event.flops for event in prof.events())
    avg_flops = total_flops / batch_size
    
    return avg_flops * 1e-6 / 2
