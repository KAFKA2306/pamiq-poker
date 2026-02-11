"""This file contains custom configuration resolvers for OmegaConf.
See: https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html
"""
import torch
from omegaconf import OmegaConf
_registered = False
def register_custom_resolvers() -> None:
    global _registered
    if not _registered:
        OmegaConf.register_new_resolver(
            "python.eval", eval
        )
        OmegaConf.register_new_resolver(
            "torch.device", torch.device
        )
        OmegaConf.register_new_resolver(
            "torch.dtype", convert_dtype_str_to_torch_dtype
        )
        _registered = True
def convert_dtype_str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert the string of dtype such as "float32" to `torch.dtype` object
    `torch.float32` .
    All dtypes are listed in
    https://pytorch.org/docs/stable/tensor_attributes.html
    """
    if hasattr(torch, dtype_str):
        dtype = getattr(torch, dtype_str)
    else:
        raise ValueError(f"Dtype name {dtype_str} does not exists!")
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Dtype name {dtype_str} is not dtype! Object: {dtype}")
    return dtype
