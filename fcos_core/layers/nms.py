# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
# from fcos_core import _C
from torchvision import ops
# nms = _C.nms
nms=ops.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
