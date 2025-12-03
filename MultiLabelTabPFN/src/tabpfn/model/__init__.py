"""Contains references to the base architecture, for backwards compatability.

DEPRECATED: import MultiLabelTabPFN.src.tabpfn.architectures.base instead

Previously MultiLabelTabPFN.src.tabpfn only supported a single architecture, which was in this MultiLabelTabPFN.src.tabpfn.model
module. Now we support multiple architectures, stored in MultiLabelTabPFN.src.tabpfn.architectures, and
tabpfn.model has moved to MultiLabelTabPFN.src.tabpfn.architectures.base .
"""

import warnings

from MultiLabelTabPFN.src.tabpfn import model_loading as loading
from MultiLabelTabPFN.src.tabpfn.architectures.base import (
    attention,
    bar_distribution,
    config,
    encoders,
    layer,
    memory,
    mlp,
    transformer,
)

__all__ = [
    "attention",
    "bar_distribution",
    "config",
    "encoders",
    "layer",
    "loading",
    "memory",
    "mlp",
    "transformer",
]

warnings.warn(
    "tabpfn.model has moved to MultiLabelTabPFN.src.tabpfn.architectures.base. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)
