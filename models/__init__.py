"""Model definitions for Silent Talk."""

from .ctc_decoder import GreedyCTCDecoder
from .lipnet import LipNet, LipNetConfig

__all__ = ["LipNet", "LipNetConfig", "GreedyCTCDecoder"]
