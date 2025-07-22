"""neurop layers module."""
from .spectral_convolution import SpectralConv as SpectralConv
from .spectral_convolution import NormType as NormType

from .skip_connections import create_skip_connection as create_skip_connection

from .fno_unit import FNOUnit as FNOUnit

from .feature_mlp import LinearFeatureMLP as LinearFeatureMLP
from .feature_mlp import ConvFeatureMLP as ConvFeatureMLP

from .normalizers import BatchNormalizer as BatchNormalizer
from .normalizers import LayerNormalizer as LayerNormalizer
from .normalizers import InstanceNormalizer as InstanceNormalizer