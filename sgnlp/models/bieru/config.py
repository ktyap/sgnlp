from typing import Tuple

from transformers import PretrainedConfig


class BieruConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sgnlp.models.bieru.modeling.BieruModel`.
    It is used to instantiate a emotional recurrent unit (ERU) according to the specified arguments, defining the
    model architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    
    Example::
        from sgnlp.models.bieru import BieruConfig
        # Initialize with default values
        config = BieruConfig()
    """

    def __init__(
        self,
        input_dim: int = 100,
        n_class: int = 6,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 5,
        padding: int = 4,
        stride: int = 2,
        dropout_rate: float = 0.8,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_dim = input_dim,
        self.n_class = n_class,
        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.kernel_size = kernel_size,
        self.padding = padding,
        self.stride = stride,
        self.dropout_rate = dropout_rate