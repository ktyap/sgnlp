from transformers import PreTrainedConfig


class DrnnConfig(PreTrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sgnlp.models.drnn_roberta.modeling.BieruModel`.
    It is used to instantiate a emotional recurrent unit (ERU) according to the specified arguments, defining the
    model architecture.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    
    Example::
        from sgnlp.models.bieru import BieruConfig
        # Initialize with default values
        config = BieruConfig()
    """
