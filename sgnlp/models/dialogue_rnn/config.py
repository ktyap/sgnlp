from .transformers import PretrainedConfig

class DialogueRNNConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~sgnlp.models.drnn_roberta.modeling.DialogueRNNModel`.
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.
    
    Example::
        from sgnlp.models.dialogue_rnn import DialogueRNNConfig
        # Initialize with default values
        config = DialogueRNNConfig()
    """

    def __init__(
        self,
        D_h: int = 1024,
        cls_model: str = "dialogrnn",
        transformer_model_family: str = "roberta",
        mode: str = '0',
        num_classes: int = 6,
        context_attention: str = "general",
        attention: bool = False,
        residual: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.D_h = D_h
        self.cls_model = cls_model
        self.transformer_model_family = transformer_model_family
        self.mode = mode
        self.num_classes = num_classes
        self.context_attention = context_attention
        self.attention = attention
        self.residual = residual

