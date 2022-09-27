from typing import List, Optional
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from .config import BieruConfig

@dataclass
class BieruModelOutput:
    """
    Output type of :class:`~sgnlp.models.lsr.modeling.BieruModel`
    TODO Args:
        prediction (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_h_t_count, num_relations)`):
            Prediction scores for all head to tail entity combinations from the final layer.
            Note that the sigmoid function has not been applied at this point.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when `labels` is provided ):
            Loss on relation prediction task.
    """
    # TODO
    prediction: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None

class BieruPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BieruConfig
    base_model_prefix = "bieru"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights"""
        pass

class BieruModel(BieruPreTrainedModel):
    """TODO The Latent Structure Refinement Model performs relation classification on all pairs of entity clusters.
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Args:
        config (:class:`~sgnlp.models.bieru.config.BieruConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    Example::
        from sgnlp.models.bieru import BieruModel, BieruConfig
        # Method 1: Loading a default model
        config = BieruConfig()
        model = BieruModel(config)
        # Method 2: Loading from pretrained
        TODO config = BieruConfig.from_pretrained('https://storage.googleapis.com/sgnlp/models/lsr/config.json')
        TODO model = BieruModel.from_pretrained('https://storage.googleapis.com/sgnlp/models/lsr/pytorch_model.bin',
                                         config=config)
    """
    def __init__(self, config: BieruConfig) -> None:
        super().__init__(config)
        self.V = nn.Parameter(torch.zeros((config.input_dim, 1, 2*config.input_dim, 2*config.input_dim)))
        self.W = nn.Linear(2*config.input_dim, config.input_dim)
        self.Ws = nn.Linear(2*(52 + config.input_dim), config.n_class)
        self.gru = nn.LSTMCell(input_size = config.input_dim, hidden_size=config.input_dim)
        self.ac = nn.Sigmoid()
        self.ac_linear = nn.ReLU()
        self.ac_tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.cnn3 = nn.Conv1d(in_channels=config.in_channels, out_channels=config.out_channels, \
            kernel_size=config.kernel_size, padding=config.padding, stride=config.stride)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, inputs: List[torch.Tensor], labels: Optional[torch.Tensor] = None) -> BieruModelOutput:
        
        return BieruModelOutput(loss=loss, logits=logits)