from dataclasses import dataclass
from typing import List, Optional
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .config import BieruConfig

from torch.utils.data.sampler import SubsetRandomSampler


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
        self.gru = nn.LSTMCell(input_size=config.input_dim, hidden_size=config.input_dim)
        self.ac = nn.Sigmoid()
        self.ac_linear = nn.ReLU()
        self.ac_tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.cnn3 = nn.Conv1d(in_channels=config.in_channels, out_channels=config.out_channels, \
            kernel_size=config.kernel_size, padding=config.padding, stride=config.stride)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()  # sentences (length)

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)
        
    def forward(self, U, mask) -> BieruModelOutput:
        """
        Implements local variant of model.
        :param U:-->seq, batch, dim
        :return:
        """
        v_mask = torch.rand(self.V.size())
        v_mask = torch.where(v_mask > 0.15, torch.full_like(v_mask, 1), torch.full_like(v_mask, 0))  #.cuda()
        self.V = nn.Parameter(self.V * v_mask)

        results1 = torch.zeros(0).type(U.type())
        results2 = torch.zeros(0).type(U.type())
        h = torch.zeros((U.size(1), U.size(2)))  #.cuda()
        c = torch.zeros((U.size(1), U.size(2)))  #.cuda()

        for i in range(U.size()[0]):
            if i == 0:
                #p = U[i]
                #t_t = U[i-1]
                t_t = U[i]
                v_cat = torch.cat((t_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                p = self.dropout(p)
                h, c = self.gru(p, (h, c))
                h3 = self.cnn3(p.unsqueeze(1)).squeeze(1)
                h_cat = torch.cat((h, h3), dim=1)

                h_cat = self.dropout(h_cat)
                results1 = torch.cat((results1, h_cat))

            else:
                
                l_t = U[i-1]
                #l_t = p
                t_t = U[i]
                
                v_cat = torch.cat((l_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                #p = self.ac_linear(p)
                p = self.dropout(p)
                h, c = self.gru(p, (h, c))
                h3 = self.cnn3(p.unsqueeze(1)).squeeze(1)
                h_cat = torch.cat((h, h3), dim=1)
                h_cat = self.dropout(h_cat)
                results1 = torch.cat((results1, h_cat))
        
        rever_U = self._reverse_seq(U, mask)

        for i in range(rever_U.size()[0]):
            # get temp and last, (batch, dim)
            if i == 0:
                #p = rever_U[i]
                #t_t = rever_U[i-1]
                t_t = rever_U[i]
                v_cat = torch.cat((t_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                p = self.dropout(p)
                h, c = self.gru(p, (h, c))

                h3 = self.cnn3(p.unsqueeze(1)).squeeze(1)
                h_cat = torch.cat((h, h3), dim=1)
                h_cat = self.dropout(h_cat)
                results2 = torch.cat((results2, h_cat))
            else:
                
                l_t = rever_U[i-1]
                #l_t = p
                t_t = rever_U[i]
                v_cat = torch.cat((l_t, t_t), dim=1)
                m_cat = v_cat.unsqueeze(1)
                p = self.ac(
                    m_cat.matmul(self.V).matmul(m_cat.transpose(1, 2)).contiguous().view(m_cat.size()[0], -1) + self.W(
                        v_cat))
                #p = self.ac_linear(p)
                p = self.dropout(p)
                # h = self.gru(p, h)
                h, c = self.gru(p, (h, c))

                h3 = self.cnn3(p.unsqueeze(1)).squeeze(1)
                h_cat = torch.cat((h, h3), dim=1)

                h_cat = self.dropout(h_cat)
                results2 = torch.cat((results2, h_cat))

        results2 = results2.contiguous().view(rever_U.size(0), rever_U.size(1), -1)
        results2 = self._reverse_seq(results2, mask)
        results2 = results2.contiguous().view(results1.size(0), results1.size(1))

        #results = torch.log_softmax(self.Ws(results1), dim=1)
        results = torch.log_softmax(self.Ws(torch.cat((results1, results2), dim=1)), dim=1)
        # results = torch.log_softmax(self.Ws(torch.cat((results1, results2, bioutput), dim=1)), dim=1)

        return results
        #return BieruModelOutput(loss=loss, logits=logits)