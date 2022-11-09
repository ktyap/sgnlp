import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union
from .transformers import PreTrainedModel
from .transformers import BertTokenizer, RobertaTokenizer
from .transformers import BertForSequenceClassification, RobertaForSequenceClassification
from .config import DialogueRNNConfig
from .modules import MatchingAttention, DialogueRNN


@dataclass
class DialogueRNNModelOutput:
    """
    Output type of :class:`~sgnlp.models.lsr.modeling.BieruModel`
    TODO Args:
        prediction (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, max_h_t_count, num_relations)`):
            Prediction scores for all head to tail entity combinations from the final layer.
            Note that the sigmoid function has not been applied at this point.
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when `labels` is provided ):
            Loss on relation prediction task.
    """
    prediction: torch.Tensor
    loss: Optional[Union[torch.FloatTensor, torch.cuda.FloatTensor]] = None

class DialogueRNNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """
    config_class = DialogueRNNConfig
    base_model_prefix = "dialogueRNN"

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the weights
        """
        pass

class DialogueRNNModel(DialogueRNNPreTrainedModel):
    """TODO The Latent Structure Refinement Model performs relation classification on all pairs of entity clusters.
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Args:
        config (:class:`~sgnlp.models.DialogueRNN.config.DialogueRNNConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration.
            Use the :obj:`.from_pretrained` method to load the model weights.
    Example::
        from sgnlp.models.bieru import DialogueRNNModel, DialogueRNNConfig
        # Method 1: Loading a default model
        config = DialogueRNNConfig()
        model = DialogueRNNModel(config)
        # Method 2: Loading from pretrained
        TODO config = DialogueRNNConfig.from_pretrained('https://storage.googleapis.com/sgnlp/models/lsr/config.json')
        TODO model = DialogueRNNModel.from_pretrained('https://storage.googleapis.com/sgnlp/models/lsr/pytorch_model.bin',
                                         config=config)
    """
    def __init__(self, config: DialogueRNNConfig) -> None:
        super().__init__(config)

        if torch.cuda.is_available() and False:  #TODO: update gpu or cpu
            self.FloatTensor = torch.cuda.FloatTensor
            self.LongTensor = torch.cuda.LongTensor
            self.ByteTensor = torch.cuda.ByteTensor

        else:
            self.FloatTensor = torch.FloatTensor
            self.LongTensor = torch.LongTensor
            self.ByteTensor = torch.ByteTensor
        
        if config.transformer_model_family == 'bert':
            if config.mode == '0':
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                hidden_dim = 768
            elif config.mode == '1':
                model = BertForSequenceClassification.from_pretrained('bert-large-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                hidden_dim = 1024
                
        elif config.transformer_model_family == 'roberta':
            if config.mode == '0':
                model = RobertaForSequenceClassification.from_pretrained('roberta-base')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                hidden_dim = 768
            elif config.mode == '1':
                model = RobertaForSequenceClassification.from_pretrained('roberta-large')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                hidden_dim = 1024
                
        elif config.transformer_model_family == 'sbert':
            if config.mode == '0':
                model = SentenceTransformer('bert-base-nli-mean-tokens')
                hidden_dim = 768
            elif config.mode == '1':
                model = SentenceTransformer('bert-large-nli-mean-tokens')
                hidden_dim = 1024
            elif config.mode == '2':
                model = SentenceTransformer('roberta-base-nli-mean-tokens')
                hidden_dim = 768
            elif config.mode == '3':
                model = SentenceTransformer('roberta-large-nli-mean-tokens')
                hidden_dim = 1024

        self.transformer_model_family = config.transformer_model_family
        self.model = model  #.cuda()
        self.hidden_dim = hidden_dim
        self.cls_model = config.cls_model
        self.D_h = config.D_h
        self.residual = config.residual
        self.attention = config.attention
        
        if config.transformer_model_family in ['bert', 'roberta']:
            self.tokenizer = tokenizer
        
        if config.cls_model == 'lstm':
            self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=config.D_h, 
                                num_layers=2, bidirectional=True)  #.cuda()
            self.fc = nn.Linear(self.hidden_dim, 2*config.D_h)  #.cuda()
            
            if config.attention:
                self.matchatt = MatchingAttention(2*config.D_h, 2*config.D_h, att_type='general2')  #.cuda()
            
            self.linear = nn.Linear(2*config.D_h, 2*config.D_h)  #.cuda()
            self.smax_fc = nn.Linear(2*config.D_h, config.num_classes)  #.cuda()
            
        elif config.cls_model == 'dialogrnn':
            self.dialog_rnn_f = DialogueRNN(self.hidden_dim, config.D_h, config.D_h, config.D_h, config.context_attention)  #.cuda()
            self.dialog_rnn_r = DialogueRNN(self.hidden_dim, config.D_h, config.D_h, config.D_h, config.context_attention)  #.cuda()
            self.fc = nn.Linear(self.hidden_dim, 2*config.D_h)  #.cuda()
            
            if config.attention:
                self.matchatt = MatchingAttention(2*config.D_h, 2*config.D_h, att_type='general2')  #.cuda()
            
            self.linear = nn.Linear(2*config.D_h, 2*config.D_h)  #.cuda()
            
            self.smax_fc = nn.Linear(2*config.D_h, config.num_classes)  #.cuda()
            self.dropout_rec = nn.Dropout(0.1)
            
        elif self.cls_model == 'logreg':
            self.linear = nn.Linear(self.hidden_dim, config.D_h)  #.cuda()
            self.smax_fc = nn.Linear(config.D_h, config.num_classes)  #.cuda()

    def pad(
        self, 
        tensor, 
        length
    ):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])  #.cuda()])
        else:
            return tensor
    
    def _reverse_seq(
        self, 
        X, 
        mask
    ):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0,1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)
        
    def forward(
        self, 
        features,  #conversations,
        lengths,
        umask,
        qmask,
        loss_function=None,
        loss_mask=None,
        label=None
    ):

        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
            
        features = torch.stack([self.pad(features.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        
        umask = umask  #.cuda()
        mask = umask.unsqueeze(-1).type(self.FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, 2*self.D_h) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        
        if self.cls_model == 'lstm':
            hidden, _ = self.lstm(features)
            if self.residual:
                features = self.fc(features)
                features = hidden + features   
            else:
                features = hidden
            features = features * mask
            
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = F.relu(self.linear(features))
            
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
            
        elif self.cls_model == 'dialogrnn':
            hidden_f, alpha_f = self.dialog_rnn_f(features, qmask)
            rev_features = self._reverse_seq(features, umask)
            rev_qmask = self._reverse_seq(qmask, umask)
            hidden_b, alpha_b = self.dialog_rnn_r(rev_features, rev_qmask)
            hidden_b = self._reverse_seq(hidden_b, umask)
            
            hidden = torch.cat([hidden_f, hidden_b],dim=-1)
            hidden = self.dropout_rec(hidden)
             
            if self.residual:
                features = self.fc(features)
                features = hidden + features   
            else:
                features = hidden  
            features = features * mask
            
            if self.attention:
                att_features = []
                for t in features:
                    att_ft, _ = self.matchatt(features, t, mask=umask)
                    att_features.append(att_ft.unsqueeze(0))
                att_features = torch.cat(att_features, dim=0)
                hidden = F.relu(self.linear(att_features))
            else:
                hidden = torch.tanh(self.linear(features))
            
            
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
            
        elif self.cls_model == 'logreg':
            hidden = self.linear(features)
            log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        
        # compute loss and metrics
        lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])
        
        if label is not None:
            loss = loss_function(lp_, label, loss_mask)
        else:
            loss = None

        pred_ = torch.argmax(lp_, 1) 

        return DialogueRNNModelOutput(prediction=pred_, loss=loss)