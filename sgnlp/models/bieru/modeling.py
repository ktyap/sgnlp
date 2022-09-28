from dataclasses import dataclass
from typing import List, Optional
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from .config import BieruConfig

from .utils import get_IEMOCAP_loaders, MaskedNLLLoss

import time
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import argparse


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
        mask_sum = torch.sum(mask, 1).int()

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
        v_mask = torch.where(v_mask > 0.15, torch.full_like(v_mask, 1), torch.full_like(v_mask, 0)).cuda()
        self.V = nn.Parameter(self.V * v_mask)

        results1 = torch.zeros(0).type(U.type())
        results2 = torch.zeros(0).type(U.type())
        h = torch.zeros((U.size(1), U.size(2))).cuda()
        c = torch.zeros((U.size(1), U.size(2))).cuda()

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
                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
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
                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
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

                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
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

                h3 = self.cnn3(p.unsqueeze(0)).squeeze(0)
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


def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()
    #it = 0
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        textf, visuf, acouf, qmask, umask, label = \
            [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        print(textf)
        print(textf.shape)
        print("umask")
        print(umask)
        print(umask.shape)
        print("labels")
        print(label)

        # log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask) # seq_len, batch, n_classes
        log_prob = model(textf, umask)  # batch*seq_len, n_classes
        labels_ = label.view(-1)  # batch*seq_len
        loss = loss_function(log_prob, labels_, umask)

        #if train and it % 10 == 0:
        #    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model,
        #                                                                                            loss_function,
        #                                                                                            test_loader, e)
        #    print(test_acc)
        pred_ = torch.argmax(log_prob, 1)  # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if train:
            loss.backward()
            # print(torch.mean(model.V.grad))
            optimizer.step()
        #it += 1
    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan')

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    batch_size = args.batch_size
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs

    # TODO: remove D_m = 100


    config = BieruConfig()
    model = BieruModel(config)



    #model = RNTN(D_m, n_classes, False)
    print('\n number of parameters {}'.format(sum([p.numel() for p in model.parameters()])))
    if cuda:
        model.cuda()
    loss_weights = torch.FloatTensor([
                                        1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668,
                                        ])

    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.99)

    train_loader, valid_loader, test_loader =\
            get_IEMOCAP_loaders(r'./bieru/IEMOCAP_features_raw.pkl',
                                valid=0.0,
                                batch_size=batch_size,
                                num_workers=2)

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_, train_fscore = train_or_eval_model(model, loss_function,
                                                train_loader, e, optimizer, True)
        valid_loss, valid_acc, _,_,_, val_fscore = train_or_eval_model(model, loss_function, valid_loader, e)
        #scheduler.step()
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask =\
                    test_loss, test_label, test_pred, test_mask

        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))



    print('Test performance..')
    print('Loss {} accuracy {}'.format(best_loss,
                                     round(accuracy_score(best_label, best_pred, sample_weight=best_mask)*100,2)))
    print(classification_report(best_label,best_pred,sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))