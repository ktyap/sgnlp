import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .config import BieruConfig
from .modeling import BieruModel, BieruModelOutput
from .utils import get_IEMOCAP_loaders, MaskedNLLLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, \
    classification_report, precision_recall_fscore_support

def train_or_eval_model(model, loss_function, dataloader, cuda, epoch, optimizer=None, train=False):

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

        # print(textf)
        # print(textf.shape)  # sentences, 1, d
        # print("umask")
        # print(umask)
        # print(umask.shape)  # 1, sentences
        # print("labels")
        # print(label)  # 1, sentences
        # print("videoSentence")
        # print(videoSentence)  # 1, sentences (quotes comma-sep)

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

def parse_args():

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

    return args


def train_loops(args):
    
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
                                                train_loader, cuda, e, optimizer, True)
        valid_loss, valid_acc, _,_,_, val_fscore = train_or_eval_model(model, loss_function, valid_loader, cuda, e)
        #scheduler.step()
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore = train_or_eval_model(model, loss_function, test_loader, cuda, e)

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


if __name__ == '__main__':
    args = parse_args()
    print(args)
    train_loops(args)