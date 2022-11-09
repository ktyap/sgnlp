import numpy as np
from tqdm import tqdm
import time
import logging
import pathlib
import torch
from datetime import datetime
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

from .config import DialogueRNNConfig
from .modeling import DialogueRNNModel
from .modules import MaskedNLLLoss
from .preprocess import DialogueRNNPreprocessor
from .utils import configure_dataloaders, parse_args_and_load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    return optimizer


def train_model(model, dataloader, loss_function, optimizer=None, train=False, no_cuda=False):
    """Run training and evaluation using training and validation sets.

    Args:
        model (DialogueRNNModel): DialogueRNNModel
        dataloader (DataLoader): Dataloader for IEMOCAP dataset
        loss_function (MaskedNLLLoss): MaskedNLLLoss
        optimizer (AdamW, optional): Optimizer
        train (bool, optional): Train if True. Defaults to False.
        no_cuda (bool, optional): Use CPU for training if True. Defaults to False.
    """
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    preprocessor = DialogueRNNPreprocessor(
                    model.transformer_model_family,
                    model.model,
                    model.tokenizer,
                    no_cuda)

    for conversations, label, loss_mask, speaker_mask in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()

        features, lengths, umask, qmask = preprocessor(conversations, speaker_mask)
        
        # create labels and mask
        if no_cuda:
            label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label], 
                                                    batch_first=True)  #.cuda()

            loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask], 
                                                        batch_first=True).long()  #.cuda()
        else:
            label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label], 
                                                    batch_first=True).cuda()

            loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask], 
                                                        batch_first=True).long().cuda()
        
        labels_ = label.view(-1) 

        
        # obtain log probabilities
        output = model(features, lengths, umask, qmask, loss_function, loss_mask, labels_, no_cuda)
        loss, pred_ = output.loss, output.prediction
        
        
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(loss_mask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks)*100, 2)
    
    if dataset in ['iemocap']:
        avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        fscores = [avg_fscore]
        
    elif dataset in ['persuasion', 'multiwoz']:
        avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
        avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
        avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
        fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
        
    elif dataset == 'dailydialog':
        if classify == 'emotion':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='weighted', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore4 = round(f1_score(labels, preds, sample_weight=masks, average='micro', labels=[0,2,3,4,5,6])*100, 2)
            avg_fscore5 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            avg_fscore6 = round(f1_score(labels, preds, sample_weight=masks, average='macro', labels=[0,2,3,4,5,6])*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3, avg_fscore4, avg_fscore5, avg_fscore6]
            
        elif classify == 'act':
            avg_fscore1 = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
            avg_fscore2 = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
            avg_fscore3 = round(f1_score(labels, preds, sample_weight=masks, average='macro')*100, 2)
            fscores = [avg_fscore1, avg_fscore2, avg_fscore3]
    
    return avg_loss, avg_accuracy, fscores, labels, preds, masks 

def train(cfg):

    logger.info(f"Training arguments: {vars(cfg)}")

    model_path = pathlib.Path(__file__).resolve().parents[0].joinpath(cfg.model_folder)
    dataset_path = pathlib.Path(__file__).resolve().parents[0].joinpath(cfg.iemocap_dataset_path)

    if cfg.save_train_res:
        curr_time = datetime.now().strftime("%y%b%d_%H-%M-%S")
        output_path = pathlib.Path(__file__).resolve().parents[0].joinpath(cfg.output_dir)
        if not pathlib.Path(pathlib.PurePath(output_path, curr_time)).exists():
            pathlib.Path(pathlib.PurePath(output_path, curr_time)).mkdir(parents=True, exist_ok=True)
        output_path = pathlib.Path(output_path).joinpath(curr_time)

    global dataset
    global classify
    batch_size = cfg.train_args["batch-size"]
    n_epochs = cfg.train_args["epochs"]
    dataset = cfg.train_args["dataset"]
    classification_model = cfg.train_args["cls-model"]
    transformer_model = cfg.train_args["model"]
    transformer_mode = cfg.train_args["mode"]
    
    if dataset == 'iemocap':
        print ('Classifying emotion in iemocap.')
        classify = cfg.train_args["classify"]
        # n_classes  = 6
        # loss_weights = torch.FloatTensor([1.0, 0.60072, 0.38066, 0.54019, 0.67924, 0.34332])
        
    elif dataset == 'multiwoz':
        print ('Classifying intent in multiwoz.')
        classify = 'intent'
        n_classes  = 11
        
    elif dataset == 'persuasion':
        classify = cfg.train_args["classify"]
        if classify == 'er':
            print ('Classifying persuador in Persuasion for Good.')
            n_classes  = 11
        elif classify == 'ee':
            print ('Classifying persuadee in Persuasion for Good.')
            n_classes  = 13
        else:
            raise ValueError('--classify must be er or ee for persuasion')
            
    elif dataset == 'dailydialog':
        classify = cfg.train_args["classify"]
        if classify == 'emotion':
            print ('Classifying emotion in dailydialog.')
            n_classes  = 7
        elif classify == 'act':
            print ('Classifying act in dailydialog.')
            n_classes  = 4
        else:
            raise ValueError('--classify must be emotion or act for dailydialog')
    
    config = DialogueRNNConfig()
    model = DialogueRNNModel(config)

    if not cfg.no_cuda:
        model.to('cuda')
    else:
        model.to('cpu')


    if cfg.train_args["class_weight"]:
        if cfg.no_cuda:
            loss_function  = MaskedNLLLoss(cfg.train_args["loss_weights"])  #.cuda())
        else:
            loss_function  = MaskedNLLLoss(cfg.train_args["loss_weights"].cuda())
    else:
        loss_function = MaskedNLLLoss()
        
    optimizer = configure_optimizers(model, cfg.train_args["weight_decay"], cfg.train_args["lr"], cfg.train_args["adam_epsilon"])
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset_path, dataset, classify, batch_size)
    
    if cfg.save_train_res:
        lf = open(pathlib.PurePath(output_path, (dataset + '_' + transformer_model + '_mode_' + transformer_mode 
            + '_' + classification_model + '_' + classify + '_train.txt')), 'a')

    valid_losses, valid_fscores = [], []
    # test_fscores = []
    best_loss, best_label, best_pred, best_mask, best_fscore = None, None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_fscore, _, _, _ = train_model(model, train_loader, loss_function, optimizer, True, cfg.no_cuda)
        
        valid_loss, valid_acc, valid_fscore, valid_label, valid_pred, valid_mask = train_model(model, valid_loader, loss_function, None, False, cfg.no_cuda)
        
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        
        # Save model based on best fscore for validation set
        if best_fscore == None or best_fscore > valid_fscore:

            if cfg.save_train_res:
                # Save model
                try:
                    model.save_pretrained(output_path)
                except:
                    pass

            best_loss, best_label, best_pred, best_mask =\
                    valid_loss, valid_label, valid_pred, valid_mask
        
        x = 'Epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} valid_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,\
                round(time.time()-start_time, 2))
        
        logger.info(x)

        if cfg.save_train_res:
            lf.write(x + '\n')

    if cfg.save_train_res:
        lf.write(str(cfg) + '\n')
        lf.write('Best Valid F1: {}'.format(best_fscore))
        lf.write('\n' + str(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)) + '\n')
        lf.write('\n' + 'Confusion matrix (valid)' + '\n')
        lf.write(str(confusion_matrix(best_label, best_pred, sample_weight=best_mask)) + '\n')
        lf.write('-'*50 + '\n\n')
        lf.close()

if __name__ == "__main__":
    """Calls the train method using training and validation sets.

    Example::
        To run with default parameters:
        python -m train
        To run with custom training config:
        python -m train --config config/dialogueRNN_config.json
    """
    cfg = parse_args_and_load_config()
    train(cfg)
    