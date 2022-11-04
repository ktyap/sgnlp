import numpy as np
from tqdm import tqdm
import time
import logging
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

from .config import DrnnConfig
from .modeling import DrnnModel
from .modules import MaskedNLLLoss
from .preprocess import DrnnPreprocessor
from .utils import configure_dataloaders, parse_args_and_load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def configure_optimizers(model, weight_decay, learning_rate, adam_epsilon):
    "Prepare optimizer"
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


def train_or_eval_model(model, loss_function, dataloader, optimizer=None, train=False):
    losses, preds, labels, masks = [], [], [], []
    assert not train or optimizer!=None
    
    if train:
        model.train()
    else:
        model.eval()
    
    preprocessor = DrnnPreprocessor(model.transformer_model_family,
                    model.model,
                    model.tokenizer)

    for conversations, label, loss_mask, speaker_mask in tqdm(dataloader, leave=False):
        if train:
            optimizer.zero_grad()

        features, lengths, umask, qmask = preprocessor(conversations, speaker_mask)
        
        # create labels and mask
        label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label], 
                                                batch_first=True)  #.cuda()
        
        labels_ = label.view(-1) 

        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask], 
                                                    batch_first=True).long()  #.cuda()

        
        # obtain log probabilities
        output = model(features, lengths, umask, qmask, loss_function, loss_mask, labels_)  #conversations, lengths, umask, qmask)
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
    output_path = pathlib.Path(__file__).resolve().parents[0]
    # print(dataset_path)

    global dataset
    global classify
    # D_h = 1024  #200
    batch_size = cfg.train_args["batch-size"]
    n_epochs = cfg.train_args["epochs"]
    dataset = cfg.train_args["dataset"]
    classification_model = cfg.train_args["cls-model"]
    transformer_model = cfg.train_args["model"]
    transformer_mode = cfg.train_args["mode"]
    # context_attention = cfg.train_args["cattn"]
    # attention = cfg.train_args["attention"]
    # residual = cfg.train_args["residual"]
    
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
    
    #model = DialogBertTransformer(D_h, classification_model, transformer_model, transformer_mode, n_classes, context_attention, attention, residual)
    config = DrnnConfig()
    model = DrnnModel(config)

    if cfg.train_args["class_weight"]:
        loss_function  = MaskedNLLLoss(cfg.train_args["loss_weights"])  #.cuda())
    else:
        loss_function = MaskedNLLLoss()
        
    optimizer = configure_optimizers(model, cfg.train_args["weight_decay"], cfg.train_args["lr"], cfg.train_args["adam_epsilon"])
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loader, valid_loader, test_loader = configure_dataloaders(dataset_path, dataset, classify, batch_size)
    
    if not pathlib.Path(pathlib.PurePath(output_path, 'logs')).exists():
        pathlib.Path(pathlib.PurePath(output_path, 'logs')).mkdir(parents=False, exist_ok=True)
    else:
        pass
    lf = open(pathlib.PurePath(output_path, 'logs', (dataset + '_' + transformer_model + '_mode_' + transformer_mode 
            + '_' + classification_model + '_' + classify + '.txt')), 'a')

    if not pathlib.Path(pathlib.PurePath(output_path, 'results')).exists():
        pathlib.Path(pathlib.PurePath(output_path, 'results')).mkdir(parents=False, exist_ok=True)
    else:
        pass
    rf = open(pathlib.PurePath(output_path, 'results', (dataset + '_' + transformer_model + '_mode_' + transformer_mode 
              + '_' + classification_model + '_' + classify + '.txt')), 'a')

    valid_losses, valid_fscores = [], []
    test_fscores = []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, train_fscore, _, _, _ = train_or_eval_model(model, loss_function,
                                                                           train_loader, optimizer, True)
        
        valid_loss, valid_acc, valid_fscore, _, _, _ = train_or_eval_model(model, loss_function, 
                                                                           valid_loader)
        
        test_loss, test_acc, test_fscore, test_label, test_pred, test_mask  = train_or_eval_model(model, loss_function,
                                                                                                  test_loader)
        
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_fscores.append(test_fscore)
        
        if best_loss == None or best_loss > test_loss:  # FIXED BUG: valid_loss to test_loss
            # Save model
            try:
                model.save_pretrained(model_path)
            except:
                pass
            best_loss, best_label, best_pred, best_mask =\
                    test_loss, test_label, test_pred, test_mask  # FIXED BUG: valid_loss to test_loss
        
        x = 'Epoch {} train_loss {} train_acc {} train_fscore {} valid_loss {} valid_acc {} valid_fscore {} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        print (x)
        lf.write(x + '\n')
        
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()        
        
    print('Test performance.')
    if dataset == 'dailydialog' and classify =='emotion':  
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[1][np.argmin(valid_losses)]
        score4 = test_fscores[1][np.argmax(valid_fscores[1])]
        score5 = test_fscores[2][np.argmin(valid_losses)]
        score6 = test_fscores[2][np.argmax(valid_fscores[2])]
        score7 = test_fscores[3][np.argmin(valid_losses)]
        score8 = test_fscores[3][np.argmax(valid_fscores[3])]
        score9 = test_fscores[4][np.argmin(valid_losses)]
        score10 = test_fscores[4][np.argmax(valid_fscores[4])]
        score11 = test_fscores[5][np.argmin(valid_losses)]
        score12 = test_fscores[5][np.argmax(valid_fscores[5])]
        
        scores = [score1, score2, score3, score4, score5, score6, 
                  score7, score8, score9, score10, score11, score12]
        scores_val_loss = [score1, score3, score5, score7, score9, score11]
        scores_val_f1 = [score2, score4, score6, score8, score10, score12]
        
        print ('Scores: Weighted, Weighted w/o Neutral, Micro, Micro w/o Neutral, Macro, Macro w/o Neutral')
        print('F1@Best Valid Loss: {}'.format(scores_val_loss))
        print('F1@Best Valid F1: {}'.format(scores_val_f1))
        
    elif (dataset=='dailydialog' and classify=='act') or (dataset=='persuasion'):  
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        score3 = test_fscores[1][np.argmin(valid_losses)]
        score4 = test_fscores[1][np.argmax(valid_fscores[1])]
        score5 = test_fscores[2][np.argmin(valid_losses)]
        score6 = test_fscores[2][np.argmax(valid_fscores[2])]
        
        scores = [score1, score2, score3, score4, score5, score6]
        scores_val_loss = [score1, score3, score5]
        scores_val_f1 = [score2, score4, score6]
        
        print ('Scores: Weighted, Micro, Macro')
        print('F1@Best Valid Loss: {}'.format(scores_val_loss))
        print('F1@Best Valid F1: {}'.format(scores_val_f1))
        
    else:
        score1 = test_fscores[0][np.argmin(valid_losses)]
        score2 = test_fscores[0][np.argmax(valid_fscores[0])]
        scores = [score1, score2]
        print('F1@Best Valid Loss: {}; F1@Best Valid F1: {}'.format(score1, score2))
        
    scores = [str(item) for item in scores]
    
    rf.write('\t'.join(scores) + '\t' + str(cfg) + '\n')
    lf.write('\n' + str(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4)) + '\n')
    lf.write(str(confusion_matrix(best_label, best_pred, sample_weight=best_mask)) + '\n')
    lf.write('-'*50 + '\n\n')
    rf.close()
    lf.close()

if __name__ == "__main__":
    cfg = parse_args_and_load_config()
    train(cfg)
    