import numpy as np
from tqdm import tqdm
import time
import logging
import pathlib
import torch
from datetime import datetime
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

from .config import DialogueRNNConfig
from .modeling import DialogueRNNModel
from .preprocess import DialogueRNNPreprocessor
from .utils import configure_dataloaders, parse_args_and_load_config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_model(model, dataloader, no_cuda=False):
    
    losses, preds, labels, masks = [], [], [], []
    
    preprocessor = DialogueRNNPreprocessor(model.transformer_model_family,
                    model.model,
                    model.tokenizer)

    for conversations, label, loss_mask, speaker_mask in tqdm(dataloader, leave=False):

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
        output = model(features, lengths, umask, qmask)
        pred_ = output.prediction
        
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(loss_mask.view(-1).cpu().numpy())

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

def eval(cfg):

    logger.info(f"Training arguments: {vars(cfg)}")

    model_path = pathlib.Path(__file__).resolve().parents[0].joinpath(cfg.model_folder)
    dataset_path = pathlib.Path(__file__).resolve().parents[0].joinpath(cfg.iemocap_dataset_path)

    if cfg.save_eval_res:
        curr_time = datetime.now().strftime("%y%b%d_%H-%M-%S")
        output_path = pathlib.Path(__file__).resolve().parents[0].joinpath(cfg.output_dir)
        if not pathlib.Path(pathlib.PurePath(output_path, curr_time)).exists():
            pathlib.Path(pathlib.PurePath(output_path, curr_time)).mkdir(parents=True, exist_ok=True)
        output_path = pathlib.Path(output_path).joinpath(curr_time)

    global dataset
    global classify
    batch_size = cfg.train_args["batch-size"]
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
    
    config = DialogueRNNConfig.from_pretrained(pathlib.Path(model_path).joinpath('config.json'))
    model = DialogueRNNModel.from_pretrained(pretrained_model_name_or_path=pathlib.Path(model_path).joinpath(cfg.eval_args["model_name"]), config=config)

    if not cfg.no_cuda:
        model.to('cuda')
    else:
        model.to('cpu')

    _, _, test_loader = configure_dataloaders(dataset_path, dataset, classify, batch_size)

    if cfg.save_eval_res:
        lf = open(pathlib.PurePath(output_path, (dataset + '_' + transformer_model + '_mode_' + transformer_mode 
            + '_' + classification_model + '_' + classify + '_test.txt')), 'a')

    start_time = time.time()
    
    test_loss, test_acc, test_fscore, test_label, test_pred, test_mask  = eval_model(model, test_loader, cfg.no_cuda)
    
    x = 'test_loss {} test_acc {} test_fscore {} time {}'.\
            format(test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))

    logger.info(x)

    if cfg.save_eval_res:
        lf.write(x + '\n')

    if cfg.save_eval_res:
        lf.write(str(cfg) + '\n')
        lf.write('Test F1: {}'.format(test_fscore))
        lf.write('\n' + str(classification_report(test_label, test_pred, sample_weight=test_mask, digits=4)) + '\n')
        lf.write(str(confusion_matrix(test_label, test_pred, sample_weight=test_mask)) + '\n')
        lf.write('-'*50 + '\n\n')
        lf.close()

if __name__ == "__main__":
    """Calls the eval method with a pretrained model using test sets.

    Example::
        To run with default parameters:
        python -m eval
        To run with custom training config:
        python -m eval --config config/dialogueRNN_config.json
    """
    cfg = parse_args_and_load_config()
    eval(cfg)
    