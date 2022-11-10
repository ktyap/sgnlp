import torch


class DialogueRNNPreprocessor:
    """Class to initialise the Preprocessor for DialogueRNNModel.
    Preprocesses inputs and tokenises them so they can be used with DialogueRNNModel.
    """
    def __init__(self, transformer_model_family, transformer_model, tokenizer=None, no_cuda=None):
        self.transformer_model_family = transformer_model_family
        self.model = transformer_model
        self.tokenizer = tokenizer
        if no_cuda is None:
            if torch.cuda.is_available():
                self.no_cuda = False
            else:
                self.no_cuda = True
        else:
            self.no_cuda = no_cuda

    def __call__(self, conversations, speaker_mask):
        # create umask and qmasks
        lengths = [len(item) for item in conversations]

        umask = torch.zeros(len(lengths), max(lengths)).long()
        for j in range(len(lengths)):
            umask[j][:lengths[j]] = 1
        
        qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask],
                                                batch_first=False).long()
        qmask = torch.nn.functional.one_hot(qmask)
        
        # construct loss_mask
        loss_mask = [[1 for i in sent] for sent in conversations]
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask],
                                                    batch_first=True).long()

        lengths = torch.Tensor(lengths).long()

        utterances = [sent for conv in conversations for sent in conv]

        if self.transformer_model_family == 'sbert':
            features = torch.stack(self.model.encode(utterances, convert_to_numpy=False))  
        
        elif self.transformer_model_family in ['bert', 'roberta']:
            batch = self.tokenizer(utterances, padding=True, return_tensors="pt")
            if self.no_cuda:
                input_ids = batch['input_ids']  #.cuda()
                attention_mask = batch['attention_mask']  #.cuda()
            else:
                self.model.to('cuda')
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                
            _, features = self.model(input_ids, attention_mask, output_hidden_states=True)

            if self.transformer_model_family == 'roberta':
                features = features[:, 0, :]
        
        return {'features': features,
                'lengths': lengths,
                'umask': umask,
                'qmask': qmask}

