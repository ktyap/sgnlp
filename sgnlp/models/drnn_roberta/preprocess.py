import torch


class DrnnPreprocessor:
    """Class to initialise the Preprocessor for DrnnModel.
    Preprocesses inputs and tokenises them so they can be used with DrnnModel.

    Args:

    Returns:
        features:
        lengths:
        umask:
        qmask:
    """
    def __init__(self, transformer_model_family, transformer_model, tokenizer=None):
        self.transformer_model_family = transformer_model_family
        self.model = transformer_model
        self.tokenizer = tokenizer

    def __call__(self, conversations, speaker_mask):  #loss_mask, speaker_mask):
        # create umask and qmasks
        lengths = [len(item) for item in conversations]

        umask = torch.zeros(len(lengths), max(lengths)).long()
        for j in range(len(lengths)):
            umask[j][:lengths[j]] = 1
        
        qmask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in speaker_mask],
                                                batch_first=False).long()
        qmask = torch.nn.functional.one_hot(qmask)

        # # create labels and mask
        # label = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in label],
        #                                         batch_first=True)
        
        # construct loss_mask
        loss_mask = [[1 for i in sent] for sent in conversations]
        loss_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in loss_mask],
                                                    batch_first=True).long()

        lengths = torch.Tensor(lengths).long()
        #start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent for conv in conversations for sent in conv]

        if self.transformer_model_family == 'sbert':
            features = torch.stack(self.model.encode(utterances, convert_to_numpy=False))  
        
        elif self.transformer_model_family in ['bert', 'roberta']:
            batch = self.tokenizer(utterances, padding=True, return_tensors="pt")
            input_ids = batch['input_ids']  #.cuda()
            attention_mask = batch['attention_mask']  #.cuda()
            _, features = self.model(input_ids, attention_mask, output_hidden_states=True)
            # print(_)
            # print(len(features))
            if self.transformer_model_family == 'roberta':
                features = features[:, 0, :]
                # features = torch.mean(features, dim=1)
        
        # features = torch.stack([self.pad(features.narrow(0, s, l), max(lengths))
        #                        for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
        
        return features, lengths, umask, qmask

