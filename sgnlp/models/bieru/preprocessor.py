import json
import re
import unicodedata

class baseTokenizer():
    """Base Tokenizer Class (Inherited by all subclasses) """
    def __init__(self):
        self.PAD_WORD = '[PAD]'
        self.UNK_WORD = '[UNK]'
    
    def unicodeToAscii(self, utterance):
        """ Normalize strings"""
        return ''.join(
            c for c in unicodedata.normalize('NFD', utterance)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, raw_utterance):
        """Remove nonalphabetics for each utterance"""
        str = self.unicodeToAscii(raw_utterance.lower().strip())
        str = re.sub(r"([,.'!?])", r" \1", str)
        str = re.sub(r"[^a-zA-Z,.'!?]+", r" ", str)
        return str
    
    def process(self, utterance):
        pass

class gloveTokenizer(baseTokenizer):
    """Glove Tokenizer for Glove Embedding (End2End Model)"""
    def __init__(self, vocab_path):
        super(gloveTokenizer, self).__init__()
        self.PAD = 0
        self.UNK = 1
        self.word2id = None
        self.loadVocabFromJson(vocab_path)

    def loadVocabFromJson(self, path):
        self.word2id = json.load(open(path))

    def process(self, utterance):
        # baseTokenizer.normalizeString : remove nonalphabetics
        utterance = self.normalizeString(utterance)
        # transform into lower mode.
        wordList = [word.lower() for word in utterance.split()]
        indexes = [self.word2id.get(word, self.UNK) for word in wordList]  # unk: 1
        return indexes

class BieruPreprocessor:
    def __init__(self):
        pass

