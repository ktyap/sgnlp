from sgnlp.models.dialogue_rnn.train import train
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

# To run train using custom config, place 'dialogueRNN_config.json' in 'config' folder
cfg = parse_args_and_load_config()

train(cfg)