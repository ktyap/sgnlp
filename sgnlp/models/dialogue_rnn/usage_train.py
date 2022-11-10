from sgnlp.models.dialogue_rnn.train import train
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

cfg = parse_args_and_load_config()
# To run train using custom config, place json file in path as follows
# cfg = parse_args_and_load_config("./config/dialogueRNN_config.json")

train(cfg)