from sgnlp.models.dialogue_rnn.train import train
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

cfg = parse_args_and_load_config()
train(cfg)