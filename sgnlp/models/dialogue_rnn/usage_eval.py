from sgnlp.models.dialogue_rnn.eval import eval
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

# To run eval using custom config, place 'dialogueRNN_config.json' in 'config' folder
cfg = parse_args_and_load_config()

eval(cfg)