from sgnlp.models.dialogue_rnn.eval import eval
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

cfg = parse_args_and_load_config()
eval(cfg)