from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class DialogueRNNArguments:
    model_folder: str = field(
        default="model",
        metadata={"help": "Path to save pretrained weights and config"}
    )
    no_cuda: bool = field(
        default=True,
        metadata={"help": "Option to use CPU if True"}
    )
    save_train_res: bool = field(
        default=True,
        metadata={"help": "Option to save training results"}
    )
    save_eval_res: bool = field(
        default=True,
        metadata={"help": "Option to save evaluation results"}
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Path to save outputs (train/eval results and model)"}
    )
    iemocap_dataset_path: str = field(
        default="datasets",
        metadata={"help": "Temporary path for IEMOCAP dataset"}
    )
    train_args: Dict[str, Any] = field(
        default_factory = lambda: {
            "epochs": 15,
            "lr": 0.00001,
            "weight_decay": 0.0,
            "adam_epsilon": 0.00000001,
            "batch-size": 1,
            "cls-model": "dialogrnn",
            "model": "roberta",
            "mode": "0",
            "dataset": "iemocap",
            "class_weight": False,
            "loss_weights": [
                1.0,
                0.60072,
                0.38066,
                0.54019,
                0.67924,
                0.34332
            ],
            "classify": "emotion",
            "cattn": "general",
            "attention": False,
            "residual": True,
        },
        metadata={"help": "Arguments for training."}
    )
    eval_args: Dict[str, Any] = field(
        default_factory= lambda: {
            "batch-size": 1,
            "cls-model": "dialogrnn",
            "model": "roberta",
            "mode": "0",
            "dataset": "iemocap",
            "classify": "emotion",
            "cattn": "general",
            "attention": False,
            "residual": True,
        }
    )

