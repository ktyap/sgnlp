from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BieruArguments:
    model_folder: str = field(
        default="model/",
        metadata={"help": "Path to save pretrained weights and config"}
    )
    no_cuda: str = field(
        default=False,
        metadata={"help": "Option to use CPU if True"}
    )
    iemocap_dataset_path: str = field(
        default="temp/",
        metadata={"help": "Temporary path for IEMOCAP dataset"}
    )
    iemocap_dataset_name: str = field(
        default="IEMOCAP_features_raw.pkl",
        metadata={"help": "File name of IEMOCAP dataset."}
    )
    train_args: Dict[str, Any] = field(
        default_factory = lambda: {
            "epochs": 20,
            "lr": 0.0001,
            "l2": 0.001,
            "loss_weights": [1/0.086747,
                            1/0.144406,
                            1/0.227883,
                            1/0.160585,
                            1/0.127711,
                            1/0.252668,
                            ],
            "class_weight": True,
            "valid": 0.0,  # validation set ratio from trainset
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
        },
        metadata={"help": "Arguments for training."},
    )
    eval_args: Dict[str, Any] = field(
        default_factory= lambda: {
            "batch_size": 1,
            "num_workers": 0,
            "model_name": "pytorch_model.bin"
        }
    )