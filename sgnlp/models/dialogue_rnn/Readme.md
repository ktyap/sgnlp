# Dialogue RNN

Identifying the label of each utterance in a conversation from a set of pre-defined labels that is a set of emotions. The IEMOCAP dataset is used in the current implementation.

The model has been updated with the RoBERTa textual feature extractor to extract utterance level feature vectors in 2020.

[(Link to original paper)](https://arxiv.org/pdf/1811.00405.pdf)
[(Link to paper with RoBERTa textual feature extractor)](https://arxiv.org/pdf/2009.13902.pdf)

## Usage

### Installation

Upon cloning this repository, you can install 'sgnlp' by executing the following command in the directory where setup.py is located:

```
pip install -e .
```

You may simply execute the following command once the Dialogue RNN model is merged with the upstream repository.
```
pip install sgnlp
```

### Dataset

Please refer to the link below to download the dataset. Please download the necessary files in the folder `datasets` according to the folder structure in the next section.

[(Link to IEMOCAP dataset)](https://github.com/declare-lab/dialogue-understanding/tree/master/roberta-end-to-end/datasets)

### Folder structure

Folder structure to use DialogueRNN.

```bash
|-- config
|   |-- config.json
|   |-- dialogueRNN_config.json
└-- datasets
|   └-- dialogue_level_minibatch
|       └-- iemocap
|           |-- iemocap_test_emotion.tsv
|           |-- iemocap_test_loss_mask.tsv
|           |-- iemocap_test_speakers.tsv
|           |-- iemocap_test_utterances.tsv
|           |-- iemocap_train_emotion.tsv
|           |-- iemocap_train_loss_mask.tsv
|           |-- iemocap_train_speakers.tsv
|           |-- iemocap_train_utterances.tsv
|           |-- iemocap_valid_emotion.tsv
|           |-- iemocap_valid_loss_mask.tsv
|           |-- iemocap_valid_speakers.tsv
|           └-- iemocap_valid_utterances.tsv
|-- model
|   |-- config.json
|   └-- pytorch_model.bin
|-- outputs
|   |-- YYMMMDD_HH-mm-SS
|   |   └-- iemocap_roberta_mode_0_dialogrnn_emotion_train.txt
|   |-- YYMMMDD_HH-mm-SS
|   |   └-- iemocap_roberta_mode_0_dialogrnn_emotion_test.txt
|   |-- ...
```

The files `config.json` and `pytorch_model.bin` will be used for evaluating the trained model.
The train or evaluation results will be saved in the `outputs` folder under a timestamped subfolder.
The file `config.json` contains the model config file called during training (if missing, the default model config will be used). And the custom training config file `dialogueRNN_config.json` will be used during training and/or evaluation when executing train or eval (if missing, the default training config will be used).

### Training

Execute the following while in the current project folder:

```
from sgnlp.models.dialogue_rnn.train import train
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

# To run train using custom config, place 'dialogueRNN_config.json' in 'config' folder
cfg = parse_args_and_load_config()

train(cfg)
```

### Evaluation

Please make sure you have a set of trained model config and model file in the `model` folder. Execute the following while in the current project folder:

```
from sgnlp.models.dialogue_rnn.eval import eval
from sgnlp.models.dialogue_rnn.utils import parse_args_and_load_config

# To run train using custom config, place 'dialogueRNN_config.json' in 'config' folder
cfg = parse_args_and_load_config()

eval(cfg)
```

### Usage

The following commands will download a pretrained model config and model file from a remote URL.
Execute the following while in the current project folder:

```
from sgnlp.models.dialogue_rnn.config import DialogueRNNConfig
from sgnlp.models.dialogue_rnn.modeling import DialogueRNNModel
from sgnlp.models.dialogue_rnn.preprocess import DialogueRNNPreprocessor
from sgnlp.models.dialogue_rnn.postprocess import DialogueRNNPostprocessor


config = DialogueRNNConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dialogue_rnn/config.json")
model = DialogueRNNModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/dialogue_rnn/pytorch_model.bin", config=config)

preprocessor = DialogueRNNPreprocessor(model.transformer_model_family, model.model, model.tokenizer)
# To force the use of CPU instead of GPU
# preprocessor = DialogueRNNPreprocessor(model.transformer_model_family, model.model, model.tokenizer, True)

postprocessor = DialogueRNNPostprocessor()

# conversations, speaker_mask
input_batch = {
    'conversations' : [
        ["Hello, how is your day?",
            "It's not been great.",
            "What happened?",
            "I lost my wallet.",
            "And I was late for my appointment.",
            "That was so unlucky. Do you know where you have left it?",
            "No, I have checked everywhere. I just went to the police station to make a report.",
            "Hope that you will find it soon.",
            "Was the appointment important?",
            "It was a job interview and I didn't get the job.",
            "Oh no. But don't worry, I am sure you will come across another great opportunity soon!",
        ]
    ],
    'speaker_mask' : [
        [1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        1
        ]
    ]
}

tensor_dict = preprocessor(**input_batch)

output = model(**tensor_dict)
# To force the use of CPU instead of GPU
# tensor_dict['no_cuda'] = True
# output = model(**tensor_dict)

predictions = postprocessor(output)
print(predictions)
```

`speaker_mask` above refers to the utterances spoken by individual speakers, corresponding to `conversations`.

