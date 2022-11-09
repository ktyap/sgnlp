import pathlib
from sgnlp.models.dialogue_rnn.config import DialogueRNNConfig
from sgnlp.models.dialogue_rnn.modeling import DialogueRNNModel
from sgnlp.models.dialogue_rnn.preprocess import DialogueRNNPreprocessor
from sgnlp.models.dialogue_rnn.postprocess import DialogueRNNPostprocessor


# model_path = pathlib.Path(__file__).resolve().parents[0].joinpath("bak")
# model_path = pathlib.PurePath(model_path, "model")
# print(model_path)

#config = DialogueRNNConfig.from_pretrained(pathlib.Path(model_path).joinpath("config.json"))
#model = DialogueRNNModel.from_pretrained(pathlib.Path(model_path).joinpath("pytorch_model.bin"), config=config)
config = DialogueRNNConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dialogue_rnn/config.json")
model = DialogueRNNModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/dialogue_rnn/pytorch_model.bin", config=config)
preprocessor = DialogueRNNPreprocessor(model.transformer_model_family, model.model, model.tokenizer)
postprocessor = DialogueRNNPostprocessor()

# conversations, speaker_mask
input_batch = (
    [
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
            "Oh no. But don't worry, I am sure you will land a great offer soon!",
        ]
    ],
    [
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
)

features, lengths, umask, qmask = preprocessor(input_batch[0], input_batch[1])

output = model(features, lengths, umask, qmask)
predictions = postprocessor(output)
print(predictions)
