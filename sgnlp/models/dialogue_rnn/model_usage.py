import pathlib
from .config import DialogueRNNConfig
from .modeling import DialogueRNNModel
from .preprocess import DialogueRNNPreprocessor
from .postprocess import DialogueRNNPostprocessor


config = DialogueRNNConfig.from_pretrained("https://storage.googleapis.com/sgnlp/models/dialogue_rnn/config.json")
model = DialogueRNNModel.from_pretrained("https://storage.googleapis.com/sgnlp/models/dialogue_rnn/pytorch_model.bin", config=config)

# preprocessor = DialogueRNNPreprocessor(model.transformer_model_family, model.model, model.tokenizer)
# To force the use of CPU instead of GPU
preprocessor = DialogueRNNPreprocessor(model.transformer_model_family, model.model, model.tokenizer, True)

postprocessor = DialogueRNNPostprocessor()

# conversations, speaker_mask
input_batch = {
    'conversations': [
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

# output = model(**tensor_dict)
# To force the use of CPU instead of GPU
tensor_dict['no_cuda'] = True
output = model(**tensor_dict)

predictions = postprocessor(output)
print(predictions)
