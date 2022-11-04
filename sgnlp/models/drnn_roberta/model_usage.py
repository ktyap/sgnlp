import pathlib
from .config import DrnnConfig
from .modeling import DrnnModel
from .preprocess import DrnnPreprocessor
from .postprocess import DrnnPostprocessor


model_path = pathlib.Path(__file__).resolve().parents[0].joinpath("temp")

config = DrnnConfig.from_pretrained(pathlib.Path(model_path).joinpath("config.json"))
model = DrnnModel.from_pretrained(pathlib.Path(model_path).joinpath("pytorch_model.bin"), config=config)
preprocessor = DrnnPreprocessor(model.transformer_model_family, model.model, model.tokenizer)
postprocessor = DrnnPostprocessor()

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
