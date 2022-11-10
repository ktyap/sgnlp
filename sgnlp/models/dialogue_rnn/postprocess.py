from typing import Any


class DialogueRNNPostprocessor:
    """This class processes :class:`~sgnlp.models.dialogue_rnn.modeling.DialogueRNNModelOutput` to a readable format.
    
    """
    def __init__(self, dataset="iemocap", classify="emotion") -> None:
        if dataset == "iemocap" and classify == "emotion":
            self.label_index_map = {
                0: "Happy",
                1: "Sad",
                2: "Neutral",
                3: "Angry",
                4: "Excited",
                5: "Frustrated"
            }
        else:
            raise ValueError("'dataset' and 'classify' must be 'iemocap' and 'emotion' respectively")
    
    def __call__(self, preds) -> Any:
        raw_preds = preds.prediction.detach().cpu().numpy()
        preds = [self.label_index_map[pred] for pred in raw_preds]

        return preds
