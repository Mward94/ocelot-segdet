from abc import abstractmethod
from typing import List, Dict, Any


class Postprocessor:
    """Base class for all other postprocessor classes.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def model_output_keys(self) -> List[str]:
        """A list of model output keys that the postprocessor requires.
        """

    @abstractmethod
    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocesses model outputs.

        Args:
            outputs: A dictionary containing all model outputs with keys. Items with unrecognised
                keys will be passed through.

        Returns:
            A dictionary containing the model outputs after postprocessing.
        """
