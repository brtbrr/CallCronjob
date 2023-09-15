from typing import Any


def predict(config: dict[str, Any]):
    from *******.src.predict import Predictor
    predictor = Predictor(config)
    predictor.step_predict()
