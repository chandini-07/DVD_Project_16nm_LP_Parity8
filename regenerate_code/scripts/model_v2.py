import numpy as np

class HybridModelV2:
    """Wrapper combining a shared 4-output model with an optional dedicated Ig model.
    predict(X) returns (n, 4) targets in symlog space, matching the output of v1's TransformedTargetRegressor."""
    def __init__(self, shared_model, dedicated_ig_model=None):
        self.shared_model = shared_model
        self.dedicated_ig_model = dedicated_ig_model
        
    def predict(self, X):
        pred = self.shared_model.predict(X).copy()
        if self.dedicated_ig_model is not None:
            ig_pred = self.dedicated_ig_model.predict(X)
            pred[:, 1] = np.asarray(ig_pred).ravel()
        return pred

class EnsembleIgModel:
    """Ensemble of dedicated Ig models averaged in symlog space."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [np.asarray(m.predict(X)).ravel() for m in self.models]
        return np.mean(preds, axis=0)

class ModelV3Wrapper:
    """Round 3 model wrapper. Takes base v2 model and selectively overrides id, ig, is outputs.
    Returns (n, 4) symlog currents."""
    def __init__(self, base_v2_model, id_model=None, ig_model=None, is_model=None, ib_model=None):
        self.base_v2_model = base_v2_model
        self.id_model = id_model
        self.ig_model = ig_model
        self.is_model = is_model
        self.ib_model = ib_model

    def predict(self, X):
        pred = self.base_v2_model.predict(X).copy()
        if self.id_model is not None:
            pred[:, 0] = np.asarray(self.id_model.predict(X)).ravel()
        if self.ig_model is not None:
            pred[:, 1] = np.asarray(self.ig_model.predict(X)).ravel()
        if self.is_model is not None:
            pred[:, 2] = np.asarray(self.is_model.predict(X)).ravel()
        if self.ib_model is not None:
            pred[:, 3] = np.asarray(self.ib_model.predict(X)).ravel()
        return pred
