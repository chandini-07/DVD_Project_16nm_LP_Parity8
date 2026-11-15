import numpy as np
class PerTerminal:
    """4 per-terminal regressors (symlog targets, each standardised). predict() -> (n,4) symlog values like the old shared model."""
    def __init__(self, parts): self.parts = parts          # list of 4: (model, y_scaler) or ('old', column_model)
    def predict(self, X):
        out = np.zeros((len(X), 4))
        for j, (m, ts) in enumerate(self.parts):
            out[:, j] = (m.predict(X)[:, None] if ts is None else ts.inverse_transform(m.predict(X)[:, None]).ravel()) if False else \
                        (m.predict(X)[:, j] if ts is None else ts.inverse_transform(m.predict(X)[:, None]).ravel())
        return out
