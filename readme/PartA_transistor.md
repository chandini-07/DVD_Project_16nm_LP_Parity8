# Part A: transistor data and ML models (16nm_LP, VDD 1 V)
Data: 4 cases (NMOS ON/OFF, PMOS ON/OFF) x 10,000 random SPICE samples (VDD +-10%, T -40..125 C, W 1-8 x 16 nm, Vd/Vs 0..VDD,
vth0/toxe/ndep/u0/xj +-10%), one ngspice run per sample, mandatory .options line in every netlist. Kirchhoff check and 3 trend plots in results/ and plots/.
Models: 70/15/15 split (seed 42); 18 features; symlog targets; poly ridge, random forest, gradient boosting, MLP (128-128-64, lbfgs).
The MLP wins in all four cases (selection: lowest 95th-percentile validation error). Rail-augmented retraining (6,000 extra samples with Vd, Vs exactly 0 or VDD) improves drain/source rail-boundary extrapolation, while introducing a trade-off in interior uniform-test accuracy.
Test set R² (symlog) 0.998–0.999. Saved models: models/PartA_transistor/final_<case>.joblib; use scripts/PartA_transistor/ml_models/predict.py.
