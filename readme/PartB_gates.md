# Part B: gate-level validation (INV, NAND2, NOR2, AND2, OR2)
18 gate/input cases x 200 corners = 3,600 transistor-level SPICE runs (reference), compared with ML leakage built from the transistor models.
Leakage = total current sourced by all VDD-level sources = power / VDD. ML: sum of V x I / VDD over all transistor terminals, ON devices KCL-consistent.
Result: 96.7% mean accuracy (SPICE node voltages), 96.5% with ML-solved node voltages (KCL bisection). Results: results/PartB_gates, plot: plots/PartB_gates/gate_parity.png.
