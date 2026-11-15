import re, subprocess, sys
out = subprocess.run(["ngspice", "-b", sys.argv[1]], capture_output=True, text=True).stdout
v = {k: float(x) for k, x in re.findall(r"(?m)^i\((v[dgsb])\)\s*=\s*(\S+)", out)}
Id, Ig, Is, Ib = -v["vd"], -v["vg"], -v["vs"], -v["vb"]   # current INTO each terminal
print("Id = %+.6e\nIg = %+.6e\nIs = %+.6e\nIb = %+.6e" % (Id, Ig, Is, Ib))
print("sum = %+.3e   (relative to biggest current: %.1e)" % (Id+Ig+Is+Ib, abs(Id+Ig+Is+Ib)/max(abs(Id),abs(Ig),abs(Is),abs(Ib))))
