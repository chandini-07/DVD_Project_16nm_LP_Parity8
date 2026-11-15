import sys, time, joblib, numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from mlcommon import load_case, features, symlog, inv_symlog, TARGETS, RAW, FLOOR, ROOT, CASES
from pert import PerTerminal
NBIG=int(sys.argv[1]); HID=tuple(int(x) for x in sys.argv[2].split('x')); ADAM=int(sys.argv[3]); POL=int(sys.argv[4]); tag=sys.argv[5]
cases=sys.argv[6].split(',') if len(sys.argv)>6 else ['nmos_off','pmos_off','nmos_on','pmos_on']
skip={'nmos_off':['ig'],'pmos_off':['ig']}          # ~1e-18 A, irrelevant to leakage -> keep old model's column
for case in cases:
    d=load_case(case); sc=d['scaler']; old=joblib.load(f'{ROOT}/models/orig/final_{case}.joblib')['model']
    bg=pd.read_csv(f'{ROOT}/data/big_{case}.csv'); nr=int(NBIG*0.6); big=pd.concat([bg.iloc[:nr],bg.iloc[24000:24000+NBIG-nr]])
    X=np.r_[d['Xtr'],sc.transform(features(big[RAW]).to_numpy())]; T=symlog(np.r_[d['Itr'],big[TARGETS].to_numpy()])
    parts=[]
    for j,t in enumerate(TARGETS):
        if t in skip.get(case,[]): parts.append((old,None)); continue
        t0=time.time(); ts=StandardScaler().fit(T[:,[j]]); y=ts.transform(T[:,[j]]).ravel()
        m=MLPRegressor(hidden_layer_sizes=HID,activation='tanh',solver='adam',batch_size=256,learning_rate_init=2e-3,max_iter=ADAM,tol=1e-9,n_iter_no_change=1000,alpha=1e-7,random_state=42).fit(X,y)
        m.set_params(solver='lbfgs',warm_start=True,max_iter=POL); m.fit(X,y); parts.append((m,ts))
        print(case,t,'trained %.0fs'%(time.time()-t0),flush=True)
    model=PerTerminal(parts); joblib.dump(dict(model=model,scaler=sc),f'{ROOT}/models/v2_{tag}_{case}.joblib')
    # report: uniform TEST split, rail_test, railval
    rv=pd.read_csv(f'{ROOT}/data/railval_{case}.csv'); rt=pd.read_csv(f'{ROOT}/data/rail_test_{case}.csv')
    for nm,Xe,Ie in (('uniform-test',d['Xte'],d['Ite']),('rail_test',sc.transform(features(rt[RAW]).to_numpy()),rt[TARGETS].to_numpy()),('railval',sc.transform(features(rv[RAW]).to_numpy()),rv[TARGETS].to_numpy())):
        P=inv_symlog(model.predict(Xe)); rel=np.abs(P-Ie)/np.maximum(np.abs(Ie),FLOOR)
        print(case,nm,'median%',np.round(100*np.median(rel,0),2),'mean%',np.round(100*rel.mean(0),2),flush=True)
print('ALLDONE',flush=True)
