"""Dev experiment: one case, several terminals. usage: exp.py case n_big act hidden solver epochs"""
import sys, time, numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from mlcommon import load_case, features, symlog, inv_symlog, TARGETS, RAW, FLOOR, ROOT
case=sys.argv[1]; nbig=int(sys.argv[2]); act=sys.argv[3]; hid=tuple(int(x) for x in sys.argv[4].split('x')); solver=sys.argv[5]; ep=int(sys.argv[6])
pol=int(sys.argv[8]) if len(sys.argv)>8 else 0
terms=sys.argv[7].split(',') if len(sys.argv)>7 else TARGETS
d=load_case(case); sc=d['scaler']
bg=pd.read_csv(f'{ROOT}/data/big_{case}.csv'); nr=int(nbig*0.6); big=pd.concat([bg.iloc[:nr],bg.iloc[24000:24000+nbig-nr]]); rv=pd.read_csv(f'{ROOT}/data/railval_{case}.csv')
X=np.r_[d['Xtr'],sc.transform(features(big[RAW]).to_numpy())]; I=np.r_[d['Itr'],big[TARGETS].to_numpy()]; T=symlog(I)
Xv=sc.transform(features(rv[RAW]).to_numpy()); Iv=rv[TARGETS].to_numpy()
for t in terms:
    j=TARGETS.index(t); ts=StandardScaler().fit(T[:,[j]]); y=ts.transform(T[:,[j]]).ravel()
    t0=time.time()
    kw=dict(hidden_layer_sizes=hid,activation=act,random_state=42,alpha=1e-7)
    m=MLPRegressor(solver=solver,max_iter=ep,tol=1e-9,n_iter_no_change=1000,**({'batch_size':256,'learning_rate_init':2e-3} if solver=='adam' else {}),**kw).fit(X,y) if solver=='adam' else MLPRegressor(solver='lbfgs',max_iter=ep,**kw).fit(X,y)
    if pol:
        m.set_params(solver='lbfgs',warm_start=True,max_iter=pol); m.fit(X,y)
    for nm,Xe,Ie in (('railval',Xv,Iv[:,j]),('unif-val',d['Xva'],d['Iva'][:,j])):
        p=inv_symlog(ts.inverse_transform(m.predict(Xe)[:,None]).ravel()); rel=np.abs(p-Ie)/np.maximum(np.abs(Ie),FLOOR)
        print(f'{case} {t} {act} {hid} {solver} n={len(X)} {nm}: median {100*np.median(rel):.2f}% mean {100*rel.mean():.2f}% p95 {100*np.percentile(rel,95):.1f}%  ({time.time()-t0:.0f}s)',flush=True)
