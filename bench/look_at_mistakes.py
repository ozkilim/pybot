import json, os, sys, numpy as np
sys.argv=['x','cv']
sys.path.insert(0,'/Users/ozkilim/pybot/bench'); sys.path.insert(0,'/Users/ozkilim/pybot')
import chooser as C
from sentence_transformers import CrossEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
log=C.Log(); ce=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
H=C.Data("heldout",log,ce); X,y,g,src=H.features(); ok=~np.isnan(y); pred=np.zeros(len(y))
for tr,te in GroupKFold(5).split(X,y,g):
    pred[te]=C.boost().fit(X[tr][ok[tr]],y[tr][ok[tr]]).predict(X[te])
for j,s in enumerate(H.all):
    k=(src==j)&ok; print('%-18s n=%4d AUROC %.3f  mean pred %.2f mean score %.2f'%(s,k.sum(),roc_auc_score(y[k]>=.5,pred[k]),pred[k].mean(),y[k].mean()))
refs={r['id']:r['output'] for r in log.rows}
rowsel=[i for i in range(len(y)) if ok[i] and src[i]<3]
hi=sorted(rowsel,key=lambda i:-(pred[i]-y[i]))[:7]; lo=sorted(rowsel,key=lambda i:(pred[i]-y[i]))[:7]
for title,L in (('PREDICTED GOOD, LOST',hi),('PREDICTED BAD, WON',lo)):
    print('\n###',title)
    for i in L:
        r=H.cands[H.all[src[i]]][g[i]]
        print('- [%s pred %.2f score %.2f] Q: %s\n    REPLY: %s\n    REF:   %s'%(r['source'],pred[i],y[i],r['instruction'][:140],r['reply'][:230],refs[H.ids[g[i]]][:200]))
