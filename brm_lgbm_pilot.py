from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

warnings.filterwarnings('ignore')
ROOT = Path('/tmp/brmcheck/tabfsbench/dataset')
OUT = Path.home() / 'output'
SEED = 42
DATASETS = {'heart':'HeartDisease','credit':'class','concrete':'csMPa'}
BUDGET = 12


def encode(train, valid, test):
    lengths = [len(train), len(valid), len(test)]
    all_x = pd.concat([train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)], ignore_index=True)
    obj = list(all_x.select_dtypes(include=['object','category']).columns)
    all_x = pd.get_dummies(all_x, columns=obj, dummy_na=True, dtype=float)
    a,b=lengths[0],lengths[0]+lengths[1]
    tr=all_x.iloc[:a].reset_index(drop=True); va=all_x.iloc[a:b].reset_index(drop=True); te=all_x.iloc[b:].reset_index(drop=True)
    fill = {c: float(tr[c].mean()) for c in tr.columns}
    return tr, va, te, fill


def fit_model(Xtr, ytr, Xva, yva, task, seed):
    common = dict(n_estimators=220, learning_rate=0.04, num_leaves=15,
                  min_child_samples=15, reg_lambda=1.0, colsample_bytree=0.9,
                  random_state=seed, n_jobs=2, verbosity=-1)
    if task == 'classification':
        m = lgb.LGBMClassifier(**common)
        m.fit(Xtr, ytr, eval_set=[(Xva,yva)], eval_metric='auc', callbacks=[lgb.early_stopping(20, verbose=False)])
    else:
        m = lgb.LGBMRegressor(**common)
        m.fit(Xtr, ytr, eval_set=[(Xva,yva)], eval_metric='rmse', callbacks=[lgb.early_stopping(20, verbose=False)])
    return m


def predict_metric(model, X, y, task):
    if task == 'classification':
        p = model.predict_proba(X)[:,1]
        return roc_auc_score(y,p)
    return mean_squared_error(y, model.predict(X)) ** 0.5


def loss_from_metric(v, task): return 1-v if task=='classification' else v

def better(a,b,task): return a>b if task=='classification' else a<b


def ranks(X, y):
    vals=[]
    for c in X.columns:
        z=X[c].to_numpy(float)
        r=np.corrcoef(z,y)[0,1] if np.std(z)>0 else 0
        vals.append((c, abs(r) if np.isfinite(r) else 0))
    return [x[0] for x in sorted(vals,key=lambda z:z[1],reverse=True)]


def make_masks(cols, order, rng, random_per_degree=3):
    p=len(cols); out=[]
    for c in cols: out.append(('single',1/p,frozenset([c])))
    for d in (.2,.4,.6,.8):
        k=min(p-1,max(1,round(d*p)))
        out.append(('most',d,frozenset(order[:k])))
        out.append(('least',d,frozenset(order[-k:])))
        for _ in range(random_per_degree):
            out.append(('random',d,frozenset(rng.choice(cols,k,replace=False).tolist())))
    seen=set(); unique=[]
    for x in out:
        key=(x[0],x[2])
        if key not in seen: seen.add(key); unique.append(x)
    return unique


def candidate_subsets(cols, order, rng):
    p=len(cols); sets={frozenset(cols)}
    for c in cols: sets.add(frozenset(x for x in cols if x!=c))
    for d in (.2,.4,.6,.8):
        k=min(p-1,max(1,round(d*p)))
        sets.add(frozenset(x for x in cols if x not in order[:k]))
        sets.add(frozenset(x for x in cols if x not in order[-k:]))
    for _ in range(35):
        d=float(rng.choice([.2,.4,.6,.8])); k=min(p-1,max(1,round(d*p)))
        miss=set(rng.choice(cols,k,replace=False).tolist())
        sets.add(frozenset(x for x in cols if x not in miss))
    return list(sets)


def design_availability(cols, order, rng, n=250):
    p=len(cols); masks=[]
    masks.append(frozenset(cols))
    for c in cols: masks.append(frozenset(x for x in cols if x!=c))
    for _ in range(n):
        d=float(rng.choice([.2,.4,.6,.8])); k=min(p-1,max(1,round(d*p)))
        mode=rng.choice(['random','most','least'],p=[.6,.2,.2])
        if mode=='most': miss=set(order[:k])
        elif mode=='least': miss=set(order[-k:])
        else: miss=set(rng.choice(cols,k,replace=False).tolist())
        masks.append(frozenset(x for x in cols if x not in miss))
    return masks


def greedy_bank(cands, val_losses, availability, fallback_loss, budget):
    selected=[]; current=np.full(len(availability),fallback_loss,float)
    remaining=set(range(len(cands)))
    for _ in range(min(budget,len(cands))):
        best=None; best_obj=float(current.mean()); best_vec=None
        for j in remaining:
            eligible=np.array([cands[j].issubset(a) for a in availability])
            trial=np.where(eligible,np.minimum(current,val_losses[j]),current)
            obj=trial.mean()
            if obj < best_obj-1e-12: best,best_obj,best_vec=j,obj,trial
        if best is None: break
        selected.append(best); remaining.remove(best); current=best_vec
    return selected


def run_dataset(name,target):
    df=pd.read_csv(ROOT/name/f'{name}.csv')
    task='regression' if name=='concrete' else 'classification'
    X=df.drop(columns=[target]); y=df[target]
    if task=='classification': y=LabelEncoder().fit_transform(y.astype(str))
    Xtr0,Xte0,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y if task=='classification' else None)
    Xtr0,Xva0,ytr,yva=train_test_split(Xtr0,ytr,test_size=.2,random_state=SEED,stratify=ytr if task=='classification' else None)
    Xtr,Xva,Xte,fill=encode(Xtr0,Xva0,Xte0)
    ytr=np.asarray(ytr); yva=np.asarray(yva); yte=np.asarray(yte)
    cols=list(Xtr.columns); order=ranks(Xtr,np.asarray(ytr,float)); rng=np.random.default_rng(SEED+len(cols))
    cache={}
    def get_model(sub):
        key=tuple(sorted(sub))
        if key not in cache:
            cc=[c for c in cols if c in sub]; cache[key]=fit_model(Xtr[cc],ytr,Xva[cc],yva,task,SEED)
        return cache[key]
    full=frozenset(cols); full_model=get_model(full)
    full_metric=predict_metric(full_model,Xte,yte,task)
    cands=candidate_subsets(cols,order,rng)
    val_losses=[]
    for s in cands:
        m=get_model(s); val_losses.append(loss_from_metric(predict_metric(m,Xva[[c for c in cols if c in s]],yva,task),task))
    if task=='classification': fallback_loss=0.5
    else: fallback_loss=float(np.sqrt(np.mean((np.asarray(yva)-np.mean(ytr))**2)))
    avail_design=design_availability(cols,order,rng)
    selected=greedy_bank(cands,np.array(val_losses),avail_design,fallback_loss,BUDGET)
    bank=[cands[j] for j in selected]; bank_loss=[val_losses[j] for j in selected]
    eval_masks=make_masks(cols,order,np.random.default_rng(SEED+999+len(cols)))
    rows=[]
    for scenario,degree,missing in eval_masks:
        available=frozenset(x for x in cols if x not in missing)
        Xm=Xte.copy()
        for c in missing: Xm[c]=fill[c]
        mean_metric=predict_metric(full_model,Xm,yte,task)
        Xn=Xte.copy()
        for c in missing: Xn[c]=np.nan
        nan_metric=predict_metric(full_model,Xn,yte,task)
        oracle=get_model(available)
        oracle_metric=predict_metric(oracle,Xte[[c for c in cols if c in available]],yte,task)
        eligible=[i for i,s in enumerate(bank) if s.issubset(available)]
        if eligible:
            chosen=min(eligible,key=lambda i:bank_loss[i]); s=bank[chosen]; m=get_model(s)
            brm_metric=predict_metric(m,Xte[[c for c in cols if c in s]],yte,task); route='compatible'
        else:
            brm_metric=0.5 if task=='classification' else float(np.sqrt(np.mean((np.asarray(yte)-np.mean(ytr))**2))); route='constant'
        for method,metric in [('TabFS mean/mode',mean_metric),('Native NaN',nan_metric),('Budget-BRM-LGBM',brm_metric),('Oracle reduced',oracle_metric)]:
            gap=(full_metric-metric)/full_metric if task=='classification' else (metric-full_metric)/full_metric
            rows.append(dict(dataset=name,task=task,scenario=scenario,degree=degree,n_missing=len(missing),method=method,metric=metric,full_metric=full_metric,relative_gap=gap,route=route if method=='Budget-BRM-LGBM' else 'n/a',bank_size=len(bank),candidate_models=len(cands)))
    meta=dict(dataset=name,task=task,rows=len(df),features=len(cols),full_metric=full_metric,bank_size=len(bank),candidate_models=len(cands),selected_feature_counts=[len(s) for s in bank],fallback_rate=np.mean([r['route']=='constant' for r in rows if r['method']=='Budget-BRM-LGBM']))
    return rows,meta

all_rows=[]; metas=[]; t0=time.time()
for name,target in DATASETS.items():
    rows,meta=run_dataset(name,target); all_rows.extend(rows); metas.append(meta)
res=pd.DataFrame(all_rows)
res.to_csv(OUT/'pilot_results.csv',index=False)
summary=(res.groupby(['dataset','method'],as_index=False).agg(mean_gap=('relative_gap','mean'),worst_gap=('relative_gap','max'),mean_metric=('metric','mean'),bank_size=('bank_size','max')))
summary.to_csv(OUT/'pilot_summary.csv',index=False)
with open(OUT/'pilot_metadata.json','w') as f: json.dump({'seed':SEED,'budget':BUDGET,'runtime_sec':time.time()-t0,'datasets':metas},f,indent=2)
plot=summary.copy(); plot['mean_gap_pct']=100*plot['mean_gap']
fig=px.bar(plot,x='dataset',y='mean_gap_pct',color='method',barmode='group',title='Feature-shift degradation in pilot<br><span style="font-size:18px;font-weight:normal">Real TabFSBench data | Lower is better</span>')
fig.update_xaxes(title_text='Dataset'); fig.update_yaxes(title_text='Gap (%)'); fig.update_traces(cliponaxis=False)
fig.update_layout(legend=dict(orientation='h',yanchor='bottom',y=1.05,xanchor='center',x=.5))
fig.write_image(OUT/'pilot_gap.png')
with open(OUT/'pilot_gap.png.meta.json','w') as f: json.dump({'caption':'Pilot feature-shift degradation','description':'Grouped bars compare mean relative degradation for four LightGBM handling strategies on three real datasets.'},f)
print(summary.to_string(index=False)); print(json.dumps(metas,indent=2)); print('runtime',time.time()-t0)
