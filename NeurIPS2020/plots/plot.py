import numpy as np
import pandas as pd
import os
import sys

DATASETS = [
        'CIFAR-10'
        ]
EXP_PATH  = "../out/"
NUM_TREES = 20
BOUNDS_BINARY = [("pbkl","FO"),("c1","Cone"),("c2","Ctwo"),("ctd","CTD"),("tnd","TND"),("dis","DIS")]
BOUNDS_MULTI  = BOUNDS_BINARY[:1]+BOUNDS_BINARY[3:-1]

# Plot error and bounds for several data sets
def multi_bounds(exp="uniform"):
    name = "bounds_"+exp
    path = name+"/datasets/"
    if not os.path.isdir(path):
        os.makedirs(path)

    for ds in DATASETS:
        df = pd.read_csv(EXP_PATH+exp+"/"+ds+"-"+str(NUM_TREES)+"-bootstrap.csv",sep=";")
        df_mean = df.mean()
        df_std  = df.std()
        bounds = BOUNDS_BINARY if df_mean["c"]==2 else BOUNDS_MULTI
        with open(path+ds+".tex", "w") as f:
            for i,(bnd,cls) in enumerate(bounds):
                f.write("\\addplot["+cls+", Bound]coordinates {("+str(i+1)+","+str(df_mean[bnd])+") +- (0,"+str(df_std[bnd])+")};\n")
                
            mean, err = df_mean['mv_risk'], df_std['mv_risk']
            up   = str(mean+err)
            lw   = str(mean-err)
            mean = str(mean)
            f.write("\\addplot+[RiskErr,name path=UP] {"+up+"};\n")
            f.write("\\addplot+[RiskErr,name path=LW] {"+lw+"};\n")
            f.write("\\addplot[RiskErr] fill between[of=UP and LW];\n")
            f.write("\\addplot[Risk] coordinates {(0,"+mean+") (7,"+mean+")};\n")

 

# Prep data for optimized MV risk comparison 
def optimized_risk_comparison():
    name = "risk_comparison_optimized"
    path = name+"/datasets/"
    if not os.path.isdir(path):
        os.makedirs(path)

    opts = ["lam","tnd","dis"]
    cols = ["dataset"]
    for opt in opts:
        cols += [opt+suf for suf in ["_diff","_q25","_q75"]]
    rows_bin = []
    rows_mul = []
    for ds in DATASETS:
        df = pd.read_csv(EXP_PATH+"optimize/"+ds+"-"+str(NUM_TREES)+"-bootstrap-iRProp.csv",sep=";")
        if (df["unf_mv_risk"]==0).sum() > 0:
            continue
        row = [ds]
        for opt in opts:
            diff   = df[opt+"_mv_risk"]/df["unf_mv_risk"]
            med = diff.median()
            row += [med, med-diff.quantile(0.25), diff.quantile(0.75)-med]
        if df["c"].iloc[0]==2:
            rows_bin.append(row)
        else:
            rows_mul.append(row)
    
    pd.DataFrame(data=rows_bin, columns=cols).to_csv(path+"bin.csv", sep=";", index_label="idx")
    pd.DataFrame(data=rows_mul, columns=cols).to_csv(path+"mul.csv", sep=";", index_label="idx")






if len(sys.argv)==1:
    multi_bounds(exp="uniform")
    optimized_risk_comparison()
elif sys.argv[1]=="uniform":
    multi_bounds(exp="uniform")
elif sys.argv[1]=="optimize":
    optimized_risk_comparison()

