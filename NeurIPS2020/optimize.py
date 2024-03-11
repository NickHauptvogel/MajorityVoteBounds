#
# Runs optimization experiments.
#
# Usage: python optimize.py <data_set> [M] [sample_mode] [optimizer] [repeats]
#        M           : number of trees
#        sample_mode : 'bootstrap', 'dim', f in [0,1]
#                      'bootstrap' = full bagging.
#                      'dim'       = sample d points with replacement
#                      float f     = sample f*|S| points with replacement
#        optimizer   : CMA, GD, RProp, iRProp (default)
#        repeats     : integer
#
# Return: results saved in out
#
import sys
import os
import numpy as np
from sklearn.utils import check_random_state

from mvb import NeuralNetworkPostTrainClassifier as EnsembleClassifier
from mvb import data as mldata

if len(sys.argv) < 2:
    print("Data set missing")
    sys.exit(1)
DATASET = sys.argv[1]
M       = int(sys.argv[2]) if len(sys.argv)>=3 else 100
SMODE   = sys.argv[3] if len(sys.argv)>=4 else 'bootstrap'
SMODE   = SMODE if (SMODE=='bootstrap' or SMODE=='dim') else float(SMODE)
OPT     = sys.argv[4] if len(sys.argv)>=5 else 'iRProp'
REPS    = int(sys.argv[5]) if len(sys.argv)>=6 else 1

inpath  = 'NeurIPS2020/data/'
ensemble_path = '../MSc-Thesis/CNN-LSTM_IMDB/results/50_independent_wenzel_bootstr_hold_out_val'
outpath = ensemble_path+'/pac-bayes/'
outfile_path = outpath+DATASET+'-'+str(M)+'-'+str(SMODE)+'-'+str(OPT)+'.csv'
rhos_file_path = outpath+DATASET+'-'+str(M)+'-rhos'+'.csv'

SEED = 1000

def _write_dist_file(rhos, risks):
    with open(rhos_file_path, 'w') as f:
        f.write("h;risk;rho_lam;rho_mv2;rho_mv2u\n")
        for i,(err,r_lam,r_mv,r_mvu) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2])):
            f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_mv)+";"+str(r_mvu)+"\n")

if not os.path.exists(outpath):
    os.makedirs(outpath)
RAND = check_random_state(SEED)

def _write_outfile(results):
    prec = 5
    with open(outfile_path, 'w') as f:
        f.write('repeat;n_train;n_test;d;c;n_min;n2_min')
        for name in ["unf","lam","tnd","dis"]:
            f.write(';'+';'.join([name+'_'+x for x in ['mv_risk_maj_vote', 'mv_risk_softmax_avg', 'gibbs','disagreement','u_disagreement','tandem_risk','pbkl','c1','c2','ctd','tnd','dis','lamda','gamma']]))
        f.write('\n')
        for (rep, n, restup) in results:
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+";"+str(restup[0][1]["n_min"])+";"+str(restup[0][1]["n2_min"]))

            for (mv_risk, stats, bounds, bl, bg) in restup:
                if isinstance(mv_risk, tuple):
                    mv_risk_maj_vote = mv_risk[0]
                    mv_risk_softmax_avg = mv_risk[1]
                else:
                    mv_risk_maj_vote = mv_risk
                    mv_risk_softmax_avg = -1
                f.write(
                        (';'+';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(14)]))
                        .format(mv_risk_maj_vote,
                            mv_risk_softmax_avg,
                            stats.get('gibbs_risk', -1.0),
                            stats.get('disagreement', -1.0),
                            stats.get('u_disagreement', -1.0),
                            stats.get('tandem_risk', -1.0),
                            bounds.get('PBkl', -1.0),
                            bounds.get('C1', -1.0),
                            bounds.get('C2', -1.0),
                            bounds.get('CTD', -1.0),
                            bounds.get('TND', -1.0),
                            bounds.get('DIS',-1.0),
                            bl,
                            bg
                            )
                        )
            f.write('\n')


smodename = 'bagging' if SMODE=='bootstrap' else ('reduced bagging ('+str(SMODE)+');')
print("Starting RFC optimization (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename+", optimizer = "+str(OPT))
results = []
X,Y = mldata.load(DATASET, path=inpath)
for rep in range(REPS):
    if REPS>1:
        print("####### Repeat = "+str(rep+1))

    rf = EnsembleClassifier(M, ensemble_path=ensemble_path)
    # If x is already a tuple, split it
    if isinstance(X, tuple):
        trainX, testX = X
        trainY, testY = Y
    else:
        trainX, trainY, testX, testY = mldata.split(X, Y, 0.8, random_state=RAND)
    C = np.unique(trainY).shape[0]
    n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)
    
    rhos = []
    print("Training...")
    _  = rf.fit(trainX,trainY)
    _, mv_risk = rf.predict(testX,testY)
    stats  = rf.stats()
    bounds, stats = rf.bounds(stats=stats)
    res_unf = (mv_risk, stats, bounds, -1, -1)
        
    # Optimize Lambda
    print("Optimizing lambda...")
    (_, rho, bl) = rf.optimize_rho('Lambda')
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds, stats = rf.bounds(stats=stats)
    res_lam = (mv_risk, stats, bounds, bl, -1)
    rhos.append(rho)
        
    # Optimize TND
    print("Optimizing TND...")
    (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':OPT})
    _, mv_risk = rf.predict(testX,testY)
    stats = rf.aggregate_stats(stats)
    bounds, stats = rf.bounds(stats=stats)
    res_mv2 = (mv_risk, stats, bounds, bl, -1)
    rhos.append(rho)

    # Optimize DIS if binary
    if(C==2):
        print("Optimizing DIS...")
        (_, rho, bl, bg) = rf.optimize_rho('DIS',options={'optimizer':OPT})
        _, mv_risk = rf.predict(testX,testY)
        stats = rf.aggregate_stats(stats)
        bounds, stats = rf.bounds(stats=stats)
        res_mv2u = (mv_risk, stats, bounds, bl, bg)
        rhos.append(rho)
    else:
        res_mv2u = (-1.0, dict(), dict(), -1, -1)
        rhos.append(-np.ones((M,)))
    
    # opt = (bound, rho, lam, gam)
    if rep==0:
        _write_dist_file(rhos, stats['risks'])
    results.append((rep, n, (res_unf, res_lam, res_mv2, res_mv2u)))
    
_write_outfile(results)

