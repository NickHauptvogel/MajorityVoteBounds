#
# Runs experiments using standard random forest with uniform weighting.
#
# Usage: python uniform.py <data_set> [M] [sample_mode] [repeats]
#        M           : number of trees
#        sample_mode : 'bootstrap', 'dim', f in [0,1]
#                      'bootstrap' = full bagging.
#                      'dim'       = sample d points with replacement
#                      float f     = sample f*|S| points with replacement
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
REPS    = int(sys.argv[4]) if len(sys.argv)>=5 else 1

inpath  = 'NeurIPS2020/data/'
ensemble_path = '../MSc-Thesis/ResNet20_CIFAR/results/50_independent_wenzel_no_checkp_bootstr'
outpath = ensemble_path+'/pac-bayes/'
outfile_path = outpath+DATASET+'-'+str(M)+'-'+str(SMODE)+'-uniform.csv'

SEED = 1000
if not os.path.exists(outpath):
    os.makedirs(outpath)
RAND = check_random_state(SEED)

def _write_outfile(results):
    prec = 5
    with open(outfile_path, 'w') as f:
        f.write('repeat;n_train;n_test;d;c;mv_risk_maj_vote;mv_risk_softmax_avg;gibbs;n_min;disagreement;disagreement_t;tandem_risk;n2_min;n2_min_t;pbkl;c1;c2;tnd;ctd;dis;dis_t;sh;best_tree;worst_tree\n')
        for (rep, risk_mv, n, bounds, stats) in results:
            if isinstance(risk_mv, tuple):
                mv_risk_maj_vote = risk_mv[0]
                mv_risk_softmax_avg = risk_mv[1]
            else:
                mv_risk_maj_vote = risk_mv
                mv_risk_softmax_avg = -1
            f.write(str(rep+1)+';'+str(n[0])+';'+str(n[1])+';'+str(n[2])+';'+str(n[3])+';'+
                    (';'.join(['{'+str(i)+':.'+str(prec)+'f}' for i in range(19)])+'\n')
                    .format(mv_risk_maj_vote,
                        mv_risk_softmax_avg,
                        stats['gibbs_risk'],
                        stats['n_min'],
                        stats['disagreement'],
                        stats.get('u_disagreement', -1.0),
                        stats['tandem_risk'],
                        stats['n2_min'],
                        stats['u_n2_min'],
                        bounds['PBkl'],
                        bounds.get('C1', -1.0),
                        bounds.get('C2', -1.0),
                        bounds['TND'],
                        bounds['CTD'],
                        bounds.get('DIS',-1.0),
                        bounds.get('DIS-T',-1.0),
                        bounds.get('SH',-1.0),
                        np.min(stats['risks']),
                        np.max(stats['risks']))
                    )

smodename = 'bagging' if SMODE=='bootstrap' else ('reduced bagging ('+str(SMODE)+');')
print("Starting RFC experiment (m = "+str(M)+") for ["+DATASET+"] using sampling strategy: "+smodename)
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
    
    print("Training...")
    _  = rf.fit(trainX,trainY)

    print("Evaluating final classifier...")
    _, mv_risk = rf.predict(testX,testY)

    print("Computing bounds...")
    stats  = rf.stats()
    bounds, stats = rf.bounds(stats=stats)
    results.append((rep, mv_risk, n, bounds, stats))
    
_write_outfile(results)

