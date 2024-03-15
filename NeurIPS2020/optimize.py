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
import argparse
import sys
import os
import numpy as np
from sklearn.utils import check_random_state

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvb import NeuralNetworkPostTrainClassifier as EnsembleClassifier
from mvb import data as mldata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset to use')
    parser.add_argument('--M', type=int, help='Number of trees', required=False, default=100)
    parser.add_argument('--smode', type=str, help='Sampling mode', required=False, default='bootstrap')
    parser.add_argument('--opt', type=str, help='Optimizer to use', required=False, default='iRProp')
    parser.add_argument('--reps', type=int, help='Number of repetitions', required=False, default=1)
    parser.add_argument('--inpath', type=str, required=False, default='NeurIPS2020/data/')
    parser.add_argument('--ensemble_path', type=str, required=False, default='../MSc-Thesis/ResNet20_CIFAR/results/10_checkp_every_40_wenzel_0_2_val')
    parser.add_argument('--write_files', action='store_true', help='Write files', required=False, default=False)

    args = parser.parse_args()
    DATASET = args.dataset
    M = args.M
    SMODE = args.smode
    OPT = args.opt
    REPS = args.reps

    inpath  = args.inpath
    ensemble_path = args.ensemble_path
    write_files = args.write_files

    optimize(DATASET, M, SMODE, OPT, REPS, inpath, ensemble_path, write_files)


def optimize(dataset, m, smode, opt, reps, inpath, ensemble_path, write_files, indices=None):
    outpath = ensemble_path + '/pac-bayes/'
    outfile_path = outpath + dataset + '-' + str(m) + '-' + str(smode) + '-' + str(opt) + '.csv'
    rhos_file_path = outpath + dataset + '-' + str(m) + '-rhos' + '.csv'

    seed = 1000

    def _write_dist_file(rhos, risks):
        with open(rhos_file_path, 'w') as f:
            f.write("h;risk;rho_lam;rho_mv2;rho_mv2u\n")
            for i,(err,r_lam,r_mv,r_mvu) in enumerate(zip(risks, rhos[0], rhos[1], rhos[2])):
                f.write(str(i+1)+";"+str(err)+";"+str(r_lam)+";"+str(r_mv)+";"+str(r_mvu)+"\n")

    if not os.path.exists(outpath) and write_files:
        os.makedirs(outpath)
    RAND = check_random_state(seed)

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


    smodename = 'bagging' if smode == 'bootstrap' else ('reduced bagging (' + str(smode) + ');')
    #print("Starting RFC optimization (m = " + str(m) + ") for [" + dataset + "] using sampling strategy: " + smodename + ", optimizer = " + str(opt))
    results = []
    X,Y = mldata.load(dataset, path=inpath)
    first_rho = None
    for rep in range(reps):
        if reps>1:
            print("####### Repeat = "+str(rep+1))

        rf = EnsembleClassifier(m, ensemble_path=ensemble_path, indices=indices)
        # If x is already a tuple, split it
        if isinstance(X, tuple):
            trainX, testX = X
            trainY, testY = Y
        else:
            trainX, trainY, testX, testY = mldata.split(X, Y, 0.8, random_state=RAND)
        C = np.unique(trainY).shape[0]
        n = (trainX.shape[0], testX.shape[0], trainX.shape[1], C)

        rhos = []
        _  = rf.fit(trainX,trainY)
        _, mv_risk = rf.predict(testX,testY)
        stats  = rf.stats()
        bounds, stats = rf.bounds(stats=stats)
        res_unf = (mv_risk, stats, bounds, -1, -1)

        # Optimize Lambda
        (_, rho, bl) = rf.optimize_rho('Lambda')
        _, mv_risk = rf.predict(testX,testY)
        stats = rf.aggregate_stats(stats)
        bounds, stats = rf.bounds(stats=stats)
        res_lam = (mv_risk, stats, bounds, bl, -1)
        rhos.append(rho)

        # Optimize TND
        (_, rho, bl) = rf.optimize_rho('TND', options={'optimizer':opt})
        _, mv_risk = rf.predict(testX,testY)
        stats = rf.aggregate_stats(stats)
        bounds, stats = rf.bounds(stats=stats)
        res_mv2 = (mv_risk, stats, bounds, bl, -1)
        rhos.append(rho)

        # Optimize DIS if binary
        if(C==2):
            (_, rho, bl, bg) = rf.optimize_rho('DIS', options={'optimizer':opt})
            _, mv_risk = rf.predict(testX,testY)
            stats = rf.aggregate_stats(stats)
            bounds, stats = rf.bounds(stats=stats)
            res_mv2u = (mv_risk, stats, bounds, bl, bg)
            rhos.append(rho)
        else:
            res_mv2u = (-1.0, dict(), dict(), -1, -1)
            rhos.append(-np.ones((m,)))

        # opt = (bound, rho, lam, gam)
        if rep==0:
            first_rho = rhos
            if write_files:
                _write_dist_file(rhos, stats['risks'])
        results.append((rep, n, (res_unf, res_lam, res_mv2, res_mv2u)))

    if write_files:
        _write_outfile(results)

    return first_rho, results

if __name__ == '__main__':
    main()