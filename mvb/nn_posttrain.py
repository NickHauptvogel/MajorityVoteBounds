#
# Implements RandomForestClassifier in the framework
#
import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.utils import check_random_state
import os
import json
import pickle

from . import util, mvbase

class NeuralNetworkPostTrain:

    def __init__(self, ensemble_path: str, model_file: str):
        model_name = model_file.replace('.h5', '')
        self.ensemble_path = ensemble_path
        # Find file that ends with config.json
        config_file = [f.path for f in os.scandir(ensemble_path) if f.name.endswith('config.json')][0]
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        # Find file that ends with scores.json
        scores_file = [f.path for f in os.scandir(ensemble_path) if f.name.endswith('scores.json')][0]
        with open(scores_file, 'r') as f:
            self.scores = json.load(f)
        # Find file that ends with predictions.pkl
        test_predictions_file = model_name+'_test_predictions.pkl'
        with open(test_predictions_file, 'rb') as f:
            self.test_predictions = pickle.load(f)
            if self.test_predictions.shape[1] == 1:
                # Add the complementary class
                self.test_predictions = np.concatenate((1-self.test_predictions, self.test_predictions), axis=1)

        val_predictions_file = model_name+'_val_holdout_predictions.pkl'
        self.val_indices_name = 'holdout_indices'
        self.use_holdout = True
        if not os.path.isfile(val_predictions_file):
            val_predictions_file = model_name+'_val_predictions.pkl'
            self.val_indices_name = 'val_indices'
            self.use_holdout = False
        with open(val_predictions_file, 'rb') as f:
            self.val_predictions = pickle.load(f)
            if self.val_predictions.shape[1] == 1:
                # Add the complementary class
                self.val_predictions = np.concatenate((1-self.val_predictions, self.val_predictions), axis=1)


    def predict(self, X):
        if X.shape[0] == self.test_predictions.shape[0]:
            return self.test_predictions
        else:
            print("ERROR: UNEXPECTED SHAPE")
            return None


class NeuralNetworkPostTrainClassifier(mvbase.MVBounds):
    def __init__(
            self,
            max_estimators: int,
            ensemble_path: str,
            indices,
            ):

        estimators = []

        # List all directories in ensemble_path
        ensemble_dirs = sorted([f.path for f in os.scandir(ensemble_path) if f.is_dir()])

        for dir in ensemble_dirs:
            model_files = sorted([f.path for f in os.scandir(dir) if f.name.endswith('.h5')])
            for model_file in model_files:
                # Load the model
                model = NeuralNetworkPostTrain(dir, model_file)
                # Add the model to the estimators
                estimators.append(model)

        if indices is not None:
            if len(indices) != max_estimators:
                raise ValueError(f"Number of indices ({len(indices)}) does not match max_estimators ({max_estimators})")
            estimators = [estimators[i] for i in indices]
        else:
            estimators = estimators[:max_estimators]

        # Make sure to warn the user
        if any([not est.use_holdout for est in estimators]):
            print("USING VALIDATION SET. MAKE SURE IT WAS NOT USED DURING TRAINING")

        super().__init__(estimators, sample_mode='DUMMY', random_state=1)

    def fit(self, X, Y):

        self._classes = np.unique(Y)
        self._rho = util.uniform_distribution(len(self._estimators))

        preds = []
        for est in self._estimators:
            M_est, P_est = np.zeros(Y.shape), np.zeros(Y.shape)
            oob_idx = est.config[est.val_indices_name]
            M_est[oob_idx] = 1
            P_est[oob_idx] = np.argmax(est.val_predictions, axis=1)


            # Save predictions on oob and validation set for later
            util.oob_risks([(M_est, P_est)], Y)
            preds.append((M_est, P_est))

        self._OOB = (preds, Y)

        return

    def predict(self, X, Y=None):
        P = self.predict_all(X)
        P_maj_vote = np.argmax(P, axis=2)
        mvP_maj_vote = util.mv_preds(self._rho, P_maj_vote)

        # Get to format (test_samples, models, classes)
        P = P.transpose(1, 0, 2)
        # Get prediction with softmax averaging
        subset_y_pred_ensemble = np.average(P, axis=1, weights=self._rho)
        mvP_softmax_avg = np.argmax(subset_y_pred_ensemble, axis=1)
        return ((mvP_maj_vote, mvP_softmax_avg), (util.risk(mvP_maj_vote, Y), util.risk(mvP_softmax_avg, Y))) if Y is not None else (mvP_maj_vote, mvP_softmax_avg)
