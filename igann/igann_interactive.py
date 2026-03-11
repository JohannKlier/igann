from igann import IGANN
from igann import ELM_Regressor

# not sure if we need everything here..... clean later!
import time
import torch
import warnings
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class IGANN_interactive(IGANN):
    """
    Extends IGANN to use shape functions directly for predictions during both training
    and inference, enabling customizable and interpretable decision logic.

    Ideas:
    -Shape Function Predictions: Simplifies predictions by relying on shape functions
        instead of the ensemble, reducing complexity and accelerating inference.
    -Interactive Customization: Facilitates real-time adaptation of decision logic
        through modifiable shape functions in json format.
    -Training Flexibility: Supports using shape functions during training, with a potenial benefit of decrasing
        memory usage for enhanced interpretability.
    -IGANN Compatibility:Closely mimics IGANN's behavior, allowing seamless integration.

    Note: This class is experimental and may not be well maintained.
    """

    def __init__(
        self,
        *args,
        GAM_detail=100,
        regressor_limit=100,
        GAMwrapper=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.GAM_detail = GAM_detail
        self.regressor_limit = regressor_limit
        self._compress_after_optimization = True
        self._refit_feature_cols = None
        self._saved_train_indices = None
        self._saved_val_indices = None
        self._saved_split_index_values = None
        self.GAM = GAMmodel(task=self.task, detail=self.GAM_detail)

    def _get_or_create_split_indices(self, X, y_arr):
        """Reuse a saved train/val split for stable iterative refits when possible."""
        current_index_values = np.asarray(X.index)
        can_reuse = (
            self._saved_train_indices is not None
            and self._saved_val_indices is not None
            and self._saved_split_index_values is not None
            and len(self._saved_split_index_values) == len(X)
            and np.array_equal(self._saved_split_index_values, current_index_values)
        )
        if can_reuse:
            return self._saved_train_indices, self._saved_val_indices

        indices = np.arange(len(X))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y_arr if self.task == "classification" else None,
            random_state=self.random_state,
        )
        self._saved_train_indices = train_indices
        self._saved_val_indices = val_indices
        self._saved_split_index_values = current_index_values.copy()
        return train_indices, val_indices

    def fit(self, X, y, val_set=None):
        super().fit(X, y, val_set=val_set)
        if self.GAM is None or not self.GAM.feature_dict:
            self.compress_to_GAM()
        return self

    def _run_optimization(self, X, y, y_hat, X_val, y_val, y_hat_val, best_loss):
        """
        This function runs the optimization for ELMs with single features. This function should not be called from outside.
        Parameters:
        X: the training feature matrix
        y: the training targets
        y_hat: the current prediction for y
        X_val: the validation feature matrix
        y_val: the validation targets
        y_hat_val: the current prediction for y_val
        best_loss: best previous loss achieved. This is to keep track of the overall best sequence of ELMs.
        """

        counter_no_progress = 0
        best_iter = 0

        # Sequentially fit one ELM after the other. Max number is stored in self.n_estimators.
        for counter in range(self.n_estimators):
            #### Start - addtional code for IGANN_interactive #####
            if self._compress_after_optimization and len(self.regressors) > self.regressor_limit:
                print("Reached regressor limit compressing GAM")
                self.compress_to_GAM()
                y_hat = torch.tensor(
                    self.predict_raw(self.raw_X_train), dtype=torch.float32
                )
                y_hat_val = torch.tensor(
                    self.predict_raw(self.raw_X_val), dtype=torch.float32
                )
            #### End - addtional code for IGANN_interactive #####
            hessian_train_sqrt = self._loss_sqrt_hessian(y, y_hat)
            y_tilde = torch.sqrt(torch.tensor(0.5).to(self.device)) * self._get_y_tilde(
                y, y_hat
            )

            # Init ELM
            regressor = ELM_Regressor(
                n_input=X.shape[1],
                n_categorical_cols=self.n_categorical_cols,
                n_hid=self.n_hid,
                seed=counter,
                elm_scale=self.elm_scale,
                elm_alpha=self.elm_alpha,
                act=self.act,
                device=self.device,
            )

            # Fit ELM regressor
            X_hid = regressor.fit(
                X,
                y_tilde,
                torch.sqrt(torch.tensor(0.5).to(self.device))
                * self.boost_rate
                * hessian_train_sqrt[:, None],
            )

            # Make a prediction of the ELM for the update of y_hat
            train_regressor_pred = regressor.predict(X_hid, hidden=True).squeeze()
            val_regressor_pred = regressor.predict(X_val).squeeze()

            self.regressor_predictions.append(train_regressor_pred)

            # Update the prediction for training and validation data
            y_hat += self.boost_rate * train_regressor_pred
            y_hat_val += self.boost_rate * val_regressor_pred

            y_hat = self._clip_p(y_hat)
            y_hat_val = self._clip_p(y_hat_val)

            train_loss = self.criterion(y_hat, y)
            val_loss = self.criterion(y_hat_val, y_val)

            # Keep the ELM, the boosting rate and losses in lists, so
            # we can later use them again.
            self.regressors.append(regressor)
            self.boosting_rates.append(self.boost_rate)
            self.train_losses.append(train_loss.cpu())
            self.val_losses.append(val_loss.cpu())

            # This is the early stopping mechanism. If there was no improvement on the
            # validation set, we increase a counter by 1. If there was an improvement,
            # we set it back to 0.
            counter_no_progress += 1
            if val_loss < best_loss:
                best_iter = counter + 1
                best_loss = val_loss
                counter_no_progress = 0

            if self.verbose >= 1:
                self._print_results(
                    counter,
                    counter_no_progress,
                    self.boost_rate,
                    train_loss,
                    val_loss,
                )

            # Stop training if the counter for early stopping is greater than the parameter we passed.
            if counter_no_progress > self.early_stopping and self.early_stopping > 0:
                break

            if self.verbose >= 2:
                if counter % 5 == 0:
                    self.plot_single()

        if self.early_stopping > 0:
            # We remove the ELMs that did not improve the performance. Most likely best_iter equals self.early_stopping.
            if self.verbose > 0:
                print(f"Cutting at {best_iter}")
            self.regressors = self.regressors[:best_iter]
            self.boosting_rates = self.boosting_rates[:best_iter]

        # Compress the ELMs to the GAM base model at the end of optimization.
        if self._compress_after_optimization:
            self.compress_to_GAM()

        return best_loss

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        #### Start - addtional code for IGANN_interactive #####
        # If GAM shape functions are available, use them as the base predictor.
        if self.GAM is not None and self.GAM.feature_dict:
            # As the GAM uses shape function that are not scaled or one-hot encoded we will >>not<< preprocess the data.
            pred_shape = self.GAM.predict_raw(X)
        else:
            # Fallback before GAM state exists during initial fit.
            pred_shape = (
                self.linear_model.coef_.astype(np.float32) @ X.transpose()
            ).squeeze()
        # GAM.predict_raw already includes intercept; only add it in non-GAM branch.
        if not (self.GAM is not None and self.GAM.feature_dict):
            pred_shape += self.linear_model.intercept_

        # if we have regressors we use them to further imporve the prediction
        if len(self.regressors) > 0:
            X_for_reg = X
            if (
                self._refit_feature_cols is not None
                and isinstance(X, pd.DataFrame)
                and all(col in X.columns for col in self._refit_feature_cols)
            ):
                X_for_reg = X[self._refit_feature_cols].copy()
            X = self._preprocess_feature_matrix(X_for_reg, fit_transform=False).to(self.device)

            pred_nn = torch.zeros(len(X), dtype=torch.float32).to(self.device)
            for boost_rate, regressor in zip(self.boosting_rates, self.regressors):
                pred_nn += boost_rate * regressor.predict(X).squeeze()
            pred_nn = pred_nn.detach().cpu().numpy()
            # convert back to numpy for further calculations
            X = X.detach().cpu().numpy()

        # add pred_nn and pred_shape to get the final prediction if they exits.
        pred_combined = locals().get("pred_shape", 0) + locals().get("pred_nn", 0)

        pred = (
            pred_combined
            # + pred_nn
            # + (self.linear_model.coef_.astype(np.float32) @ X.transpose()).squeeze()
            # + self.linear_model.intercept_
        )
        #### End - addtional code for IGANN_interactive #####

        return pred

    def _get_pred_of_i(self, i, x_values=None):
        # print("get_pred_of_i of igann_interactive")
        # print(self.feature_names)
        feat_name = self.feature_names[i]
        if x_values == None:
            feat_values = self.unique[i]
        else:
            feat_values = x_values[i]

        #### Start - addtional code for IGANN_interactive #####
        # If GAM shape functions are available, use them for partial predictions.
        if self.GAM and self.GAM.feature_dict:
            gam_feature_name, raw_feat_values = self._resolve_gam_feature_input(
                feat_name, feat_values
            )
            pred = self.GAM.predict_single(gam_feature_name, raw_feat_values)
            pred = torch.from_numpy(np.array(pred))
        #### End - addtional code for IGANN_interactive #####

        else:
            if self.task == "classification":
                pred = self.linear_model.coef_[0, i] * feat_values
            else:
                pred = self.linear_model.coef_[i] * feat_values
        # print(pred)

        feat_values = feat_values.to(self.device)
        # print(f"Regressors: {len(self.regressors)}")
        for regressor, boost_rate in zip(self.regressors, self.boosting_rates):
            pred += (
                boost_rate
                * regressor.predict_single(feat_values.reshape(-1, 1), i).squeeze()
            ).cpu()
        return feat_values, pred

    def fit_from_shape_functions(
        self,
        X,
        y,
        feature_dict,
    ):
        """
        Initialize the model from edited GAM shape functions and train new ELM
        regressors on top using a selected subset of features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        fit_cols = [str(c) for c in X.columns]
        if len(fit_cols) == 0:
            raise ValueError("No features available for refit.")
        self._refit_feature_cols = list(fit_cols)

        y_arr_raw = np.asarray(y).reshape(-1)
        train_indices, val_indices = self._get_or_create_split_indices(X, y_arr_raw)

        # Keep full raw X for GAM predictions.
        self.raw_X = X.copy()
        self.raw_X_train = X.iloc[train_indices]
        self.raw_X_val = X.iloc[val_indices]
        if type(y) == pd.Series or type(y) == pd.DataFrame:
            self.raw_y_train = y.iloc[train_indices]
            self.raw_y_val = y.iloc[val_indices]

        self._reset_state()
        X_fit = X[fit_cols].copy()
        X_proc = self._preprocess_feature_matrix(X_fit, fit_transform=True)
        self.X_min = list(X_proc.min(axis=0))
        self.X_max = list(X_proc.max(axis=0))
        self.unique = [torch.unique(X_proc[:, i]) for i in range(X_proc.shape[1])]
        self.hist = [torch.histogram(X_proc[:, i]) for i in range(X_proc.shape[1])]

        y_arr = y_arr_raw
        if self.task == "regression":
            y_arr = self.scale_y(y_arr_raw, fit_transform=True)
        y_torch = torch.from_numpy(y_arr.squeeze()).float()
        if self.task == "classification":
            self.criterion = lambda prediction, target: torch.nn.BCEWithLogitsLoss()(
                prediction, torch.nn.ReLU()(target)
            )
            if torch.min(y_torch) != -1:
                self.target_remapped_flag = True
                y_torch = 2 * y_torch - 1
        elif self.task == "regression":
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Task not implemented. Can be classification or regression")

        if self.GAM is None:
            self.GAM = GAMmodel(task=self.task, detail=self.GAM_detail)
        self.GAM.set_feature_dict(self._normalize_feature_dict_for_gam(feature_dict))
        self.GAM.calibrate_intercept(X, y_arr)

        X_train = X_proc[train_indices]
        X_val = X_proc[val_indices]
        y_train = y_torch[train_indices]
        y_val = y_torch[val_indices]
        y_hat_train = torch.tensor(self.predict_raw(self.raw_X_train), dtype=torch.float32)
        y_hat_val = torch.tensor(self.predict_raw(self.raw_X_val), dtype=torch.float32)
        best_loss = self.criterion(y_hat_val, y_val)

        X_train, y_train, y_hat_train, X_val, y_val, y_hat_val = (
            X_train.to(self.device),
            y_train.to(self.device),
            y_hat_train.to(self.device),
            X_val.to(self.device),
            y_val.to(self.device),
            y_hat_val.to(self.device),
        )

        # Keep edited GAM as base; do not recompress after this optimization.
        prev_flag = self._compress_after_optimization
        self._compress_after_optimization = False
        try:
            self._run_optimization(
                X_train,
                y_train,
                y_hat_train,
                X_val,
                y_val,
                y_hat_val,
                best_loss,
            )
        finally:
            self._compress_after_optimization = prev_flag

        return self

    #### Start - addtional code for IGANN_interactive #####
    def compress_to_GAM(self):
        """
        Compress the model to a GAM model. This is useful if the model is too large and the user wants to make fast predictions.
        """
        print("Compressing to GAM")
        if self.GAM is None:
            self.GAM = GAMmodel(task=self.task, detail=self.GAM_detail)
        self.GAM.set_shape_data(
            self._normalize_feature_dict_for_gam(self.get_shape_functions_as_dict()),
            intercept=self._get_linear_intercept(),
        )

        # ass we now use the shape function for the prediction we do not need the regressors or bossting rates.
        self.regressors = []
        self.boosting_rates = []

    def interact(self):
        from igann import run_igann_interactive

        run_igann_interactive(self)

    def get_feature_wise_pred(self, X):
        if self.GAM is not None:
            y = pd.DataFrame(self.GAM.get_feature_wise_pred(X))
        else:
            print(
                "fist the model should be compressed to a GAM. Try to call 'compress_to_GAM'-function."
            )
        return y

    def get_feature_wise_pred_as_dict(self, X):
        """
        this does not work jet ...
        """
        # y = self.get_feature_wise_pred(X)
        # for
        # for featue in y.Columns():

    def center_shape_functions(self, X=None, update_intercept=True):
        """
        Center GAM feature functions so E[f_j(X_j)] = 0 on reference data X.
        If X is None, use raw_X_train/raw_X from the most recent fit.
        """
        if self.GAM is None:
            self.compress_to_GAM()

        if X is None:
            if hasattr(self, "raw_X_train"):
                X = self.raw_X_train
            elif hasattr(self, "raw_X"):
                X = self.raw_X
            else:
                raise RuntimeError("No reference data available. Provide X explicitly.")

        if self.GAM is None or not self.GAM.feature_dict:
            raise RuntimeError("No GAM shape functions available.")

        feature_means = {}
        intercept_shift_scaled = 0.0

        for feature_name, feature in self.GAM.feature_dict.items():
            if feature_name not in X.columns:
                continue

            contrib_scaled = np.asarray(
                self.GAM.predict_single(feature_name, X[feature_name]),
                dtype=float,
            )
            mu_scaled = float(np.mean(contrib_scaled)) if contrib_scaled.size else 0.0
            feature_means[feature_name] = mu_scaled

            y_vals = np.asarray(feature.get("y", []), dtype=float)
            if y_vals.size == 0:
                continue

            feature["y"] = (y_vals - mu_scaled).tolist()
            intercept_shift_scaled += mu_scaled

        if update_intercept:
            self.GAM.intercept_ += intercept_shift_scaled

        return {
            "feature_means": feature_means,
            "intercept_shift": intercept_shift_scaled,
        }

    def _get_linear_intercept(self):
        if not hasattr(self, "linear_model"):
            return 0.0
        return float(np.asarray(self.linear_model.intercept_).reshape(-1)[0])

    def _get_raw_feature_values_for_gam(self, feature_name, feat_values):
        if hasattr(self, "scaler_dict_") and feature_name in self.scaler_dict_:
            return self.rescale_x(feature_name, feat_values)
        return feat_values

    def _resolve_gam_feature_input(self, feature_name, feat_values):
        if feature_name in self.GAM.feature_dict:
            return feature_name, self._get_raw_feature_values_for_gam(
                feature_name, feat_values
            )

        if "_" in feature_name:
            base_feature_name, class_name = feature_name.rsplit("_", 1)
            if (
                base_feature_name in self.GAM.feature_dict
                and hasattr(self, "dropped_features")
                and base_feature_name in self.dropped_features
            ):
                raw_values = [
                    class_name if value == 1 else self.dropped_features[base_feature_name]
                    for value in feat_values
                ]
                return base_feature_name, raw_values

        raise KeyError(f"Unknown GAM feature: {feature_name}")

    def _normalize_feature_dict_for_gam(self, feature_dict):
        normalized = deepcopy(feature_dict)
        for feature in normalized.values():
            y_values = np.asarray(feature.get("y", []), dtype=float)
            feature["y"] = self.scale_y_per_feature(y_values).tolist()
        return normalized

    def get_gam_feature_dict(self, scaled=False):
        feature_dict = deepcopy(self.GAM.get_feature_dict())
        if scaled:
            return feature_dict
        for feature in feature_dict.values():
            y_values = np.asarray(feature.get("y", []), dtype=float)
            feature["y"] = self.rescale_y_per_feature(y_values).tolist()
        return feature_dict

    #### End - addtional code for IGANN_interactive #####


class GAMmodel:
    """
    This is a wrapper class for the GAM model it handels the alternative functions that are based on the shapefunctions.
    """

    def __init__(self, task, detail=100):
        self.task = task
        self.feature_dict = {}
        self.detail = detail
        self.intercept_ = 0.0

    def get_feature_dict(self):
        return self.feature_dict

    def set_feature_dict(self, feat_dict):
        self.feature_dict = feat_dict
        return

    def set_shape_data(self, shape_data, intercept=0.0):
        self.feature_dict = {}
        self.set_intercept(intercept)
        for feature, feature_dict in shape_data.items():
            feature_type = feature_dict["datatype"]
            feature_x = feature_dict["x"]
            feature_y = feature_dict["y"]
            if feature_type == "categorical":
                feature_x_new = feature_x
                feature_y_new = feature_y
            else:
                feature_x_new, feature_y_new = self.create_points(
                    feature_x, feature_y, self.detail
                )
            self.feature_dict[feature] = {
                "datatype": feature_type,
                "x": feature_x_new,
                "y": feature_y_new,
            }
        return

    def set_intercept(self, intercept):
        self.intercept_ = float(np.asarray(intercept).reshape(-1)[0])
        return

    def update_feature_dict(self, feat_dict):
        self.feature_dict.update(feat_dict)
        return

    def calibrate_intercept(self, X, y_arr):
        base_without_intercept = np.asarray(self.predict_raw(X, include_intercept=False)).reshape(-1)
        if self.task == "classification":
            target_mean = float(np.mean((y_arr >= 0.5).astype(float)))
            target_mean = min(max(target_mean, 1e-4), 1 - 1e-4)
            low, high = -12.0, 12.0
            for _ in range(40):
                mid = (low + high) / 2
                probs = 1 / (1 + np.exp(-(base_without_intercept + mid)))
                if float(np.mean(probs)) < target_mean:
                    low = mid
                else:
                    high = mid
            self.intercept_ = (low + high) / 2
        else:
            self.intercept_ = float(np.mean(y_arr - base_without_intercept))
        return self.intercept_

    def create_points(self, X, Y, num_points):
        """
        this function creates the points for the shape functions that are saved for numeric features.
        """
        min_x, max_x = min(X), max(X)
        x_values = np.linspace(min_x, max_x, num_points)
        artificial_points_X = []
        artificial_points_Y = []

        for x in x_values:
            # Find the indices of the points on either side of x
            idx1 = np.searchsorted(X, x)
            if idx1 == 0:
                y = Y[0]
            elif idx1 == len(X):
                y = Y[-1]
            else:
                x1, y1 = X[idx1 - 1], Y[idx1 - 1]
                x2, y2 = X[idx1], Y[idx1]
                # Compute the weighted average of the y-values of the points on either side
                y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            # Append the artificial point
            artificial_points_X.append(x)
            artificial_points_Y.append(y)

        return artificial_points_X, artificial_points_Y

    def predict_single(self, feature_name, x):
        if feature_name not in self.feature_dict:
            raise KeyError(f"Unknown GAM feature: {feature_name}")
        feature = self.feature_dict[feature_name]
        if feature["datatype"] == "categorical":
            x_classes = feature["x"]
            y_values = feature["y"]
            y = []
            for x in x:
                if str(x) in x_classes:
                    y.append(y_values[x_classes.index(str(x))])
                else:
                    y.append(0)

        else:
            x_values = feature["x"]
            y_values = feature["y"]
            y = np.interp(
                x, x_values, y_values
            )  # Linear interpolation # also strategies for interpolation beyond x limtis can be created here.
        return np.asarray(y, dtype=float)

    def predict_raw(
        self,
        X,
        include_intercept=True,
    ):
        """
        Predict raw values using scaled numerical features and original (raw) categorical features.
        """
        y = {}
        for col in X.columns:
            y[col] = self.predict_single(col, X[col])

        y = pd.DataFrame(y)
        # print(y)

        y = np.array(y.sum(axis=1))
        # if self.base_model.task == "regression" and self.base_model.scale_target:
        #     y = y + self.base_model.y_scaler.mean_

        # y_scaled = self.base_model.scale_y(y, fit_transform=False)

        # print(y_scaled)
        # print(self.base_model.linear_model.intercept_)
        if include_intercept:
            y = y + self.intercept_

        return y

    def get_feature_wise_pred(self, X):
        """
        This function returns the prediction of the GAM model for each feature.
        """
        y = {}
        for col in X.columns:
            y[col] = self.predict_single(col, X[col])

        return y
