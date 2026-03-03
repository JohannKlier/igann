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


class _LinearInitModel:
    """Minimal linear-model stub used when initializing from edited shape functions."""

    def __init__(self, coef, intercept):
        self.coef_ = coef
        self.intercept_ = intercept


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
        self, *args, GAMwrapper=True, GAM_detail=100, regressor_limit=100, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.GAMwrapper = GAMwrapper
        self.GAM = None
        self.GAM_detail = GAM_detail
        self.regressor_limit = regressor_limit
        self.residual_model = None
        self.residual_feature_names = []

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
            if len(self.regressors) > self.regressor_limit:
                print("Reached regressor limit compressing GAM")
                if self.GAMwrapper == True:
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
            mult_coef = (
                torch.sqrt(torch.tensor(0.5).to(self.device))
                * self.boost_rate
                * hessian_train_sqrt[:, None]
            )

            X_hid = regressor.fit(X, y_tilde, mult_coef)

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

        # if we use the GAMwrapper, we compress the ELMs to a GAM model in the end of the optimization
        if self.GAMwrapper == True:
            self.compress_to_GAM()

        return best_loss

    def predict_raw(self, X):
        """
        This function returns a prediction for a given feature matrix X.
        Note: for a classification task, it returns the raw logit values.
        """
        #### Start - addtional code for IGANN_interactive #####
        # if we have a GAM wrapper, we use the GAM model for prediction
        if self.GAMwrapper == True and self.GAM is not None:
            # As the GAM uses shape function that are not scaled or one-hot encoded we will >>not<< preprocess the data.
            pred_shape = self.GAM.predict_raw(X)
            # pred_shape = pred_ + (self.linear_model.intercept_)
        else:
            # if we do not have a GAM wrapper we use the linear shape function for init prediction
            pred_shape = (
                self.linear_model.coef_.astype(np.float32) @ X.transpose()
            ).squeeze()
            # add the intercept when no GAM wrapper handles it already
            pred_shape += self.linear_model.intercept_

        # if we have regressors we use them to further imporve the prediction
        if len(self.regressors) > 0:
            X = self._preprocess_feature_matrix(X, fit_transform=False).to(self.device)

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
        pred += self._predict_residual_raw(X)
        #### End - addtional code for IGANN_interactive #####

        return pred

    def _predict_residual_raw(self, X):
        """Return raw predictions of the unlocked-feature booster, if present."""
        if self.residual_model is None or not self.residual_feature_names:
            return 0
        if not isinstance(X, pd.DataFrame):
            return 0
        X_residual = X[self.residual_feature_names].copy()
        return self.residual_model.predict_raw(X_residual)

    def _initialize_shape_function_state(self, X, y):
        """Fit preprocessing/target scaling state without learning a fresh GAM."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        self.raw_X = X.copy()
        indices = np.arange(len(X))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=y if self.task == "classification" else None,
            random_state=self.random_state,
        )
        self.raw_X_train = X.iloc[train_indices]
        self.raw_X_val = X.iloc[val_indices]
        if type(y) == pd.Series or type(y) == pd.DataFrame:
            self.raw_y_train = y.iloc[train_indices]
            self.raw_y_val = y.iloc[val_indices]

        self._reset_state()
        X_proc = self._preprocess_feature_matrix(X)
        _ = self.scale_y(y, fit_transform=True)

        self.X_min = list(X_proc.min(axis=0))
        self.X_max = list(X_proc.max(axis=0))
        self.unique = [torch.unique(X_proc[:, i]) for i in range(X_proc.shape[1])]
        self.hist = [torch.histogram(X_proc[:, i]) for i in range(X_proc.shape[1])]

        coef_shape = (1, len(self.feature_names)) if self.task == "classification" else len(self.feature_names)
        self.linear_model = _LinearInitModel(
            coef=np.zeros(coef_shape, dtype=np.float32),
            intercept=0.0,
        )

    def _set_base_shape_functions(self, feature_dict):
        """Install edited feature shapes as the current GAM state."""
        self.GAM = GAMmodel(self, self.task, self.GAM_detail)
        self.GAM.set_feature_dict(deepcopy(feature_dict))

    def _get_base_feature_dict(self):
        if self.GAM is None:
            return {}
        return self.GAM.feature_dict

    def _merge_numeric_feature_dicts(self, base_feature, delta_feature):
        base_x = np.asarray(base_feature.get("x", []), dtype=float)
        base_y = np.asarray(base_feature.get("y", []), dtype=float)
        delta_x = np.asarray(delta_feature.get("x", []), dtype=float)
        delta_y = np.asarray(delta_feature.get("y", []), dtype=float)
        if base_x.size == 0:
            return {
                "datatype": "numerical",
                "x": delta_x.tolist(),
                "y": delta_y.tolist(),
            }
        if delta_x.size == 0:
            return deepcopy(base_feature)

        point_count = max(len(base_x), len(delta_x), int(self.GAM_detail))
        target_x = np.linspace(
            min(float(base_x.min()), float(delta_x.min())),
            max(float(base_x.max()), float(delta_x.max())),
            point_count,
        )
        base_interp = np.interp(target_x, base_x, base_y)
        delta_interp = np.interp(target_x, delta_x, delta_y)
        return {
            "datatype": "numerical",
            "x": target_x.tolist(),
            "y": (base_interp + delta_interp).tolist(),
        }

    def _merge_categorical_feature_dicts(self, base_feature, delta_feature):
        categories = []
        for value in list(base_feature.get("x", [])) + list(delta_feature.get("x", [])):
            value = str(value)
            if value not in categories:
                categories.append(value)

        base_map = {
            str(cat): float(base_feature.get("y", [])[i])
            for i, cat in enumerate(base_feature.get("x", []))
            if i < len(base_feature.get("y", []))
        }
        delta_map = {
            str(cat): float(delta_feature.get("y", [])[i])
            for i, cat in enumerate(delta_feature.get("x", []))
            if i < len(delta_feature.get("y", []))
        }
        return {
            "datatype": "categorical",
            "x": categories,
            "y": [base_map.get(cat, 0.0) + delta_map.get(cat, 0.0) for cat in categories],
        }

    def _get_effective_feature_dict(self):
        """Combine edited base GAM shapes with the unlocked residual booster."""
        combined = deepcopy(self._get_base_feature_dict())
        if self.residual_model is None:
            return combined

        delta_shapes = self.residual_model.get_shape_functions_as_dict()
        locked = {str(name) for name in getattr(self, "locked_feature_names", [])}
        for feature_name, delta_feature in delta_shapes.items():
            if feature_name in locked:
                continue

            base_feature = combined.get(feature_name)
            if not base_feature:
                combined[feature_name] = deepcopy(delta_feature)
                continue

            if delta_feature.get("datatype") == "categorical" or base_feature.get("datatype") == "categorical":
                combined[feature_name] = self._merge_categorical_feature_dicts(base_feature, delta_feature)
            else:
                combined[feature_name] = self._merge_numeric_feature_dicts(base_feature, delta_feature)

        return combined

    def fit_from_shape_functions(
        self,
        X,
        y,
        feature_dict,
        locked_features=None,
        refit_estimators=0,
        refit_early_stopping=None,
    ):
        """
        Build an interactive model directly from edited shape functions and learn
        a residual booster only on unlocked features.
        """
        self.locked_feature_names = [str(name) for name in (locked_features or [])]
        self.residual_model = None
        self.residual_feature_names = []

        self._initialize_shape_function_state(X, y)
        self._set_base_shape_functions(feature_dict)

        target_scaled = self.scale_y(y, fit_transform=False)
        if type(target_scaled) == pd.Series or type(target_scaled) == pd.DataFrame:
            target_scaled = target_scaled.values
        target_scaled = np.asarray(target_scaled).reshape(-1)
        base_pred_without_intercept = np.asarray(self.GAM.predict_raw(X)).reshape(-1)
        self.linear_model.intercept_ = float(
            np.mean(target_scaled - base_pred_without_intercept)
        )

        unlocked_features = [
            str(col) for col in X.columns if str(col) not in set(self.locked_feature_names)
        ]
        self.residual_feature_names = unlocked_features
        if refit_estimators <= 0 or len(unlocked_features) == 0:
            return self

        base_prediction = np.asarray(self.predict(X)).reshape(-1)
        residual_target = np.asarray(y).reshape(-1) - base_prediction

        residual_model = IGANN(
            task="regression",
            n_hid=self.n_hid,
            n_estimators=refit_estimators,
            boost_rate=self.boost_rate,
            init_reg=self.init_reg,
            elm_scale=self.elm_scale,
            elm_alpha=self.elm_alpha,
            act=self.act,
            early_stopping=(
                refit_early_stopping if refit_early_stopping is not None else self.early_stopping
            ),
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
            scale_y=self.scale_target,
        )
        residual_model.fit(X[unlocked_features].copy(), residual_target)
        self.residual_model = residual_model
        return self

    def _get_pred_of_i(self, i, x_values=None):
        # print("get_pred_of_i of igann_interactive")
        # print(self.feature_names)
        feat_name = self.feature_names[i]
        if x_values == None:
            feat_values = self.unique[i]
        else:
            feat_values = x_values[i]

        #### Start - addtional code for IGANN_interactive #####
        # if there is a GAMwarapper and its feature dict is set up we use this for a prediction
        if self.GAMwrapper and self.GAM and self.GAM.feature_dict:
            # print(self.scaler_dict_.keys())
            # print("feat_values before")
            # print(feat_values)
            if feat_name in self.scaler_dict_.keys():
                raw_feat_values = self.rescale_x(feat_name, feat_values)
            else:
                raw_feat_values = feat_values
            pred = self.GAM.predict_single(i, raw_feat_values)
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

    #### Start - addtional code for IGANN_interactive #####
    def compress_to_GAM(self):
        """
        Compress the model to a GAM model. This is useful if the model is too large and the user wants to make fast predictions.
        """
        print("Compressing to GAM")
        if self.GAM is None:
            self.GAM = GAMmodel(self, self.task, self.GAM_detail)
        self.GAM.set_shape_functions()

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
        Enforce an empirical centering constraint E[f_j(X_j)] = 0 for all feature
        shape functions in the GAM representation.

        This is optional and not called automatically.
        """
        if self.GAM is None:
            self.compress_to_GAM()

        if X is None:
            if hasattr(self, "raw_X_train"):
                X = self.raw_X_train
            elif hasattr(self, "raw_X"):
                X = self.raw_X
            else:
                raise RuntimeError(
                    "No reference data available. Provide X explicitly."
                )

        return self.GAM.center_feature_dict(X, update_intercept=update_intercept)

    #### End - addtional code for IGANN_interactive #####


class GAMmodel:
    """
    This is a wrapper class for the GAM model it handels the alternative functions that are based on the shapefunctions.
    """

    def __init__(
        self,
        model,
        task,
        detail=100,
    ):
        self.base_model = model
        self.task = task
        self.GAM = None
        self.feature_dict = {}
        self.detail = detail
        # print(self.base_model.feature_names)

    def get_feature_dict(self):
        if hasattr(self.base_model, "_get_effective_feature_dict"):
            return self.base_model._get_effective_feature_dict()
        return self.feature_dict

    def set_feature_dict(self, feat_dict):
        self.feature_dict = feat_dict
        return

    def update_feature_dict(self, feat_dict):
        self.feature_dict.update(feat_dict)
        return

    def set_shape_functions(self):
        """
        This function creates the shape functions for the GAM model.
        it simply call the IGANN function get_shape_functions_as_dict and then creates the shape functions for the GAM model.
        This might looks redundant but could be helpful if we want to use a different model for the shape functions.
        """
        # TODO: Check if we can use the Base IGANN Shapefunction without manipulating it.
        shape_data = self.base_model.get_shape_functions_as_dict()
        for feature, feature_dict in shape_data.items():
            # print(feature_dict["y"])
            feature_name = feature_dict["name"]
            feature_type = feature_dict["datatype"]
            feature_x = feature_dict["x"]
            feature_y = feature_dict["y"]

            # for categorical features we need use one point per class
            if feature_type == "categorical":
                feature_x_new = feature_x
                feature_y_new = feature_y
            else:
                feature_x_new, feature_y_new = self.create_points(
                    feature_x, feature_y, self.detail
                )
            self.feature_dict[feature_name] = {
                "datatype": feature_type,
                "x": feature_x_new,
                "y": feature_y_new,
            }

    def center_feature_dict(self, X, update_intercept=True):
        """
        Center each feature contribution around zero on reference data X by
        subtracting the empirical mean contribution from the feature's shape.
        Optionally add the removed mass back to the model intercept to preserve
        overall predictions.
        """
        if not self.feature_dict:
            raise RuntimeError("feature_dict is empty. Call set_shape_functions() first.")

        feature_means = {}
        intercept_shift = 0.0

        for feature_name, feature in self.feature_dict.items():
            if feature_name not in X.columns:
                continue

            contrib = np.asarray(self.predict_single(feature_name, X[feature_name]))
            mu = float(np.mean(contrib)) if contrib.size else 0.0
            feature_means[feature_name] = mu

            y_vals = np.asarray(feature.get("y", []), dtype=float)
            if y_vals.size == 0:
                continue
            feature["y"] = (y_vals - mu).tolist()
            intercept_shift += mu

        if update_intercept:
            self.base_model.linear_model.intercept_ = (
                self.base_model.linear_model.intercept_ + intercept_shift
            )

        return {
            "feature_means": feature_means,
            "intercept_shift": intercept_shift,
        }

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
        # Some times we use a interger to get the feature name (not sure why)
        if type(feature_name) == int:
            feature_name = self.base_model.feature_names[feature_name]

        # If the feature is not in the feature dict try to handle it like a one-hot encoded one.
        if feature_name not in self.feature_dict.keys():
            # reconstuct original feature name
            new_feature_name = feature_name.rsplit("_", 1)[0]
            # extract new class name
            class_name = feature_name.rsplit("_", 1)[-1]
            # create the new feature
            x = [
                (
                    class_name
                    if x == 1
                    # we fill in the class name of the droped feature which results in y = 0
                    else self.base_model.dropped_features[new_feature_name]
                )
                for x in x
            ]
            feature_name = new_feature_name

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
        # print(len(y))
        y_scaled = self.base_model.scale_y_per_feature(y)
        return y_scaled

    def predict_raw(
        self,
        X,
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
        y_predict_raw = y + self.base_model.linear_model.intercept_

        return y_predict_raw

    def get_feature_wise_pred(self, X):
        """
        This function returns the prediction of the GAM model for each feature.
        """
        y = {}
        for col in X.columns:
            y[col] = self.predict_single(col, X[col])

        return y
