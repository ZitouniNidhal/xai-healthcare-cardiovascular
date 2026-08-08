import os
import pickle
import xgboost as xgb
import shap

class XGBoostModelWrapper:
    """XGBoost model handler containing training, saving, and basic SHAP explainability hook."""
    def __init__(self, config):
        self.config = config
        xgb_params = config.get("xgboost")
        self.model = xgb.XGBClassifier(
            n_estimators=xgb_params.get("n_estimators", 200),
            max_depth=xgb_params.get("max_depth", 5),
            learning_rate=xgb_params.get("learning_rate", 0.05),
            subsample=xgb_params.get("subsample", 0.8),
            colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
            random_state=xgb_params.get("random_state", 42),
            eval_metric="logloss"
        )
        self.explainer = None

    def train(self, X_train, y_train):
        """Train the classifier."""
        self.model.fit(X_train, y_train)
        # Initialize SHAP explainer right after training
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, X):
        """Predict labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_shap_values(self, X):
        """Calculate SHAP values for tabular samples."""
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        return self.explainer(X)

    def save(self, file_path=None):
        """Serialize the trained model to a pickle file."""
        if file_path is None:
            file_path = self.config.get_path("models")["xgboost_path"]
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
            
    def load(self, file_path=None):
        """Load serialized model."""
        if file_path is None:
            file_path = self.config.get_path("models")["xgboost_path"]
            
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
        self.explainer = shap.TreeExplainer(self.model)
