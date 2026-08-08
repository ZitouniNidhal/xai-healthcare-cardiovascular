import numpy as np
from lime import lime_tabular

class LIMEExplainerWrapper:
    """LIME explainer wrapper for explanation of structured patient record predictions."""
    def __init__(self, training_data, feature_names, class_names=None):
        """
        Args:
            training_data (np.ndarray): Training input features.
            feature_names (list): List of feature string names.
            class_names (list): Class names list.
        """
        self.class_names = class_names if class_names else ["No CVD", "CVD"]
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(training_data),
            feature_names=feature_names,
            class_names=self.class_names,
            mode="classification",
            random_state=42
        )

    def explain_instance(self, instance, predict_fn, num_features=5):
        """
        Generate local explanation weights for a single tabular instance.
        
        Args:
            instance (np.ndarray / pd.Series): Sample features.
            predict_fn (callable): Predict probability function returning probability arrays.
            num_features (int): Top important features to include in explanation.
            
        Returns:
            list: List of tuples (feature_name, weight)
        """
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features
        )
        return exp.as_list()
