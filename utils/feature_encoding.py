from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

AGE_MAPPING = {
        '80+': 82.5, 
        '75-79': 77,
        '70-74': 72,
        '65-69': 67,
        '60-64': 62,
        '55-59': 57,
        '50-54': 52,
        '45-49': 46.5,
        '40-44': 42,
        '35-39': 37,
        '30-34': 32,
        '25-29': 27,
        '18-24': 21,
    }


BINS_DICT = {
    "MentalHealthDays": [0, 3, 7, 14, 29, float("inf")],
    "PhysicalHealthDays": [0, 3, 7, 14, 29, float("inf")]
}

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freqs_ = {}
        for col in X.columns:
            freqs = X[col].value_counts(normalize=True)
            self.freqs_[col] = freqs
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].map(self.freqs_[col]).fillna(0)
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        return input_features
    
class NamedFunctionTransformer(FunctionTransformer):
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class CustomBinner(BaseEstimator, TransformerMixin):
    def __init__(self, bins_dict=None, features=None, n_bins=4):
        self.bins_dict = bins_dict if bins_dict else {}
        self.features = features if features else []
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()

        for feature in self.features:
            missing_mask = X[feature].isnull()

            if feature in self.bins_dict:
                bins = self.bins_dict[feature]
                X_transformed.loc[~missing_mask, feature] = np.digitize(X.loc[~missing_mask, feature], bins=bins, right=False)
            else:
                est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
                X_transformed.loc[~missing_mask, feature] = est.fit_transform(X.loc[~missing_mask, [feature]]).flatten()

        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
                return self.features
    
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        self.mappings = {key:value for key, value in self.mappings.items() if key in X.columns.to_list()}
        self.inverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in self.mappings.items()}

        return self 

    def transform(self, X):
        X_transformed = X.copy()
        for col, mapping in self.mappings.items():
            X_transformed[col] = X_transformed[col].map(mapping)
        return X_transformed

    def inverse_transform(self, X):
        X_inverse = X.copy()
        for col, inverse_mapping in self.inverse_mappings.items():
            X_inverse[col] = X_inverse[col].map(inverse_mapping)
        return X_inverse
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else list(self.mappings.keys())

# converts the output NumPy array (of preprocessing) into a dataframe, assigning it the feature names
class DFConverter(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_names = self.transformer.get_feature_names_out()
        df = pd.DataFrame(data=X, columns=features_names)
        # converts features names to json format
        df.columns = df.columns.str.replace(r'[^\w]+', '_', regex=True)
        return df


def create_preprocessor(**kwargs):
    kwargs.setdefault("bins_num", 4)
    kwargs.setdefault("remainder", False)

    # Num features transformations
    log_transformer = NamedFunctionTransformer(np.log, inverse_func=np.exp)
    continous_pipe = Pipeline([
        ("log_transform", log_transformer),
        ("standartization", StandardScaler())
    ])
    


    binning_transformer = CustomBinner(bins_dict=BINS_DICT, 
                                       features=kwargs["binning_features"], 
                                       n_bins=kwargs["bins_num"])



    num_transformer = ColumnTransformer([
        ("continuous", continous_pipe, kwargs["continuous_features"]),
        ("binned", binning_transformer, kwargs["binning_features"]),
    ], remainder="passthrough")

    # Categorical features transformations

    mappings = {
        "GeneralHealth": {
            'Poor': 0,
            'Fair': 1, 
            'Good': 2, 
            'Very good': 3,  
            'Excellent': 4,
        },
        "AgeCategory": AGE_MAPPING, # created on the start of bivariate analysis in num section,
        "RemovedTeeth": {
            'None of them': 0,
            '1 to 5': 1,
            '6 or more, but not all': 2,
            'All': 3
        }
    }
    remainder = kwargs["remainder"]
    if remainder == "ordinal":
        remainder_enc = OrdinalEncoder()
    elif remainder == "frequency":
        remainder_enc = FrequencyEncoder()
    else:
        remainder_enc = "passthrough"

    custom_ordinal = CustomOrdinalEncoder(mappings)
    num_features =  kwargs["continuous_features"] +  kwargs["binning_features"]
    
    preprocessing = ColumnTransformer([
        ("ordinal_enc", custom_ordinal,  kwargs["ordinal_features"]),
        ("onehot_enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False),  kwargs["onehot_features"]),
        ("num", num_transformer, num_features)
    ], remainder=remainder_enc)

    preprocessing_pipe = Pipeline(
    [
        ("preprocessing", preprocessing),
        ("to_dataframe", DFConverter(preprocessing))
    ])
    
    return preprocessing_pipe