import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTENC, SMOTEN 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer

class Preprocessor:
    def __init__(self):
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self.sampler = None
        self.discretizer = None


        
    def load_data(self, path):
        """
        Load the dataset
        """
        return pd.read_csv(path)
        
    
    def impute(self, df_h: pd.DataFrame, imputation_strategy='mean'):
        """
        Impute the missing values
        Possible strategies:
        1. mean
        2. median
        3. mode
        4. drop
        5. knn
        6. interpolate
        8. mode_mean
        9. mode_median
        10. mode_mode
        11. mode_knn
        12. mode_interpolate
        13. mode_iterative
        """
        if imputation_strategy == 'None':
            return df_h
        elif imputation_strategy == 'mean':
            for col in df_h.columns:
                if df_h[col].dtype != 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mean())
        elif imputation_strategy == 'median':
            for col in df_h.columns:
                if df_h[col].dtype != 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].median())
        elif imputation_strategy == 'mode':
            for col in df_h.columns:
                if df_h[col].dtype != 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0]) 
        elif imputation_strategy == 'drop':
            df_h = df_h.dropna()
        elif imputation_strategy == 'knn':
            # if self.imputer is None:
            categorical_cols = df_h.select_dtypes(include='object').columns
            numerical_cols = df_h.select_dtypes(include=np.number).columns

            self.imputer = KNNImputer(n_neighbors=2)
            # df_h = pd.DataFrame(self.imputer.fit_transform(df_h),columns = df_h.columns)
            # df_h = pd.DataFrame(self.imputer.fit_transform(df_h[numerical_cols]),columns = df_h[numerical_cols].columns)
            df_h[numerical_cols] = pd.DataFrame(self.imputer.fit_transform(df_h[numerical_cols]),columns = df_h[numerical_cols].columns)

        elif imputation_strategy == 'interpolate':
            df_h = df_h.interpolate(method='linear', axis=0).ffill().bfill()
        # mixed (category and numeric) data
        elif imputation_strategy == 'mode_mean':    
            for col in df_h.columns:
                if df_h[col].dtype == 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0])
                else:
                    df_h[col] = df_h[col].fillna(df_h[col].mean())
                    
        elif imputation_strategy == 'mode_mode':    
            for col in df_h.columns:
                if df_h[col].dtype == 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0]) 
                else:
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0])

        elif imputation_strategy == 'mode_median':
            for col in df_h.columns:
                if df_h[col].dtype == 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0])
                else:
                    df_h[col] = df_h[col].fillna(df_h[col].median())
        elif imputation_strategy == 'mode_mode':
            for col in df_h.columns:
                if df_h[col].dtype == 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0])
                else:
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0])
        elif imputation_strategy == 'mode_knn':
            # mode for categorical, knn for numeric

            numeric_cols = df_h.select_dtypes(include=np.number).columns
            categorical_cols = df_h.select_dtypes(include='object').columns
            df_h[categorical_cols] = df_h[categorical_cols].fillna(df_h[numeric_cols].mode().iloc[0])
            # if self.imputer is None:
            self.imputer = KNNImputer(n_neighbors=2)
            df_h[numeric_cols] = pd.DataFrame(self.imputer.fit_transform(df_h[numeric_cols]),columns = df_h[numeric_cols].columns)
            # for col in df_h.columns:
            #     if df_h[col].dtype == 'object':
            #         df_h[col] = df_h[col].fillna(df_h[col].mode()[0])
            #     else:
            #         # if self.imputer is None:
            #         self.imputer = KNNImputer(n_neighbors=2)
            #         df_h[col] = pd.DataFrame(self.imputer.fit_transform(df_h[col]),columns = df_h[col].columns)
        elif imputation_strategy == 'mode_interpolate':
            for col in df_h.columns:
                if df_h[col].dtype == 'object':
                    df_h[col] = df_h[col].fillna(df_h[col].mode()[0])
                else:
                    df_h[col] = df_h[col].interpolate(method='linear', axis=0).ffill().bfill()
        
        return df_h

    def imputation_types(self):
        return ['mean', 
                    'median', 
                    'mode', 
                    'drop', 
                    'knn', 
                    'interpolate', 
                    'mode_mean', 
                    'mode_median', 
                    'mode_knn', 
                    'mode_interpolate', 
                    'mode_mode',
                    'None']

    
    def encode(self, df_h: pd.DataFrame, encoding_strategy='onehot'):
        """
        Encode the categorical variables
        Possible strategies:
        1. onehot
        2. ordinal

        """
        if encoding_strategy == 'None':
            return df_h
        elif encoding_strategy == 'onehot':
            categorical_cols = df_h.select_dtypes(include='object').columns
            categorical_cols = categorical_cols.drop('RainTomorrow')
            numerical_cols = df_h.select_dtypes(include=np.number).columns
            df_h2 = pd.get_dummies(df_h, columns=categorical_cols)
            df_h = self._resetRainTomorrow(df_h2)
        elif encoding_strategy == 'ordinal':
            for col in df_h.columns:
                df_h[col] = df_h[col].astype('category')
                df_h[col] = df_h[col].cat.codes
        return df_h
    
    def encoding_types(self):
        return ['onehot', 'ordinal', 'None']


    def scale(self, df_h: pd.DataFrame, scaling_strategy='standard'):
        """
        Scale the dataset
        Possible strategies:
        1. standard
        2. minmax
        3. robust
        """
        # if self.scaler is not None:
        #     df_h = pd.DataFrame(self.scaler.fit_transform(df_h))
        #     return df_h
        
        if scaling_strategy == 'None':
            return df_h
        elif scaling_strategy == 'standard':
            self.scaler = StandardScaler()
            df_h = pd.DataFrame(self.scaler.fit_transform(df_h))
        elif scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
            df_h = pd.DataFrame(self.scaler.fit_transform(df_h))
        elif scaling_strategy == 'robust':
            self.scaler = RobustScaler()
            df_h = pd.DataFrame(self.scaler.fit_transform(df_h))
        return df_h
    
    def scaleX(self, X: np.ndarray, scaling_strategy='standard'):
        """
        Scale the dataset
        Possible strategies:
        1. standard
        2. minmax
        3. robust
        """
        if scaling_strategy == 'None':
            return X
        # if self.scaler is not None:
        #     X = self.scaler.fit_transform(X)
        #     return X
        if scaling_strategy == 'standard':
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif scaling_strategy == 'minmax':
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
        elif scaling_strategy == 'robust':
            self.scaler = RobustScaler()
            X = self.scaler.fit_transform(X)
        return X
    def scaling_types(self):
        return ['standard', 'minmax', 'robust', 'None']

    def sample(self, df_h: pd.DataFrame, sampling_strategy='smote'):
        """
        Sample the dataset
        X: all columns except the last one
        y: last column
        Possible strategies:
        1. smote
        2. random
        3. smotenc
        4. smoten
        5. under
        """

        if sampling_strategy == 'None':
            return df_h
        
        if sampling_strategy == 'smote':
            X = df_h.iloc[:, :-1]
            y = df_h.iloc[:, -1]
            self.sampler = SMOTE(random_state=0)

            X_smote, y_smote = self.sampler.fit_resample(X, y)
            df_h = pd.concat([X_smote, y_smote], axis=1)

        elif sampling_strategy == 'random':
            X = df_h.iloc[:, :-1]
            y = df_h.iloc[:, -1]
            self.sampler = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            df_h = pd.concat([X_resampled, y_resampled], axis=1)

        elif sampling_strategy == 'smotenc':
            X = df_h.iloc[:, :-1]
            y = df_h.iloc[:, -1]
            self.sampler = SMOTENC(categorical_features=[0, 2], random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            df_h = pd.concat([X_resampled, y_resampled], axis=1)

        elif sampling_strategy == 'smoten':
            X = df_h.iloc[:, :-1]
            y = df_h.iloc[:, -1]
            self.sampler = SMOTEN(random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            df_h = pd.concat([X_resampled, y_resampled], axis=1)

        elif sampling_strategy == 'under':
            X = df_h.iloc[:, :-1]
            y = df_h.iloc[:, -1]
            self.sampler = RandomUnderSampler(random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            df_h = pd.concat([X_resampled, y_resampled], axis=1)

        return df_h
    
    def sampling_types(self):
        return ['smote', 'random', 'smotenc', 'smoten', 'under', 'None']
    

    def sampleXy(self, X:np.ndarray, y:np.ndarray, sampling_strategy='smote'):
        """
        Sample the dataset
        X: all columns except the last one
        y: last column
        Possible strategies:
        1. smote
        """
        if sampling_strategy == 'None':
            return X, y
        if sampling_strategy == 'smote':
            self.sampler = SMOTE(random_state=0)
            X_smote, y_smote = self.sampler.fit_resample(X, y)
            return X_smote, y_smote
        elif sampling_strategy == 'random':
            self.sampler = RandomOverSampler(random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        elif sampling_strategy == 'smotenc':
            self.sampler = SMOTENC(categorical_features=[0, 2], random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        elif sampling_strategy == 'smoten':
            self.sampler = SMOTEN(random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        elif sampling_strategy == 'under':
            self.sampler = RandomUnderSampler(random_state=0)
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            return X_resampled, y_resampled
        
        return X, y
    

    def discretize(self, df_h: pd.DataFrame, discretize_strategy='equal_width'):
        """
        discretize the dataset (only for numeric data) For Naive Bayes
        Possible strategies:
        1. equal_width
        2. equal_freq
        """
        if discretize_strategy == 'None':
            return df_h
        elif discretize_strategy == 'equal_width':
            self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            df_h = pd.DataFrame(self.discretizer.fit_transform(df_h))
        elif discretize_strategy == 'equal_freq':
            self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            df_h = pd.DataFrame(self.discretizer.fit_transform(df_h))
        return df_h
    
    def discretizeX(self, X: np.ndarray, discretize_strategy='equal_width'):
        """
        discretize the dataset (only for numeric data) For Naive Bayes
        Possible strategies:
        1. equal_width
        2. equal_freq
        """
        if discretize_strategy == 'None':
            return X
        elif discretize_strategy == 'equal_width':
            self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
            X = self.discretizer.fit_transform(X)
        elif discretize_strategy == 'equal_freq':
            self.discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            X = self.discretizer.fit_transform(X)
        return X
    
    def discretize_types(self):
        return ['equal_width', 'equal_freq', 'None']
        
    def outlier(self, df_h: pd.DataFrame, replace_strategy='mean'):
        """
        Outlier detection
        Possible strategies:
        1. mean
        2. median
        3. mode
        4. drop
        """
        if replace_strategy == 'None':
            return df_h
        for col in df_h.columns:
            if df_h[col].dtype != 'object':
                q1 = df_h[col].quantile(0.25)
                q3 = df_h[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 -(1.5 * iqr)
                upper_bound = q3 +(1.5 * iqr)
                if replace_strategy == 'drop':
                    df_h = df_h[df_h[col] < upper_bound]
                    df_h = df_h[df_h[col] > lower_bound]
                elif replace_strategy == 'mean':
                    cond = (df_h[col] > upper_bound) | (df_h[col] < lower_bound)
                    df_h.loc[cond, col] = df_h[col].mean()

                elif replace_strategy == 'median':
                    cond = (df_h[col] > upper_bound) | (df_h[col] < lower_bound)
                    df_h.loc[cond, col] = df_h[col].median()
                elif replace_strategy == 'mode':
                    cond = (df_h[col] > upper_bound) | (df_h[col] < lower_bound)
                    df_h.loc[cond, col] = df_h[col].mode()[0]

        return df_h
    
    def outlier_types(self):
        return ['mean', 'median', 'mode', 'drop', 'None']


    def handleDate(self, df_h: pd.DataFrame, date_col):
        """
        Handle date column
        """
        df_h[date_col] = pd.to_datetime(df_h[date_col])
        df_h['year'] = df_h[date_col].dt.year
        df_h['month'] = df_h[date_col].dt.month
        df_h['day'] = df_h[date_col].dt.day
        df_h = df_h.drop(date_col, axis=1)
        df_h = self._resetRainTomorrow(df_h)
        

        return df_h
    def _resetRainTomorrow(self, df_h: pd.DataFrame):
        """
        Move RainTomorrow to the last column
        """
        # drop RainTomorrow, then add it back
        df_h2 = df_h.copy()
        df_h = df_h.drop('RainTomorrow', axis=1)  
        df_h['RainTomorrow'] = df_h2['RainTomorrow']
        return df_h
    

    def _preprocess(self, df_h, date_col = 'Date', imputation_strategy='mode_mean', encoding_strategy='ordinal', scaling_strategy='standard', sampling_strategy='None', discretize_strategy='None', outlier_strategy='median'):
        """
        Preprocess the dataset
        """
        df_h = self.handleDate(df_h, date_col)
        if imputation_strategy is not None:
            df_h = self.impute(df_h, imputation_strategy)
        if outlier_strategy is not None:
            df_h = self.outlier(df_h, outlier_strategy)
        if encoding_strategy is not None:
            df_h = self.encode(df_h, encoding_strategy)
        X_train, X_test, y_train, y_test = self.split(df_h)
        if sampling_strategy is not None:
            X_train, y_train = self.sampleXy(X_train, y_train, sampling_strategy)
        if scaling_strategy is not None:
            X_train = self.scaleX(X_train, scaling_strategy)
            if self.scaler is not None:
                X_test = self.scaler.transform(X_test)
        if discretize_strategy is not None:
            X_train = self.discretizeX(X_train, discretize_strategy)
            if self.discretizer is not None:
                X_test = self.discretizer.transform(X_test)

        return X_train, X_test, y_train, y_test
        # if scaling_strategy is not None:
        #     df_h = self.scale(df_h, scaling_strategy)
        # print('scaling ==========')
        # print('imputation =======')
        # if discretize_strategy is not None:
        #     df_h = self.discretize(df_h, discretize_strategy)
        # print('discretization ===')
        # if sampling_strategy is not None:
        #     df_h = self.sample(df_h, sampling_strategy)            
        # print('sampling =========')
        # return df_h
    

    def split(self, df_p: pd.DataFrame, test_size=0.2, random_state=42):
        """
        Split the dataset into train and test
        """
        return train_test_split(df_p.drop('RainTomorrow', 
                                          axis=1), 
                                          df_p['RainTomorrow'], 
                                          test_size=test_size, 
                                          random_state=random_state)
        # X = df_h.iloc[:, :-1]
        # y = df_h.iloc[:, -1]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # return X_train, X_test, y_train, y_test