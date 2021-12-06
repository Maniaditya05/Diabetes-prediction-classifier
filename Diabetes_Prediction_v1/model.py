import pandas
import sklearn
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plot
import seaborn
from sklearn import linear_model
from sklearn.svm import SVC 
from sklearn.metrics import mean_squared_error, r2_score, recall_score, precision_score

class Model:

    def __init__(self, dataset_file_path, dataset_filenames):
        self.dataset_filenames = dataset_filenames
        self.dataset_file_path = dataset_file_path 
    

    def read_dataset(self):
        dataframes = []

        for dataset in self.dataset_filenames:
            dataframe = pandas.read_csv(self.dataset_file_path + '/' + dataset)
            dataframes.append(dataframe)

        return dataframes
    
    def merge_and_select_attribute(self, dataframes_read):
        dataframes = dataframes_read
        
        for dataframe in dataframes[2:]:
            dataframe.drop(['SEQN'], axis=1, inplace=True)
  
        dataframe = pandas.concat(dataframes, axis=1, join='inner')

        return dataframe

    def imputate_dataframe(self, dataframe):
        if dataframe.isnull().values.any():
            dataframe.dropna(axis=1, how="all")
            dataframe.dropna(axis=0, how="all") 

            return dataframe
        else:
            return dataframe
    
    def feature_selection(self, dataframe):
        merged_dataframe = dataframe 

        columns_to_rename = {
            'SEQN': 'ID',
            'RIAGENDR': 'Gender',
            'DMDYRSUS': 'Years_in_US',
            'INDFMPIR': 'Family_income',
            'LBXGH': 'GlycoHaemoglobin',
            'BMXARMC': 'ArmCircum',
            'BMDAVSAD': 'SaggitalAbdominal',
            'MGDCGSZ': 'GripStrength',
            'DRABF': 'Breast_fed'
        }

        merged_dataframe = merged_dataframe.rename(columns=columns_to_rename)

        merged_dataframe = merged_dataframe.loc[:, ['ID', 'Gender', 'Years_in_US', 'Family_income', 'GlycoHaemoglobin', 'ArmCircum', 'SaggitalAbdominal', 'GripStrength', 'Breast_fed']]

        return merged_dataframe
    
    def ValuesManipulations(self, dataframe):
        if dataframe.isnull().values.any():  
            dataframe['Years_in_US'] = dataframe['Years_in_US'].apply(lambda x: x if x > 0 else 0)
            dataframe['GlycoHaemoglobin'] = dataframe['GlycoHaemoglobin'].fillna( dataframe['GlycoHaemoglobin'].median())
            dataframe['SaggitalAbdominal'] = dataframe['SaggitalAbdominal'].fillna(dataframe['SaggitalAbdominal'].median())
            dataframe['ArmCircum'] = dataframe['ArmCircum'].fillna(dataframe['ArmCircum'].median())
            dataframe['GripStrength'] = dataframe['GripStrength'].fillna(dataframe['GripStrength'].median())
            dataframe['Family_income'] = dataframe['Family_income'].fillna(method='ffill')
            dataframe['Breast_fed'] = dataframe['Breast_fed'].fillna(value=1)

            set_threshold_value = VarianceThreshold(threshold=(.8 * (1 - .8)))
            set_threshold_value.fit_transform(dataframe)

            dataframe.loc[dataframe['GlycoHaemoglobin'] < 6.0, 'Diabetes'] = 0
            dataframe.loc[(dataframe['GlycoHaemoglobin'] >= 6.0) & (dataframe['GlycoHaemoglobin'] <= 6.4), 'Diabetes'] = 1
            dataframe.loc[dataframe['GlycoHaemoglobin'] >= 6.5, 'Diabetes'] = 2

            dataframe.head(10)

            return dataframe
        else:
            return dataframe
    
    def labels_plot_visualize(self, dataframe):
        colormap = plot.cm.viridis

        plot.figure(figsize=(14, 14))
        seaborn.heatmap(dataframe.astype(float).drop(axis=1, labels='ID').corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True)
        
        diabetes_diagram_representation = seaborn.pairplot(dataframe.drop(['ID', 'GlycoHaemoglobin'], axis=1), hue='Diabetes', height=1.5, diag_kind='kde')
        diabetes_diagram_representation.set(xticklabels=[])

        return plot

    def drop_off_haemoglobin_column(self, dataframe):
        dataframe.drop(['GlycoHaemoglobin'], axis=1, inplace=True)
        dataframe.head(5)

        return dataframe

    def dataset_split(self, dataframe):
        data = dataframe.drop(['Diabetes'], axis=1)
        target = dataframe[['Diabetes']]

        #seperate training set and test set
        x_train = data[:6000]
        x_test = data[6000:]

        y_train = target[:6000]
        y_test = target[6000:]

        return [x_train, x_test, y_train, y_test]
    
    def diabetes_prediction_model(self, dataframe): 
        logistic_regression = linear_model.LinearRegression()

        # train our data on linear regression classifier 
        logistic_regression.fit(self.dataset_split(dataframe)[0], self.dataset_split(dataframe)[2])

        model = logistic_regression

        return model

    def predict_diabetes(self, model, dataframe):  
        prediction = model.predict(self.dataset_split(dataframe)[1])

        logistic_regression_score = model.score(self.dataset_split(dataframe)[1], self.dataset_split(dataframe)[3])
        
        return [prediction, logistic_regression_score]
        
    def visualize_model_performance(self, model, dataframe):
        print('Logistic Regression Coefficients: ', model.coef_)
        print('Logistic Regression Mean Square Error: %.2f' % mean_squared_error(self.dataset_split(dataframe)[3], self.predict_diabetes(model, dataframe)[0]))
        print('Logistic Regression Variance score: %.2f' % r2_score(self.dataset_split(dataframe)[3], self.predict_diabetes(model, dataframe)[0]))
        print('Logistic Regression Score: %.2f' % self.predict_diabetes(model, dataframe)[1])
        