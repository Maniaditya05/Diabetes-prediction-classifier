import matplotlib.pyplot as plot
from model import Model

# DATASET csv file path
csv_file_path = './public/csv' 
labs = 'labs.csv'
examination = 'examination.csv'
demographic = 'demographic.csv'
diet = 'diet.csv'
questionnaire = 'questionnaire.csv'

csv_file_names = [labs,
                 examination,
                 demographic,
                 diet,
                 questionnaire]

# print dataframe
DiabetesModel = Model(csv_file_path, csv_file_names)


# print pandas dataframe 
dataframe = DiabetesModel.read_dataset()
print(dataframe)

merged_dataframe = DiabetesModel.merge_and_select_attribute(dataframe)
print(merged_dataframe.describe())

# imputation of our dataframe
imputed_dataframe = DiabetesModel.imputate_dataframe(merged_dataframe)
print(imputed_dataframe.describe())

# feature selection
dataframe_done_feature_selection = DiabetesModel.feature_selection(imputed_dataframe)
print(dataframe_done_feature_selection.describe())

dataframe_done_values_manipulations = DiabetesModel.ValuesManipulations(dataframe_done_feature_selection)
print(dataframe_done_values_manipulations.describe())

LabelVisualize = DiabetesModel.labels_plot_visualize(dataframe_done_values_manipulations)

dataframe_dropped_off_haemoglobin = DiabetesModel.drop_off_haemoglobin_column(dataframe_done_values_manipulations)
print(dataframe_dropped_off_haemoglobin)

model = DiabetesModel.diabetes_prediction_model(dataframe_dropped_off_haemoglobin)

predictions = DiabetesModel.predict_diabetes(model, dataframe_done_values_manipulations)

if (predictions):
    visalizations = DiabetesModel.visualize_model_performance(
        model, dataframe_dropped_off_haemoglobin)


LabelVisualize.show()


