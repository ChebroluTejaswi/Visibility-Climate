import preprocess
import file_methods
import pandas as pd


class predict_value:
    def __intit__(self):
        pass
    
    def predict_from_file(self,data):
        """ Preprocessing Data """
        preprocessor= preprocess.calculate()
        data=preprocessor.remove_columns(data,['DATE','Precip','WETBULBTEMPF','DewPointTempF','StationPressure'])
        data=preprocessor.replaceInvalidValueswithNull(data)
        is_null_present=preprocessor.is_null_present(data)
        if(is_null_present):
            data=preprocessor.impute_missing_values(data)
        data_scaled = pd.DataFrame(preprocessor.standardScalingData(data),columns=data.columns)

        """ Finding out clusters """
        file_op = file_methods.File_Operation()
        kmeans=file_op.load_model('KMeans')
        clusters=kmeans.predict(data_scaled)#drops the first column for cluster prediction
        data_scaled['clusters']=clusters
        clusters=data_scaled['clusters'].unique()
        result=[]
        for i in clusters:
            cluster_data = data_scaled[data_scaled['clusters'] == i]
            cluster_data = cluster_data.drop(['clusters'], axis=1)
            model_name = file_op.find_correct_model_file(i)
            model = file_op.load_model(model_name)
            for val in (model.predict(cluster_data.values)):
                result.append(val)
            result = pd.DataFrame(result,columns=['Predictions'])
            output_csv=result.to_csv("templates\Predictions.csv",header=True)
        return output_csv
    