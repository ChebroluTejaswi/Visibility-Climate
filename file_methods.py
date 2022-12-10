import pickle
import os
import shutil


class File_Operation:
    def __init__(self):
        self.model_directory='models/'

    def save_model(self,model,filename):
        try:
            path = os.path.join(self.model_directory,filename) #create seperate directory for each cluster
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path) #
            with open(path +'/' + filename+'.sav','wb') as f:
                pickle.dump(model, f) # save the model to file
            return 'success'
        except Exception as e:
            print("Exception occured in saving model")
            raise Exception()

    def load_model(self,filename):
        try:
            with open(self.model_directory + filename + '/' + filename + '.sav','rb') as f:
                return pickle.load(f)
        except Exception as e:
            print("Exception occured in loading model")
            raise Exception()

    def find_correct_model_file(self,cluster_number):
        try:
            cluster_number= cluster_number
            folder_name=self.model_directory
            list_of_model_files = []
            list_of_files = os.listdir(folder_name)
            for file in list_of_files:
                try:
                    if (file.index(str(cluster_number))!=-1):
                        model_name=file
                except:
                    continue
            model_name=model_name.split('.')[0]
            return model_name
        except Exception as e:
            print("Exception occured in finding correct model file")
            raise Exception()