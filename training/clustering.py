import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import file_methods

class KMeansClustering:
    def __init__(self):
        pass

    def elbow_plot(self,data):
        wcss=[]
        try:
            for i in range (1,11):
                kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42) 
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)
            plt.plot(range(1,11),wcss)
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.savefig('K-Means_Elbow.PNG') # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            return kn.knee

        except Exception as e:
            print("Exception occured in elbow_plot method")
            raise Exception()

    def create_clusters(self,data,number_of_clusters):
        try:
            kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=42)
            y_kmeans=kmeans.fit_predict(data) #  divide data into clusters

            file_op = file_methods.File_Operation()
            save_model = file_op.save_model(kmeans, 'KMeans') # saving the KMeans model to directory 
            # passing 'Model' as the functions need three parameters
            data['Cluster']=y_kmeans  # create a new column in dataset for storing the cluster information
            return data
        except Exception as e:
            print("Exception occured in create cluster method")
            raise Exception()