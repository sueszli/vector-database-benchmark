from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from distance import Distance
import time


class Annoy:
    def __init__(self, dataset_name, exclude_list, n_bins):
        self.csv_file = "features/" + dataset_name + "_all_features_normalized.csv"
        self.name_cat = pd.read_csv(self.csv_file, usecols=[0, 1])
        self.exclude_list = exclude_list
        self.n_bins = n_bins
        self.n_features = 5 + 5 * self.n_bins
        self.features = pd.read_csv(self.csv_file, usecols=np.arange(0,self.n_features+2))
        self.index_to_mesh_name = self.build_trees()
        self.mesh_name_to_index = {self.index_to_mesh_name[index]: index for index in self.index_to_mesh_name}
        self.distance = Distance(dataset_name, exclude_list)
        #result = self.query("LabeledDB_new/Octopus/121.off", k=10)


    def build_trees(self):
        index_to_mesh_name = dict()
        t = AnnoyIndex(self.n_features, "euclidean")
        for i in range(len(self.name_cat)):
            index_to_mesh_name[i] = self.name_cat.loc[i, "mesh name"]
            v = self.features.loc[i, ~self.features.columns.isin(["mesh name", "category"])]
            t.add_item(i, v)

        t.build(50)
        t.save("query_trees.ann")
        return index_to_mesh_name


    def query(self, query_mesh_file_path, k=10):
        #start_time = time.perf_counter()
        query_mesh = self.distance.meshify(query_mesh_file_path)
        query_features = self.distance.extract_features_mesh(query_mesh)
        t = AnnoyIndex(self.n_features, 'euclidean')
        t.load("query_trees.ann")
        result = t.get_nns_by_vector(query_features, k, include_distances=True)
        result = [(self.index_to_mesh_name[x], y) for x, y in list(zip(result[0], result[1]))]
        #print("--- % seconds ---" % (time.perf_counter() - start_time))
        return result


    def query_inside_db(self, query_mesh_file_path, k=10):
        #start_time = time.perf_counter()
        t = AnnoyIndex(self.n_features, 'euclidean')
        t.load("query_trees.ann")
        mesh_name = query_mesh_file_path.split("/")[-1]
        result = t.get_nns_by_item(self.mesh_name_to_index[mesh_name], k, include_distances=True)
        result = [(self.index_to_mesh_name[x], y) for x, y in list(zip(result[0], result[1]))]
        #print("--- % seconds ---" % (time.perf_counter() - start_time))
        return result
