# license = Apache License, Version 2.0

class Sample_data(object):
    def __init__(self, data, decreasing_ratio_threshold=0.01):
        self.data = data
        self.scaler = self.scale(data)
        self.pca_table, self.pca_var_num = self.pca_diag(self.scaler, decreasing_ratio_threshold)
        self.pca_array, self.pca_model = self.pca_forecast(self.pca_var_num, self.scaler)
        
    def scale(self, data):
        from sklearn import preprocessing
        std_scaler = preprocessing.MaxAbsScaler()
        return std_scaler.fit_transform(data)

    def pca_diag(self, input, decreasing_ratio_threshold):
        from sklearn.decomposition import PCA
        import pandas as pd
        pca_rate = []
        for n in range(1, input.shape[1]):
            pca_model = PCA(n_components=n).fit(input)
            exp_ratio_sum = sum(pca_model.explained_variance_ratio_)
            pca_rate.append(exp_ratio_sum)
        pca_res_data = pd.DataFrame({'component_no':range(1, input.shape[1]), 'explain_ratio':pca_rate})
        pca_res_data['ratio_diff'] = pca_res_data.explain_ratio.diff()
        pca_cat_num = pca_res_data[pca_res_data.ratio_diff>=decreasing_ratio_threshold].component_no.max()
        return pca_res_data, pca_cat_num

    def pca_forecast(self, input_num, input_data):
        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=input_num).fit(input_data)
        return pca_model.transform(input_data), pca_model


class Sample_label:
    def __init__(self, data):
        self.data = data.reshape((len(data), 1))
        self.stats = self.show_stats(data)

    def show_stats(self, data):
        import pandas as pd
        return pd.Series(data).value_counts().append(pd.Series({'total':len(data), 'minority_percentage(%)':round(pd.Series(data).value_counts().min()/len(data)*100, 3)}))


class cluster_process:
    def __init__(self, sample_data, cl_upper_num=50, accum_wcss_drop_ratio_threshold=0.01):
        import pandas as pd
        self.cluster_num, self.kms_table = self.clusting_diag(sample_data.pca_array, cl_upper_num, accum_wcss_drop_ratio_threshold)
        self.cluster_label, self.kms_model = self.clusting_forecast(sample_data.pca_array, self.cluster_num)
        self.cluster_stats = pd.DataFrame(pd.Series(self.cluster_label.flatten()).value_counts().sort_index()).reset_index().rename(columns={'index':'cluster', 0:'counts'})
        self.cluster_stats['percentage(%)'] = round(self.cluster_stats.counts/self.cluster_stats.counts.sum()*100, 2)

    def clusting_diag(self, input, cl_upper_num, accum_wcss_drop_ratio_threshold):
        from joblib import Parallel, delayed
        import pandas as pd
        def kmeans_mini_process(n):
            from sklearn.cluster import MiniBatchKMeans
            kmean_mini = MiniBatchKMeans(init='k-means++', n_clusters=n, batch_size=256, verbose=0, random_state=7).fit(input)
            wcss_i = kmean_mini.inertia_
            return n, wcss_i
        def key_fun(l):
            return(kmeans_mini_process(l))
        results = Parallel(n_jobs=-1)(delayed(key_fun)(l) for l in range(2, cl_upper_num))
        res_data = pd.DataFrame({'cluster':[i[0] for i in results], 'wcss':[i[1] for i in results]})
        res_data['wcss_diff'] = res_data.wcss.diff()
        res_data['wcss_drop_rate'] = res_data.wcss_diff / res_data.wcss.median() 
        res_data['wcss_drate_2rolling_sum'] = res_data.wcss_drop_rate.rolling(2).sum()
        us_cat_num =  res_data[res_data.wcss_drate_2rolling_sum >= -accum_wcss_drop_ratio_threshold].cluster.min() - 1
        return us_cat_num, res_data
    
    def clusting_forecast(self, input, us_cat_num):
        from sklearn.cluster import MiniBatchKMeans
        kmean_mini = MiniBatchKMeans(init='k-means++', n_clusters=us_cat_num, batch_size=256, verbose=0, random_state=7).fit(input)
        return kmean_mini.labels_.reshape((len(kmean_mini.labels_), 1)), kmean_mini


class under_sampling:
    def __init__(self, sample_data, sample_label, cluster_label, under_sampling_multiplier=1, random_multiplier = 0.05, n_neighbors=None):
        import numpy as np
        import pandas as pd
        self.array_merge = np.append(sample_data.pca_array, cluster_label, axis=1)
        self.array_merge = np.append(self.array_merge, sample_label.data, axis=1)
        cluster_tab = pd.DataFrame(self.array_merge[:, -2:], columns=['cat', 'label'])
        self.cat_stat_tab = pd.DataFrame(cluster_tab.groupby(['cat'])['label'].value_counts()).rename(columns={'label':'counts'})
        filter_s = cluster_tab.groupby('cat')['label'].sum()
        self.tar_cat_list = filter_s[filter_s>0].index.tolist()
        self.free_cat_list = filter_s[filter_s==0].index.tolist()
        arr_us_res = np.empty((0, self.array_merge.shape[1]-1))
        self.array_usampled, self.usampling_info_df = self.under_sampling_process(self.array_merge, self.tar_cat_list, under_sampling_multiplier, n_neighbors)
        self.array_random_usampled, self.random_usampling_info_df = self.random_usampling_process(self.array_merge, self.free_cat_list, random_multiplier)
        self.array_under_sampled_final = np.append(self.array_usampled, self.array_random_usampled, axis=0)
        final_label = self.array_under_sampled_final[:,-1]
        sample_num = len(final_label)
        minority_sample_num = final_label.sum()
        majority_sample_num = sample_num - minority_sample_num
        self.stats_info = 'Sample number: ' + str(sample_num) + '\n' + 'Minority sample number: ' + str(minority_sample_num) + '\n' + 'Majority sample number: ' + str(majority_sample_num) + '\n' + 'Neg-Pos sample ratio: ' + str(round(minority_sample_num/majority_sample_num*100, 3)) + '%'

    def under_sampling_process(self, array_merge, tar_cat_list, under_sampling_multiplier, n_neighbors):
        from imblearn.under_sampling import CondensedNearestNeighbour
        import numpy as np
        import pandas as pd
        def is_in_notebook():
            import sys
            return 'ipykernel' in sys.modules

        def clear_output():
            """
            clear output for both jupyter notebook and the console
            """
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            if is_in_notebook():
                from IPython.display import clear_output as clear
                clear()
        
        arr_us_res = np.empty((0,array_merge.shape[1]-1))
        info_box = pd.DataFrame(columns=['cluster no.', 'Majority counts', 'U-sampled Majority counts', 'Minority counts'], dtype=int)
        for cat in tar_cat_list:
            cat_arr = array_merge[array_merge[:,-2]==cat][:, :-2]
            cat_label = array_merge[array_merge[:,-2]==cat][:, -1]
            s1 = cat_label.sum()
            s0 = len(cat_label) - s1
            seed_num = int(s1*under_sampling_multiplier)
            if s0 > seed_num:
                enn = CondensedNearestNeighbour(sampling_strategy='majority', 
                                                n_neighbors = n_neighbors,
                                                n_seeds_S = seed_num,
                                                n_jobs=-1
                                             )
                x_res, y_res = enn.fit_resample(cat_arr, cat_label.astype(int))
            else:
                x_res = cat_arr
                y_res = cat_label
            info_box = info_box.append({'cluster no.': int(cat), 'Majority counts': int(len(cat_label)-cat_label.sum()), 'U-sampled Majority counts': int(len(y_res)-y_res.sum()), 'Minority counts': int(y_res.sum())}, ignore_index=True)
            x_res = np.append(x_res, y_res.reshape(len(y_res), 1), axis = 1)
            arr_us_res = np.append(arr_us_res, x_res, axis = 0)
            clear_output()
            display(info_box)
        return arr_us_res, info_box

    def random_usampling_process(self, array_merge, free_cat_list, random_multiplier):
        import numpy as np
        import pandas as pd
        arr_rd_res = np.empty((0,array_merge.shape[1]-1))
        info_box = pd.DataFrame(columns=['cluster no.', 'Sample counts', 'Random U-sampled counts'], dtype=int)
        for cat in free_cat_list:
            cat_arr = array_merge[array_merge[:,-2]==cat]
            cat_arr = np.delete(cat_arr, -2, axis=1)
            # seed_num = max(int(len(cat_arr[:, -2])*random_multiplier), 1)
            seed_num = np.ceil(cat_arr.shape[0]*random_multiplier)
            np.random.shuffle(cat_arr)
            arr_rd_res = np.append(arr_rd_res, cat_arr[0:int(seed_num)], axis=0)
            info_box = info_box.append({'cluster no.': int(cat), 'Sample counts': int(cat_arr.shape[0]), 'Random U-sampled counts': seed_num}, ignore_index=True)
        return arr_rd_res, info_box

    def final_stats(self):
        print(self.stats_info)

# This product is licensed under the Apache License 2.0
# [http://www.apache.org/licenses/LICENSE-2.0.html]
