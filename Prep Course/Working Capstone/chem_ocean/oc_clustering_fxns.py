# from chem_ocean.oc_db_fxns import min_max_print
import numpy as np
from sklearn import cluster, datasets
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


def build_cluster_data(oc_np_db, var_names,flt_, depth_target, d2):
    sum_names = ['Station', 'Longitude', 'Latitude']+var_names
    mini_test_arr = []
    for ik in xrange(len(flt_[:,0])):
        station = flt_[ik][0]
        try:
            ind = np.where(oc_np_db[station][d2['Depth']] > depth_target)[0]-1
            #if len(ind)>0:
            #print ind[0], oc_np_db[station][d2['Depth']][ind[0]]
            lst = [station, oc_np_db[station][d2['Longitude']], oc_np_db[station][d2['Latitude']]]+[oc_np_db[station][d2[_var]][ind[0]] for _var in var_names]     
            mini_test_arr.append(np.array(lst))
        except:
            continue
            
    mini_test_arr = np.array(mini_test_arr)    
    # min_max_print(sum_names,mini_test_arr)
    return mini_test_arr


def test_clustering(data, lons, lats, N_CLUSTERS):
    pred_dict = {}
    np.random.seed(0)

    n_samples = 1500

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1
    for i_dataset, dataset in enumerate([data]):
        X = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # Compute distances
        #distances = np.exp(-euclidean_distances(X))
        distances = euclidean_distances(X)

        # create clustering estimators
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS)
        ward = cluster.AgglomerativeClustering(n_clusters=N_CLUSTERS,
                        linkage='ward', connectivity=connectivity)
        spectral = cluster.SpectralClustering(n_clusters=N_CLUSTERS,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=.2)
        affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                           preference=-200)

        average_linkage = cluster.AgglomerativeClustering(linkage="average",
                                affinity="cityblock", n_clusters=N_CLUSTERS,
                                connectivity=connectivity)

        for name, algorithm in [
                                ('MiniBatchKMeans', two_means),
                                ('AffinityPropagation', affinity_propagation),
                                ('MeanShift', ms),
                                ('SpectralClustering', spectral),
                                ('Ward', ward),
                                ('AgglomerativeClustering', average_linkage),
                                ('DBSCAN', dbscan)
                               ]:
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            # plot
            plt.subplot(4, 7, plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            plt.scatter(lons, lats, color=colors[y_pred].tolist(), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                try:
                    centers = algorithm.cluster_centers_
                    center_colors = colors[:len(centers)]
                    plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
                except:
                    continue
            plt.xlim(min(lons)-1, max(lons)+1)
            plt.ylim(min(lats)-1, max(lats)+1)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1
            pred_dict[name] = [lons, lats, colors[y_pred].tolist()]

    plt.show()
    return pred_dict    

