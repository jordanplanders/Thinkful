from chem_ocean import ocean_data as oc_data
from chem_ocean.ocean_plt_util import make_section, subplot_labels

import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn import cluster, metrics
from sklearn.metrics import euclidean_distances, silhouette_samples, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from scipy import stats

import time
from collections import defaultdict

title_sz = 27
axis_sz = 22
tick_sz = 21

# Break water column into statistically different watermasses by regressing
# sections of the slope of at least 60 data points. as long as the r**2 is increasing, 
# expand the intervale by 5 meters. When the r**2 value begins to decline save the depth 
# and start a new section


# kwargs: takes 'exist_plt' if this is a second plot in a fig, and 'depth_lim' in case the second plot is a section with a different minimum depth
def column_split_byslope(_feat_data, _d, **kwargs):
    _feat_data = np.asarray(_feat_data).reshape(-1, 1)
    lower_bound = max(_d)
    lower_increment = 5
    n_1 = 0
    middle_bound = lower_bound - n_1*lower_increment
    X = []
    r2 = 0
    larger_interval= True
    intervals = [lower_bound]
    
    if 'exist_plt' in kwargs:
        (fig, ax) = kwargs['exist_plt']
#         ax = fig.add_subplot(222)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 7), facecolor='w')
    
    while middle_bound >0:
        while len(X)< 60:
            middle_bound = lower_bound - n_1*lower_increment
            X = _feat_data[(_d<=lower_bound) & (_d > middle_bound)]
            y = _d[(_d<=lower_bound) & (_d > middle_bound)]
            n_1+=1

        while larger_interval == True:
            middle_bound = lower_bound - n_1*lower_increment
            X = _feat_data[(_d<=lower_bound) & (_d > middle_bound)]
            y = _d[(_d<=lower_bound) & (_d > middle_bound)]
            n_1+=1

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create linear regression object
            regr = linear_model.LinearRegression()

            # Train the model using the training sets
            regr.fit(X_train, y_train)

            # Make predictions using the testing set
            y_pred = regr.predict(X_test)

            # The coefficients

            # The mean squared error
    #         print("Mean squared error: %.2f"
    #               % mean_squared_error(y_test, y_pred))
            # Explained variance score: 1 is perfect prediction
            if (r2 -r2_score(y_test, y_pred))<0 and middle_bound >0:
                r2 = r2_score(y_test, y_pred)
                larger_interval = True
            else:
                larger_interval = False        

            # Plot outputs
            ax.scatter(X_test, y_test,  color='black')
            ax.plot(X_test, y_pred, color='blue', linewidth=3)
            ax.scatter(X_train, y_train, color='black',alpha= .2)

#         print('Variance score: %.2f' % r2)
#         print('Coefficients: \n', regr.coef_)
#         print('upper bound: ', middle_bound + lower_increment)
        intervals.append(middle_bound + lower_increment)
        lower_bound = middle_bound+ lower_increment
        X = []
        larger_interval=True
        n_1 = 0

    for y_val in intervals:
        ax.axhline(y=y_val, color='r', linestyle='--', alpha = .4)    
    
    ax.set_xlabel(fig.get_axes()[-1].get_ylabel(), fontsize=axis_sz-3, labelpad = 0)
    
    # set tick parameters
    ax.yaxis.tick_right()
    xtickNames = ax.get_xticklabels()
    ytickNames = ax.get_yticklabels()
    
    for names in [ytickNames, xtickNames]:
        plt.setp(names, rotation=0, fontsize=tick_sz-4)
    
    ax.invert_yaxis()
    if 'depth_lim' in kwargs:
        ax.set_ylim(kwargs['depth_lim'])
    return intervals, fig, ax



#Find depths of statistically different water masses by 1 tracer
# Assumes miniumum sample size of 200 of both samples
# starts from the bottom and if the samples between bottom bound and middle bound 
# are not statistically different from the sample between middle bound and upper bound,
# push the middle bound and upper bound up and try again
# once they are stastistically different drop the middle bound incrementally until reach the 
# depth where the p value is just less than .01
# then set bottom bound = middle bound and repeat

# kwargs: takes 'exist_plt' if this is a second plot in a fig, and 'depth_lim' in case the second plot is a section with a different minimum depth
def column_split_ptest(_feat_data2, _d, **kwargs):
    _feat_data = np.asarray(_feat_data2).reshape(-1, 1)

    sample_size = 200
    lower_bound = max(_d)
    lower_increment = 100
    n_1 = 0
    middle_bound = lower_bound - n_1*lower_increment
    upper_increment = 100
    n_2 = 0
    upper_bound = middle_bound - n_2*upper_increment

    intervals = [lower_bound]
    
    if 'exist_plt' in kwargs:
        (fig, ax) = kwargs['exist_plt']
        ax.yaxis.tick_right()
        ax.set_xlabel(fig.get_axes()[-1].get_ylabel(), fontsize=axis_sz-3, labelpad = 0)

#         ax = fig.add_subplot(222)
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 7), facecolor='w')
        ax.set_ylabel('Depth (m)', fontsize=tick_sz-4)
        ax.invert_yaxis()
        
    while upper_bound >0:
        n_1 = 0
        middle_bound = lower_bound - n_1*lower_increment
        n_2 = 0
        upper_bound = middle_bound - n_2*upper_increment

        nitrate_lower = _feat_data[(_d<=lower_bound) & (_d > middle_bound)]
        nitrate_upper = _feat_data[(_d<=middle_bound) & (_d > upper_bound)]
        p2 = 1
    #     print(len(nitrate_upper))
        #split into upper and lower sections with at least 30 data points each
        while p2>.01 or len(nitrate_lower)<sample_size or len(nitrate_upper)<sample_size:
            if len(nitrate_lower) < sample_size or  p2>.01:
                n_1+=.5
                middle_bound = lower_bound - n_1*lower_increment
                nitrate_lower = _feat_data[(_d<=lower_bound) & (_d > middle_bound)]

                nitrate_lower = [nitrate_lower[ik][0] for ik in range(len(nitrate_lower))]

                if middle_bound - n_2*upper_increment != upper_bound:
                    n_2 = .5
                    upper_bound = middle_bound - n_2*upper_increment
                    nitrate_upper = _feat_data[(_d<=middle_bound) & (_d > upper_bound)]
                    while len(nitrate_upper) < sample_size:
                        n_2+=.5
                        upper_bound = middle_bound - n_2*upper_increment
                        nitrate_upper = _feat_data[(_d<=middle_bound) & (_d > upper_bound)]

                        nitrate_upper = [nitrate_upper[ik][0] for ik in range(len(nitrate_upper))]

            # test to see if they are significantly different
            t2, p2 = stats.ttest_ind(nitrate_upper,nitrate_lower)

    #     print(t2,p2)
    #     print(lower_bound, middle_bound, upper_bound, len(nitrate_lower), len(nitrate_upper))
        intervals.append(middle_bound)
        lower_bound = middle_bound
        
    ax.scatter(_feat_data2, _d)
    for y_val in intervals:
        ax.axhline(y=y_val, color='r', linestyle='--', alpha = .4)    
    
    # set tick parameters
    xtickNames = ax.get_xticklabels()
    ytickNames = ax.get_yticklabels()
    
    for names in [ytickNames, xtickNames]:
        plt.setp(names, rotation=0, fontsize=tick_sz-4)
        
    if 'depth_lim' in kwargs:
        ax.set_ylim(kwargs['depth_lim'])
        
    return intervals, fig, ax

#takes a list of models rather than a single model
def test_clustering4(_x, _y, _data, _xLab, _yLab, N_CLUSTERS, _latLon_params, _basemp, models):
    
    _data = np.asarray(_data).reshape(-1, 1)

    pred_dict = {}
    np.random.seed(0)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    plot_num = 1
    X = _data
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    
    # Compute distances
    # create clustering estimators
    alg_list = []
    for model in models:
        
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        
        if model in ['ward', 'agglomerative_clustering']:
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=10)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            if model == 'ward':
                ward = cluster.AgglomerativeClustering(n_clusters=N_CLUSTERS,
                                linkage='ward', connectivity=connectivity)
                alg_list.append(('ward', ward))
            if model == 'agglomerative_clustering':
                average_linkage = cluster.AgglomerativeClustering(linkage="average",
                                        affinity="cityblock", n_clusters=N_CLUSTERS,
                                        connectivity=connectivity)
                alg_list.append(('agglomerative_clustering', average_linkage))
        if model == 'mini_batch_kmeans':
            two_means = cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS)
            alg_list.append(('mini_batch_kmeans', two_means))
        if model == 'spectral_clustering':
            spectral = cluster.SpectralClustering(n_clusters=N_CLUSTERS,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
            alg_list.append(('spectral_clustering', spectral))
        if model == 'dbscan':
            dbscan = cluster.DBSCAN(eps=.2)
            alg_list.append(('dbscan', dbscan))
        if model == 'affinity_propagation':
            affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)
            alg_list.append(('affinity_propagation', affinity_propagation))

    print(alg_list)
    models = {}
    for name, algorithm in alg_list:
        models[name] = {}
        # predict cluster memberships
        models[name]['start'] = time.time()
        algorithm.fit(X)
        models[name]['end'] = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
            models[name]['y_pred']= y_pred
        else:
            y_pred = algorithm.predict(X)
            models[name]['y_pred']= y_pred

        models[name]['sil_score'] = metrics.silhouette_score(X, y_pred, metric='euclidean')
        models[name]['sample_sil_vals'] = silhouette_samples(X, y_pred)
        
        #models[name]['model']= algorithm
        models[name]['N_CLUSTERS']= N_CLUSTERS

    return models