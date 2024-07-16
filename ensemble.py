import numpy as np
import model
from common import EnsembleResult, Model, State
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

__PER_CLUSTER = 10

def __extract_alphas(single_trajectory_results):
    alphas = []
    for fov in single_trajectory_results:
        for traj_ind in fov:
            for segment in fov[traj_ind]:
                if segment.length > 6:
                    alphas.append(segment.alpha) 
    return alphas

def __extract_ds(single_trajectory_results):
    ds = []
    for fov in single_trajectory_results:
        for traj_ind in fov:
            for segment in fov[traj_ind]:
                if segment.length > 6:
                    ds.append(segment.K)
    return ds


def get_optim_clustering(data_points):
    max_cluster_num = 2 + int(len(data_points) / __PER_CLUSTER)
    best_n_clusters = 1
    sil_score_max = -1
    for n_clusters in range(2, max_cluster_num):
        model = KMeans(n_clusters = n_clusters)
        labels = model.fit_predict(data_points)
        sil_score = silhouette_score(data_points, labels)
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    model = KMeans(n_clusters = best_n_clusters)
    return model.fit_predict(data_points), best_n_clusters

def __split_data_by_cluster(data, labels, cluster_num):
    clusters = [[]  for _ in range(cluster_num)]
    for i, label in enumerate(labels):
        clusters[label].append(data[i])
    return clusters

def __get_basic_characteristics(values):
    return np.mean(values), np.var(values)


def __get_characteristics_of_cluster(cluster):
    alphas = [cluster[i][0] for i in range(len(cluster))]
    ds = [cluster[i][1] for i in range(len(cluster))]
    alphas_mean, alphas_vars = __get_basic_characteristics(alphas)
    ds_mean, ds_vars = __get_basic_characteristics(ds)
    return alphas_mean, alphas_vars, ds_mean, ds_vars

def __get_weights(segments, labels, cluster_num):
    weights = [0 for _ in range(cluster_num)]
    for i, label in enumerate(labels):
        weights[label] = weights[label] + segments[i].length
    return weights

def __get_ensemble_result_from_clusters(estim_points, segments, labels, cluster_num, model):
    clusters = __split_data_by_cluster(estim_points, labels, cluster_num)
    alpha_means = []
    alpha_vars = []
    d_means = []
    d_vars = []
    for i in range(len(clusters)):
        a_m, a_v, d_m, d_v = __get_characteristics_of_cluster(clusters[i])
        alpha_means.append(a_m)
        alpha_vars.append(a_v)
        d_means.append(d_m)
        d_vars.append(d_v)
    weights = __get_weights(segments, labels, cluster_num)
    return EnsembleResult(model, alpha_means, alpha_vars, d_means, d_vars, weights)
    
def __get_number_of_trajectories(results):
    acc = 0
    for fov in results:
        acc += len(fov)
    return acc

def __get_restricted_states_count(segments):
    immobile_counter = 0
    confinement_counter = 0
    for i in range(len(segments)):
        if segments[i].state == State.Immobile:
            immobile_counter += 1
        elif segments[i].state == State.Confined:
            confinement_counter += 1
    return immobile_counter, confinement_counter

def __create_list_with_all_segments(results):
    segments = []
    for fov in results:
        for traj_ind in fov:
            for segment in fov[traj_ind]:
                segments.append(segment) 
    return segments


def naive_aggregation(single_trajectory_results):
    # TDOD: This ensmble really needs to be replaced
    alphas = __extract_alphas(single_trajectory_results)
    alphas_mean, alphas_var = __get_basic_characteristics(alphas)
    ds = __extract_ds(single_trajectory_results)
    ds_mean, ds_var = __get_basic_characteristics(ds)
    return EnsembleResult(Model.SINGLE_STATE, [alphas_mean], [alphas_var], [ds_mean], [ds_var], [1.0])


def __get_model_by_vote(classifier, raw_trajectoriers):
    count_dict= {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    probabilities = classifier.predict(model.prepare_input(raw_trajectoriers))
    preds = model.get_preds(probabilities)
    for pred in preds:
        count_dict[pred] = count_dict[pred] + 1
    max_count = 0
    max_id = 0
    print("States: ", count_dict)
    for state in count_dict:
        if count_dict[state] > max_count:
            max_id = state
            max_count = count_dict[state]
    return  Model.get_model_by_id(max_id)


def simple_aggregation(results, classifier, raw_trajectoriers):
    # TODO: This method also should be upgraded/replaced in the future
    all_segments = __create_list_with_all_segments(results)
    estimates = [ (all_segments[i].alpha, all_segments[i].K) for i in range(len(all_segments))  ]
    labels = None
    num_of_states = None
    predicted_model = __get_model_by_vote(classifier, raw_trajectoriers)
    if predicted_model in (Model.CONFINEMENT, Model.IMMOBILE_TRAPS, Model.DIMERIZATION):
        num_of_states = 2
        clusterer = KMeans(n_clusters = 2)
        labels = clusterer.fit_predict(estimates)
    elif predicted_model == Model.SINGLE_STATE:
        return naive_aggregation(results)
    else:
        # It is multistate
        labels, num_of_states = get_optim_clustering(estimates)
    return __get_ensemble_result_from_clusters(estimates, all_segments, labels, num_of_states, predicted_model)