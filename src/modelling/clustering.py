from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)


import os
from itertools import cycle
from functools import partial
from copy import deepcopy
from collections import (
    defaultdict,
    Counter
)
import random


import numpy as np
from sklearn.cluster import KMeans
import networkx as nx


class SameSizeKMeans(object):
    '''SameSize K-Means clustering algorithm

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    init_model: KMeans object, default: None
        The initial KMeans model to fit on. Leaving as None
        defaults to KMeans with default parameters except for
        passing the as-specified n_clusters.

    save_labels: bool, default: False
        Whether to save labels at each step of the fitting
        process.

    metric: str, default: 'l2'
        Specifies the distance metric to use after the initial
        KMeans clustering algorithm is run. Only 'l2' (Euclidean
        distance) is currently guaranteed to work.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Final coordinates of cluster centers

    final_labels: array, [n_points, 1]
        Final labels of each point

    all_labels_: None or list
        Labels of each point at each step of the fitting
        process. None unless save_labels is set to True.

    '''

    LOOP_BUFFER = 1
    ORDER_DICT = {
        'largest_first': 'l',
        'smallest_first': 's',
        'l': 'l',
        's': 's',
        'min_v': 'min_v',
        'max_v': 'max_v'
    }

    def __init__(self, n_clusters=8, init_model = None,
                 save_labels=True, metric='l2'):
        self.n_clusters = n_clusters
        if init_model is None:
            self.init_model = KMeans(n_clusters=n_clusters)
        else:
            self.init_model = init_model
        self.init_params = self.init_model.get_params()
        self.save_labels = save_labels
        if save_labels:
            self.all_labels_ = []
        else:
            self.all_labels_ = None
        self.metric = metric

    def fit(self, X, weights=None, weight_tol=0, order='s'):
        ''' Fit the SSKMeans model, populating final_labels

        Parameters
        ----------
        X: array, [n_points, 2]
            Coordinates of each

        weights: float, default: 1e-4
            Fractional tolerance of the weight

        weight_tol : float, default: 1e-4
            Fractional tolerance of the weight

        order: str or list of strings, default: 's'
            The order in which to adjust clusters. Options are:
                - "smallest_first" or "s": Adjusts the smallest
                    clusters first
                - "largest_first" or "l": Adjusts the largest
                    clusters first
                - "min_v": Adjusts the cluster closest to the
                    optimal size first
                - "max_v": Adjusts the cluster farthest from
                    the optimal size first
            Alternatively, you may pass a list of any combination
            of these options. The list must be as long as n_clusters

        Returns
        -------
        None

        Populates the following model attributes:
            final_labels
            all_labels_ (if save_labels == True)
        '''

        self._save_fit_params(X, weights, weight_tol, order)

        # Get temporary labels from a naive KMeans clustering
        _temp_labels = self._fit_naive_KMeans(X)

        for step in range(self.n_clusters - self.LOOP_BUFFER):

            print('Starting step ',step+1)

            # Determine the cluster to update on this step, along
            # with its associated info
            _coords, _weights, _label, _score_arr, _centroid = (
                self._get_cluster_info(order, step)
            )

            # Adjust the determined cluster, and return its finalized
            # associated coordinates
            _coords = self._adjust_cluster(
                _coords, _weights, _label, _score_arr, order, step
            )

            # Set _coords as having finalized labels, and set their labels
            for coord in _coords:
                _mask = np.isclose(self.X, coord).all(1)
                self._labels_finalized[_mask] = True
                self.final_labels[_mask] = _label

            # Remove _label from the list of unfinalized clusters
            self._clusters_unfinalized = [
                label for label in self._clusters_unfinalized
                if label != _label
            ]

            # Update centroids
            self._update_centroids()

            if self.save_labels:
                _temp_labels = deepcopy(self.final_labels)
                self.all_labels_.append(deepcopy(_temp_labels))

    def _save_fit_params(self, X, weights, weight_tol, order):
        '''Saves the fit parameters during fit()

        Parameters
        ----------
        X : array, shape = [n_samples, 2]

        weights: array, shape = (n_samples,)

        weight_tol: float

        For descriptions, see fit()

        Returns
        -------
        None
        '''

        self.X = X
        self.weight_tol = weight_tol
        self.order = order

        if weights is None:
            self.weights = np.ones(X.shape[0])
        else:
            try:
                assert(weights.shape[0] == X.shape[0])
                self.weights = weights
            except AssertionError:
                raise AssertionError('X and weights are not the same length')

        self._ideal_cluster_weight = np.sum(weights)/self.n_clusters

        return None

    def _fit_naive_KMeans(self, X):
        '''Fits the initial KMeans model

        Parameters
        ----------
        X : array, shape = [n_samples, 2]
            See fit()

        Returns
        -------
        _temp_labels: array, shape = [n_samples,]
            The inital KMeans labels assigned to each point
        '''

        # Setup naive KMeans model
        _temp_model = KMeans(**self.init_params)

        # Get labels from KMeans model, and save them if desired
        _temp_labels = _temp_model.fit_predict(X)
        self._unique_labels = np.unique(_temp_labels)
        if self.save_labels:
            self.all_labels_.append(deepcopy(_temp_labels))

        # Create a dictionary of cluster centers by label
        self.cluster_centers_ = {
            label: np.mean(X[_temp_labels == label], axis=0)
            for label in self._unique_labels
        }

        # Save the current labels as the final labels. To be
        # updated as the program progresses.
        self.final_labels = deepcopy(_temp_labels)

        # Numpy array specifying whether the point is finalized
        # according to its assignment. Updated accordingly later.
        self._labels_finalized = np.array([False for coords in X])

        # List of unfinalized clusters
        self._clusters_unfinalized = [label for label in self._unique_labels]

        return _temp_labels

    def _get_cluster_info(self, order, step):
        '''Convenience function getting properties of the chosen cluster

        Parameters
        ----------
        order: str or list
            See fit()

        step: int
            The current iteration of the SSKMeans adjustment.

        Returns
        -------
        _coords

        _weights

        _label

        _score_arr

        _centroid
        '''

        _order = self._get_order(order, step)

        # Find the coordinates, weights, and label of the cluster
        # determined by order.
        _coords, _weights, _label = self._find_cluster(_order)
        _centroid = self.cluster_centers_[_label]

        if _order == 'l':
            _score_arr = None
        elif _order == 's':
            _score_arr = self._score_other_points(
                _coords, _centroid
            )
        elif _order == 'min_v':
            if np.sum(_weights) > self._ideal_cluster_weight:
                _score_arr = None
            else:
                _score_arr = self._score_other_points(
                    _coords, _centroid
                )
        elif _order == 'max_v':
            if np.sum(_weights) > self._ideal_cluster_weight:
                _score_arr = None
            else:
                _score_arr = self._score_other_points(
                    _coords, _centroid
                )

        return(_coords, _weights, _label, _score_arr, _centroid)

    def _get_order(self, order, step):
        '''Convenience function for setting the clustering order
        '''

        try:
            _order = self.ORDER_DICT[order]
        except KeyError:
            raise KeyError(
                'order must be one of: {}'.format(
                    self.ORDER_DICT.keys())
            )
        except TypeError:
            try:
                _order = self.ORDER_DICT[order[step]]
            except KeyError:
                raise KeyError(
                    'order must be one of: {}'.format(
                        self.ORDER_DICT.keys())
                )
            except TypeError:
                raise TypeError('Order must be a string or list of strings.')

        return _order

    def _find_cluster(self, order):
        '''Finds the unfinalized cluster according to self.order
        '''

        # Find the coordinates, weights, and current assigned
        # label of each point which doesn't have a finalized
        # cluster
        _X = self.X[
            np.logical_not(self._labels_finalized)
        ]
        _temp_weights = self.weights[
            np.logical_not(self._labels_finalized)
        ]
        _labels = self.final_labels[
            np.logical_not(self._labels_finalized)
        ]

        # Get the weights of the clusters that aren't finalized
        # if order is 'l' or 's'
        if (order == 'l' or order == 's'):
            _label_weights = {
                label: np.sum(_temp_weights[_labels == label])
                for label in self._clusters_unfinalized
            }
        # Get the distance of weights of the clusters from the
        # ideal cluster weight for the clusters that aren't
        # finalized if order is 'min_v' or 'max_v'
        elif (order == 'min_v' or order == 'max_v'):
            _label_weights = {
                label: abs(
                    np.sum(_temp_weights[_labels == label])
                    - self._ideal_cluster_weight
                )
                for label in self._clusters_unfinalized
            }

        if order == 'l':
            # Find the label of the largest unfinalized cluster
            _cluster = max(_label_weights, key=_label_weights.get)
        elif order == 's':
            # Find the label of the smallest unfinalized cluster
            _cluster = min(_label_weights, key=_label_weights.get)
        elif order == 'min_v':
            # Find the label of the cluster closest to the optimal size
            _cluster = min(_label_weights, key=_label_weights.get)
        elif order == 'max_v':
            # Find the label of the cluster farthest from the optimal size
            _cluster = max(_label_weights, key=_label_weights.get)

        _coords = _X[_labels == _cluster]
        _weights = _temp_weights[_labels == _cluster]

        return(_coords, _weights, _cluster)

    def _adjust_cluster(self, coords, weights, label,
                        score_arr, order, step):
        '''Performs the adjustment of the given cluster
        '''

        _order = self._get_order(order, step)

        # Check if the cluster has the right total weight by comparing it
        # to the ideal cluster weight and the specified weight tolerance.
        # If it doesn't, find the point farthest from the center of mass,
        # and reassign it. Repeat until satisfied.
        while (
                abs(np.sum(weights) - self._ideal_cluster_weight)
                / self._ideal_cluster_weight > self.weight_tol):

            if _order == 'l':
                coords, weights = self._reassign_farthest(
                    coords, weights, label)
            elif _order == 's':
                coords, weights, score_arr = (
                    self._reassign_closest(
                        coords, weights, label, score_arr)
                )
            elif _order == 'min_v':
                if np.sum(weights) > self._ideal_cluster_weight:
                    coords, weights = self._reassign_farthest(
                        coords, weights, label)
                else:
                    coords, weights, score_arr = (
                        self._reassign_closest(
                            coords, weights, label, score_arr)
                    )
            elif _order == 'max_v':
                if np.sum(weights) > self._ideal_cluster_weight:
                    coords, weights = self._reassign_farthest(
                        coords, weights, label)
                else:
                    coords, weights, score_arr = (
                        self._reassign_closest(
                            coords, weights, label, score_arr)
                    )

        return(coords)

    def _score_centroids_one_point(self, coords, label):
        '''Gets distance between point and remaining clusters
        '''

        point_dict = {
            _label: self._calculate_distance(
                coords, self.cluster_centers_[_label]
            )
            for _label in self._clusters_unfinalized
            if _label != label
        }

        return point_dict

    def _reassign_farthest(self, coords, weights, label):
        '''Sends point farthest from center to closest cluster
        '''

        # Calculated the cluster's current center of mass
        center_of_mass = (
            np.sum(weights[:, None]*coords, axis=0)
            / np.sum(weights)
        )[None, :]

        # Reassign this cluster's center to the center of mass
        self.cluster_centers_[label] = center_of_mass

        # Find the squared distances of the points to the center
        # of mass of the cluster, and find the point farthest
        _sqrd_dists = np.sum((coords - center_of_mass)**2, axis=1)
        _farthest_point_ind = np.argmax(_sqrd_dists)
        _farthest_point_dist = _sqrd_dists[_farthest_point_ind]
        _farthest_point = coords[_sqrd_dists == _farthest_point_dist]

        # Score the unfinalized centroids that are not from
        # the current cluster on the farthest point, then
        # find the best cluster for that point
        _centroid_scores = self._score_centroids_one_point(
            _farthest_point, label)
        _best_cluster = min(
            _centroid_scores, key=_centroid_scores.get)

        # Reassign the label in the final_label array
        _full_point_mask = np.isclose(self.X, _farthest_point).all(1)
        self.final_labels[_full_point_mask] = _best_cluster

        # Remove the farthest point from the current coordinate
        # and weight array
        _point_mask = (_sqrd_dists != _farthest_point_dist)
        _coords = coords[_point_mask]
        _weights = weights[_point_mask]

        return(_coords, _weights)

    def _score_other_points(self, coords, centroid):
        '''Finds the distance between points in coords and centroid
        '''

        # Make a mask for each point in coords that tells us
        # if it's in X
        _coords_masks = [
            np.isclose(self.X, coord).all(1)
            for coord in coords
        ]

        # Sum all the masks, and negate the result to obtain
        # a mask for only those points not in coords, then
        # use it to obtain an array of only those points
        _other_points_mask = np.logical_not(
            np.sum(_coords_masks, axis=0).astype(np.bool)
        )
        _other_points = self.X[np.logical_and(
            np.logical_not(self._labels_finalized),
            _other_points_mask
        )]

        # Make an array where the first column is the point
        # being considered, and the 2nd column is the squared
        # distance to the specified centroid
        score_arr = np.array([
            [point, np.sum((point - centroid)**2)]
            for point in _other_points
        ])

        return score_arr

    def _reassign_closest(
            self, coords, weights, label, score_arr):
        '''Reassigns closest point to current cluster (label)
        '''

        # Select the closest point
        _best_score_row = np.argmin(score_arr[:,1])
        _best_point = score_arr[_best_score_row][0]

        # Remove the closest point from score_arr to pass it out
        # of the function
        _score_arr = np.delete(score_arr, (_best_score_row), axis=0)

        # Reassign the label in the final_label array, and get
        # the weight of the best point
        _best_point_mask = np.isclose(self.X, _best_point).all(1)
        self._labels_finalized[_best_point_mask] = True
        self.final_labels[_best_point_mask] = label
        _best_weight = self.weights[_best_point_mask]

        # Add the best point to the _coords and _weights arrays
        _coords = np.vstack((coords, _best_point))
        _weights = np.append(weights, _best_weight)

        return(_coords, _weights, _score_arr)

    def _update_centroids(self):
        '''Recalculates cluster centroids
        '''

        # Create a dictionary of cluster centers by label
        self.cluster_centers_ = {
            label: np.mean(
                self.X[self.final_labels == label], axis=0)
            for label in self._unique_labels
        }

        return None

    def _calculate_distance(self, coord_arr, point=None):
        '''Distance between points in coord_arr and point
        '''

        if coord_arr.ndim == 1:
            _axis = 0
        else:
            _axis = 1

        if point is None:
            _point = 0
        else:
            _point = point

        if self.metric == 'l2':
            _distances = np.sum(
                (coord_arr - _point)**2, axis=_axis)
        elif self.metric == 'l1':
            _distances = np.sum(
                np.abs(coord_arr - _point), axis=_axis)
        elif self.metrid == 'l_inf':
            _distances = np.max(
                np.abs(coord_arr - _point), axis=_axis)

        return _distances

class SSGraphKMeans(object):
    '''Same-size clustering in fully-connected, undirected graphs

    Parameters
    ----------
    n_clusters: int, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    tol: numpy array(floats), default: np.linspace(5e4, 1.5e3, 8)
        The tolerances through which to step as the algorithm brings
        each cluster's weight towards equilibrium. The default is
        chosen for working with state populations. That is, for
        weights of each node on the order of 1000.

    save_labels: bool, default: False
        Whether to save labels at each step of the fitting
        process.

    Attributes
    ----------
    clusters: dict
        Dictionary of GraphCluster objects. Keys are ints from
        1:n_clusters. After calling fit(), each GraphCluster
        object will have its members attribute appropriately
        populated.

    cluster_history: None or dict
        If save_labels is True, cluster_history will be a dictionary
        whose keys are either: 'initial', 'final', and all but the final
        element of tol. The values will be dictionaries in the same
        style as clusters above.
    '''

    def __init__(self, n_clusters=8, tol=np.linspace(5e4, 1.5e3, 8),
                 save_labels=False):
        self.n_clusters = n_clusters
        self.tol = tol
        self.save_labels = save_labels
        self.cluster_weights = {i+1: 0 for i in range(n_clusters)}

    def fit(self, graph, graph_distances, node_weights):
        '''Fit the model on the given graph

        Parameters
        ----------
        graph: networkx Graph instance
            A graph of N nodes. Graph *must* be fully connected.

        graph_distances: dict of dicts
            graph_distances should be precomputed using
            networkx.shortest_path_length(graph)

        node_weights: dict of ints or floats
            Each node label of the input graph must be a key in node_weights
        '''

        self.graph = graph
        self._isolated_node_dict = {
            graph[node].keys()[0]: node
            for node in graph
            if len(graph[node]) == 1
        }
        self.graph_distances = graph_distances
        self.node_weights = node_weights
        self._ideal_cluster_weight = sum(node_weights.values())/self.n_clusters

        self._seed_clusters()
        self._grow_clusters()

        for tolerance in self.tol:
            print('Annealing cluster weights to within {}'.format(tolerance))
            self._frozen_nodes = set() # thaw clusters for this tolerance
            for cluster_id in self._randomized_clusters():
                break
                print('Annealing cluster {}'.format(cluster_id))
                self._anneal(cluster_id, tolerance)
                self.clusters[cluster_id].set_center()

    def _seed_clusters(self):
        '''Add single nodes to each cluster in self.clusters
        '''

        initial_nodes = random.sample(self.graph.nodes(), self.n_clusters)
        self._frozen_nodes = set(initial_nodes)
        self._node_clusters = {node: None for node in self.graph}
        self.clusters = {}

        for i, node in enumerate(initial_nodes):
            self.clusters[i+1] = GraphCluster({node}, {node})
            self.cluster_weights[i+1] += self.node_weights[node]
            self._node_clusters[node] = i+1


    def _grow_clusters(self):
        '''Iteratively grow cluster seeds until all nodes are in a cluster
        '''

        n_nodes = len(self.graph.nodes())
        while len(self._frozen_nodes) != n_nodes:
            for cluster_id in self._randomized_clusters():
                self._absorb_neighbors(cluster_id)

        for cluster_id in self.clusters:
            self.clusters[cluster_id].set_center(self.graph, self.node_weights)

        self._set_borders()


    def _randomized_clusters(self):
        '''Create randomized list of cluster IDs
        '''

        rand_cluster_ids = random.sample(
            range(1,self.n_clusters+1), self.n_clusters
        )

        return rand_cluster_ids

    def _absorb_neighbors(self, cluster_id):
        '''Assigns available neghbors of given cluster to that cluster
        '''

        for border_member in self.clusters[cluster_id].border.copy():
            neighbors = [
                neighbor for neighbor in self.graph[border_member]
                if self._node_clusters[neighbor] is None
            ]
            for neighbor in neighbors:
                self._reassign_node(neighbor, cluster_id, border=True)

    def _reassign_node(self, node, cluster_id, border=False):
        '''Reassign given node to given cluster
        '''

        current_cluster = self._node_clusters[node]
        if current_cluster is not None:
            self.clusters[current_cluster].remove_member(node)
            self.cluster_weights[current_cluster] -= self.node_weights[node
        self._node_clusters[node] = cluster_id
        self.clusters[cluster_id].add_member(node)
        if border:
            self.clusters[cluster_id].add_to_border(node)
        self.cluster_weights[cluster_id] += self.node_weights[node]
        self._frozen_nodes.add(node)]
        # Handle isolated nodes, if applicable
        if node in self._isolated_node_dict:
            _isolated_node = self._isolated_node()
            self._reassign_node(_isolated_node, cluster_id)

    def _set_borders(self, cluster=None):
        '''Determine the borders of all or given cluster

        Parameters
        ----------
        cluster: int, default: None
            Specifies the cluster to determine the borders of. If left
            as None, all clusters have their borders determined.
        '''

        def reset_cluster_border(cluster):
            for node in self.clusters[cluster]:
                neighbors_in_this_cluster = [
                    neighbor in self.clusters[cluster]
                    for neighbor in self.graph[node]
                ]
                if all(neighbors_in_this_cluster):
                    self.clusters[cluster].remove_from_border(node)
                else:
                    self.clusters[cluster].add_to_border(node)

        if cluster is None:
            for _cluster in self.clusters:
                reset_cluster_border(_cluster)
        else:
            reset_cluster_border(cluster)

    def _freeze_cluster(self, cluster_id):
        '''Add members of cluster to the set of "frozen out" nodes
        '''

        self._frozen_nodes.update(self.clusters[cluster_id])

    def _anneal(self, cluster_id, tol):
        '''Update the given cluster until its weight is within tol
        '''

        start_weight = self.cluster_weights[cluster_id]
        while not self._cluster_within_tolerance(cluster_id, tol):
            if start_weight < self._ideal_cluster_weight:
                self._grow_cluster(cluster_id, tol)
            elif start_weight > self._ideal_cluster_weight:
                self._shrink_cluster(cluster_id, tol)

        # Reset cluster centers for next iteration in fit()
        for _clstr in self.clusters:
            self.clusters[_clstr].set_center(self.graph, self.node_weights)
        self._freeze_cluster(cluster_id)

    def _cluster_within_tolerance(self, cluster_id, tol):
        '''Check if given cluster's weight is within tol of ideal weight
        '''

        cluster_bool = abs(
            self.cluster_weights[cluster_id]
            - self._ideal_cluster_weight) < tol

        return cluster_bool

    def _grow_cluster(self, cluster_id, tol):
        '''Grow cluster until border is exhausted or weight is within tol
        '''

        changed_clusters = {cluster_id}

        _center = self.clusters[cluster_id].center
        sorted_border = self._sort_border(cluster_id)
        # Iterate through members of the cluster's border, then add their
        # neighbors from other clusters to the current cluster
        for border_member in sorted_border:
            for neighbor in self.graph[border_member]:
                if (neighbor in self._frozen_nodes
                        or neighbor in self.clusters[cluster_id]):
                    continue
                self._reassign_node(neighbor, cluster_id, border=True)
                changed_clusters.add(self._node_clusters[neighbor])
                if self._cluster_within_tolerance(cluster_id, tol):
                    for _cluster in changed_clusters:
                        self._set_borders(cluster=_cluster)
                    return

        # Determine the borders of the clusters that have been changed
        # for the next iteration in _anneal()
        for _cluster in changed_clusters:
            self._set_borders(cluster=_cluster)

    def _sort_border(self, cluster_id):
        '''Create list of cluster.border sorted by distance to cluster center
        '''

        _center = self.clusters[cluster_id].center
        sorted_border = sorted(
            self.clusters[cluster_id].border,
            key=lambda x: self.graph_distances[_center][x], reverse=True
        )
        print('successfully sorted border')

        return sorted_border


    def _shrink_cluster(self, cluster_id, tol):
        '''Shrink cluster until border is exhausted or weight is within tol
        '''

        _center = self.clusters[cluster_id].center
        sorted_border = self._sort_border(cluster_id)
        # Iterate through the
        for border_member in sorted_border:
            _new_cluster = self._preferred_cluster(border_member)
            self._reassign_node(border_member, _new_cluster)

    def _preferred_cluster(self, node):
        '''Find most common cluster of node's neighbors not in node's cluster
        '''

        node_cluster = self._node_clusters[node]
        # Get node neighbors that aren't in the node's cluster
        node_neighbors = [
            _node for _node in self.graph[node]
            if _node not in self.clusters[node_cluster]
        ]

        cluster_votes = Counter([
            self._node_clusters[neighbor]
            for neighbor in node_neighbors
        ])

        winning_cluster = max(cluster_votes, key=cluster_votes.get)

        return winning_cluster

class GraphCluster(object):
    '''Container for clusters in SSGraphKMeans

    Parameters
    ----------
    members: set, default: set()
        A set of nodes which are contained within a given cluster
    border: set, default: set()
        The subset of members which have neighbors that are not in
        its own cluster

    Attributes
    ----------
    members: See above
    border: See above
    center: str or int
        ID of node at the center of the cluster. Ties are broken by
        the weights of the nodes
    '''

    def __init__(self, members=set(), border=set()):
        self.members = members
        self.border = border
        self.center = None

    def __contains__(self, node):
        return node in self.members

    def __iter__(self):
        return self.members.__iter__()

    def __add__(self, other):
        return self.members + other

    def __sub__(self, other):
        return self.members + other

    def __str__(self):
        return self.members.__str__()

    def __repr__(self):
        return self.members.__repr__()

    def __len__(self):
        return self.members.__len__()

    def add_member(self, node):
        self.members.add(node)

    def add_to_border(self, node):
        self.border.add(node)
        self.add_member(node)

    def remove_member(self, node):
        self.members.discard(node)
        self.remove_from_border(node)

    def remove_from_border(self, node):
        self.border.discard(node)

    def set_center(self, full_graph, weights):
        '''Finds the center of this cluster, breaking ties by node weight
        '''
        sub_graph = full_graph.subgraph(self)
        self.center = max(nx.center(sub_graph), key=weights.get)
