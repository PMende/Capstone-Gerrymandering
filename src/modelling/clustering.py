from __future__ import absolute_import, division, print_function
from builtins import (
    ascii, bytes, chr, dict, filter, hex, input, int, map,
    next, oct, open, pow, range, round, str, super, zip)

import os
from itertools import cycle
from functools import partial
from copy import deepcopy
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

class SameSizeKMeans(object):
    '''K-Means clustering

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    weight_tol : float, default: 1e-4
        Fractional tolerance of the

    init_model: KMeans object, default: None
        The initial KMeans model to fit on. Leaving as None
        defaults to KMeans with default parameters except for
        passing the as-specified n_clusters.

    save_labels: bool, default: False
        Whether to save labels at each step of the fitting
        process. Setting to True will cause the creation

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Final coordinates of cluster centers

    final_labels: array, [n_clusters, 1]
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

        self._save_fit_params(X, weights, weight_tol, order)

        # Get temporary labels from a naive KMeans clustering
        _temp_labels = self._fit_naive_KMeans(X)

        for step in range(self.n_clusters - self.LOOP_BUFFER):

            print('Starting step ',step)

            # Determine the cluster to update on this step, along
            # with its associated info
            _coords, _weights, _label, _score_arr, _centroid = (
                self._get_cluster_info(order, step)
            )

            # Adjust the determined cluster, and return its finalized
            # associated coordinates
            _coords = self._adjust_cluster(
                _coords, _weights,
                _label, _score_arr,
                order, step)

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

    def _fit_naive_KMeans(self, X):

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

        # Create a dictionary of centers of mass by label
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
        '''Finds the unfinalized cluster according to order'''

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

        # Check if a cluster is

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
        '''Gets distance between point and remaining clusters'''

        point_dict = {
            _label: self._calculate_distance(
                coords, self.cluster_centers_[_label]
            )
            for _label in self._clusters_unfinalized
            if _label != label
        }

        return point_dict

    def _reassign_farthest(self, coords, weights, label):

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

        # Create a dictionary of cluster centers by label
        self.cluster_centers_ = {
            label: np.mean(
                self.X[self.final_labels == label], axis=0)
            for label in self._unique_labels
        }

        return None

    def _calculate_distance(self, coord_arr, point=None):

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


def find_nearest(coord_arr, point):
    idx = np.abs(coord_arr - point).argmin()
    return{'dist': coord_arr[idx], 'idx': idx}
