# distutils: language=c++
import ctypes
cimport cython
from ctree cimport CMinMaxStatsList, CNode, CRoots, CSearchResults, cbatch_back_propagate, cbatch_traverse, cuncertainty_batch_back_propagate
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.list cimport list as cpplist

import numpy as np
cimport numpy as np

ctypedef np.npy_float FLOAT
ctypedef np.npy_intp INTP


cdef class MinMaxStatsList:
    cdef CMinMaxStatsList *cmin_max_stats_lst

    def __cinit__(self, int num):
        self.cmin_max_stats_lst = new CMinMaxStatsList(num)

    def set_delta(self, float value_delta_max):
        self.cmin_max_stats_lst[0].set_delta(value_delta_max)

    def __dealloc__(self):
        del self.cmin_max_stats_lst


cdef class ResultsWrapper:
    cdef CSearchResults cresults

    def __cinit__(self, int num):
        self.cresults = CSearchResults(num)

    def get_search_len(self):
        return self.cresults.search_lens


cdef class Roots:
    cdef int root_num
    cdef int pool_size
    cdef CRoots *roots

    def __cinit__(self, int root_num, int action_num, int tree_nodes, float beta=-1):
        self.root_num = root_num
        self.pool_size = action_num * (tree_nodes + 2)
        #MuExplore: init a CRoots where the first root is standard but the rest are exploratory
        if beta >= 0:
            self.roots = new CRoots(root_num, action_num, self.pool_size, beta)
        else:
            self.roots = new CRoots(root_num, action_num, self.pool_size)

    #MuExplore: prepare_explore prepares roots for exploration episodes
    def prepare_explore(self, float root_exploration_fraction, list noises, list value_prefix_pool, list policy_logits_pool, list value_prefixs_uncertainty_pool, float beta):
        self.roots[0].prepare_explore(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool, value_prefixs_uncertainty_pool, beta)

    def prepare(self, float root_exploration_fraction, list noises, list value_prefix_pool, list policy_logits_pool):
        self.roots[0].prepare(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)

    def prepare_no_noise(self, list value_prefix_pool, list policy_logits_pool):
        self.roots[0].prepare_no_noise(value_prefix_pool, policy_logits_pool)

    def get_trajectories(self):
        return self.roots[0].get_trajectories()

    def get_distributions(self):
        return self.roots[0].get_distributions()

    def get_values_uncertainty(self):
        return self.roots[0].get_values_uncertainty()

    def get_values(self):
        return self.roots[0].get_values()

    def clear(self):
        self.roots[0].clear()

    def __dealloc__(self):
        del self.roots

    @property
    def num(self):
        return self.root_num


cdef class Node:
    cdef CNode cnode

    def __cinit__(self):
        pass

    def __cinit__(self, float prior, int action_num):
        # self.cnode = CNode(prior, action_num)
        pass

    #MuExplore: an expand function that takes value_prefix_uncertainty
    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, list policy_logits, float value_prefix_uncertainty):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, cpolicy, value_prefix_uncertainty)

    def expand(self, int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, list policy_logits):
        cdef vector[float] cpolicy = policy_logits
        self.cnode.expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, cpolicy)

def batch_back_propagate(int hidden_state_index_x, float discount, list value_prefixs, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies

    cbatch_back_propagate(hidden_state_index_x, discount, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst)


#MuExplore: batch_back_propagate that backprops uncertainty
def uncertainty_batch_back_propagate(int hidden_state_index_x, float discount, list value_prefixs, list values, list policies, MinMaxStatsList min_max_stats_lst, ResultsWrapper results, list is_reset_lst, list value_prefixs_uncertainty, list values_uncertainty):
    cdef int i
    cdef vector[float] cvalue_prefixs = value_prefixs
    cdef vector[float] cvalues = values
    cdef vector[vector[float]] cpolicies = policies
    cdef vector[float] cvalue_prefixs_uncertainty = value_prefixs_uncertainty
    cdef vector[float] cvalues_uncertainty = values_uncertainty

    cuncertainty_batch_back_propagate(hidden_state_index_x, discount, cvalue_prefixs, cvalues, cpolicies,
                          min_max_stats_lst.cmin_max_stats_lst, results.cresults, is_reset_lst,
                          cvalue_prefixs_uncertainty, cvalues_uncertainty)

def batch_traverse(Roots roots, int pb_c_base, float pb_c_init, float discount, MinMaxStatsList min_max_stats_lst, ResultsWrapper results):

    cbatch_traverse(roots.roots, pb_c_base, pb_c_init, discount, min_max_stats_lst.cmin_max_stats_lst, results.cresults)

    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions
