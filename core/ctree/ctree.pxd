# distutils: language=c++
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t


cdef extern from "cminimax.cpp":
    pass


cdef extern from "cminimax.h" namespace "tools":
    cdef cppclass CMinMaxStats:
        CMinMaxStats() except +
        float maximum, minimum, value_delta_max

        void set_delta(float value_delta_max)
        void update(float value)
        void clear()
        float normalize(float value)

    cdef cppclass CMinMaxStatsList:
        CMinMaxStatsList() except +
        CMinMaxStatsList(int num) except +
        int num
        vector[CMinMaxStats] stats_lst

        void set_delta(float value_delta_max)

cdef extern from "cnode.cpp":
    pass


cdef extern from "cnode.h" namespace "tree":
    cdef cppclass CNode:
        CNode() except +
        CNode(float prior, int action_num, vector[CNode]* ptr_node_pool) except +
        #MuExplore: New CNode declaration that can take the exploration constant beta
        CNode(float prior, int action_num, vector[CNode]* ptr_node_pool, float beta) except +
        int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, best_action
        float value_prefixs, prior, value_sum
        vector[int] children_index;
        vector[CNode]* ptr_node_pool;
        #MuExplore: New CNode attributes
        float value_prefix_uncertainty, value_uncertainty_sum, beta
        bool_t mu_explore

        void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefixs, vector[float] policy_logits)
        #MuExplore: an expand function that takes value_prefix_uncertainty
        void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefixs,
                    vector[float] policy_logits, float value_prefixs_uncertainty)
        void add_exploration_noise(float exploration_fraction, vector[float] noises)
        float get_mean_q(int isRoot, float parent_q, float discount)
        #MuExplore: get_mean_q_uncertainty function that computes q_uncertainty instead of q
        float get_mean_q_uncertainty(int isRoot, float parent_q_uncertainty, float discount)

        int expanded()
        float value()
        #MuExplore: get value_uncertainty of node
        float value_uncertainty()
        vector[int] get_trajectory()
        vector[int] get_children_distribution()
        #MuExplore: get value uncertainty of children of node
        vector[float] get_children_uncertainties(float discount)
        CNode* get_child(int action)

    cdef cppclass CRoots:
        CRoots() except +
        CRoots(int root_num, int action_num, int pool_size) except +
        CRoots(int root_num, int action_num, int pool_size, float beta) except +
        int root_num, action_num, pool_size
        vector[CNode] roots
        vector[vector[CNode]] node_pools

        void prepare(float root_exploration_fraction, const vector[vector[float]] &noises, const vector[float] &value_prefixs, const vector[vector[float]] &policies)
        void prepare_no_noise(const vector[float] &value_prefixs, const vector[vector[float]] &policies)
        #MuExplore: prepare_explore prepares roots for exploration episodes
        void prepare_explore(float root_exploration_fraction, const vector[vector[float]] &noises, const vector[float] &value_prefixs, const vector[vector[float]] &policies, const vector[float] &value_prefixs_uncertainty, float beta)
        void clear()
        vector[vector[int]] get_trajectories()
        vector[vector[int]] get_distributions()
        # MuExplore: get the uncertainties of the children of each node in roots
        vector[vector[float]] get_roots_children_uncertainties(float discount)
        vector[float] get_values()
        vector[float] get_values_uncertainty()

    cdef cppclass CSearchResults:
        CSearchResults() except +
        CSearchResults(int num) except +
        int num
        vector[int] hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens
        vector[CNode*] nodes

    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount)
    #MuExplore: cback_propagate that backprops uncertainty
    cdef void cback_propagate(vector[CNode*] &search_path, CMinMaxStats &min_max_stats, int to_play, float value, float discount, float value_uncertainty)
    void cbatch_back_propagate(int hidden_state_index_x, float discount, vector[float] value_prefixs,
                               vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults & results, vector[int] is_reset_lst)
    #MuExplore: cbatch_back_propagate that backprops uncertainty
    void cuncertainty_batch_back_propagate(int hidden_state_index_x, float discount, vector[float] value_prefixs, vector[float] values, vector[vector[float]] policies,
                               CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, vector[int] is_reset_lst,
                                          vector[float] value_prefixs_uncertainty, vector[float] values_uncertainty)
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, CMinMaxStatsList *min_max_stats_lst, CSearchResults &results)
