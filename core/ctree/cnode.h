#ifndef CNODE_H
#define CNODE_H

#include "cminimax.h"
#include <math.h>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <sys/timeb.h>
#include <sys/time.h>

const int DEBUG_MODE = 0;

namespace tree {

    class CNode {
        public:
            int visit_count, to_play, action_num, hidden_state_index_x, hidden_state_index_y, best_action, is_reset;
            float value_prefix, prior, value_sum;

            //MuExplore: Stores additionally the uncertainty variables
            float value_prefix_uncertainty, value_uncertainty_sum, beta;
            bool mu_explore;

            std::vector<int> children_index;
            std::vector<CNode>* ptr_node_pool;


            CNode();

            //MuExplore: Constructor for node that includes the exploration coeff. beta
            CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool, float beta);

            CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool);
            ~CNode();


            // MuExplore: Expand takes an uncertainty argument
            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, const std::vector<float> &policy_logits, float value_prefix_uncertainty);
            void expand(int to_play, int hidden_state_index_x, int hidden_state_index_y, float value_prefix, const std::vector<float> &policy_logits);
            void add_exploration_noise(float exploration_fraction, const std::vector<float> &noises);
            //MuExplore: Get the mean q_uncertainty of a node
            float get_mean_q_uncertainty(int isRoot, float parent_q_uncertainty, float discount);
            float get_mean_q(int isRoot, float parent_q, float discount);
            void print_out();

            int expanded();

            float value();
            //MuExplore: Compute the accumulated value_uncertainty
            float value_uncertainty();

            std::vector<int> get_trajectory();
            std::vector<int> get_children_distribution();
            CNode* get_child(int action);
    };

    class CRoots{
        public:
            int root_num, action_num, pool_size;
            std::vector<CNode> roots;
            std::vector<std::vector<CNode>> node_pools;

            CRoots();
            CRoots(int root_num, int action_num, int pool_size);
            //MuExplore: Setup a CRoots that are exploratory (mu_explore = true, beta = beta)
            CRoots(int root_num, int action_num, int pool_size, float beta);
            ~CRoots();

            //MuExplore: prepare_explore prepares roots for exploration episodes
            void prepare_explore(float root_exploration_fraction, const std::vector<std::vector<float>> &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies, const std::vector<float> &value_prefixs_uncertainty, float beta);

            void prepare(float root_exploration_fraction, const std::vector<std::vector<float>> &noises, const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies);
            void prepare_no_noise(const std::vector<float> &value_prefixs, const std::vector<std::vector<float>> &policies);
            void clear();
            std::vector<std::vector<int>> get_trajectories();
            std::vector<std::vector<int>> get_distributions();
            std::vector<float> get_values();
            //MuExplore: Returns the value-uncertainty of the nodes
            std::vector<float> get_values_uncertainty();

    };

    class CSearchResults{
        public:
            int num;
            std::vector<int> hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions, search_lens;
            std::vector<CNode*> nodes;
            std::vector<std::vector<CNode*>> search_paths;

            CSearchResults();
            CSearchResults(int num);
            ~CSearchResults();

    };

    //*********************************************************
    void update_tree_q(CNode* root, tools::CMinMaxStats &min_max_stats, float discount);
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount);
    void cbatch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst);
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q);
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount);
    void cbatch_traverse(CRoots *roots, int pb_c_base, float pb_c_init, float discount, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results);
    //MuExplore:
    // A cbatch_back_propagate that also also backpropagates value_uncertainty
    void cuncertainty_batch_back_propagate(int hidden_state_index_x, float discount, const std::vector<float> &value_prefixs, const std::vector<float> &values, const std::vector<std::vector<float>> &policies, tools::CMinMaxStatsList *min_max_stats_lst, CSearchResults &results, std::vector<int> is_reset_lst, const std::vector<float> &value_prefixs_uncertainty, const std::vector<float> &values_uncertainty);
    // A cback_propagate that also backpropagates value_uncertainty
    void cback_propagate(std::vector<CNode*> &search_path, tools::CMinMaxStats &min_max_stats, int to_play, float value, float discount, float value_uncertainty);
    // A cselect_child function that uses mean_q_uncertainty
    int cselect_child(CNode* root, tools::CMinMaxStats &min_max_stats, int pb_c_base, float pb_c_init, float discount, float mean_q, float mean_q_uncertainty);
    // A cucb_score function that also takes parent_mean_q_uncertainty and parent_value_prefix_uncertainty
    float cucb_score(CNode *child, tools::CMinMaxStats &min_max_stats, float parent_mean_q, int is_reset, float total_children_visit_counts, float parent_value_prefix, float pb_c_base, float pb_c_init, float discount, float parent_mean_q_uncertainty);
}

#endif