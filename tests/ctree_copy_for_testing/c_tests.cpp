#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <sys/types.h>
#include "cnode.h"
#include <cassert>

void test_CNode_constructor() {
    printf("Smoke testing CNode constructor CNode(float prior, int action_num, std::vector<CNode> *ptr_node_pool, float beta) \n");
    float prior = 0.1;
    int action_num = 3;
    float beta = 100.0;
    tree::CNode *node_1 = new tree::CNode();
    tree::CNode *node_2 = new tree::CNode();
    tree::CNode *node_3 = new tree::CNode();

    std::vector<tree::CNode> *ptr_node_pool = new std::vector<tree::CNode>();
    ptr_node_pool->push_back(*node_1);
    ptr_node_pool->push_back(*node_2);
    ptr_node_pool->push_back(*node_3);

    tree::CNode *contructor_tester = new tree::CNode(prior, action_num, ptr_node_pool, beta);

    printf("CNode constructor smoke test passed \n");
}

void smoke_test_expand() {
    printf("Smoke testing expand(..., float value_prefix_uncertainty) \n");
    float prior = 0.1;
    int action_num = 3;
    float beta = 100.0;
    tree::CNode *node_1 = new tree::CNode();
    tree::CNode *node_2 = new tree::CNode();
    tree::CNode *node_3 = new tree::CNode();

    std::vector<tree::CNode> *ptr_node_pool = new std::vector<tree::CNode>();
    ptr_node_pool->push_back(*node_1);
    ptr_node_pool->push_back(*node_2);
    ptr_node_pool->push_back(*node_3);

    tree::CNode *contructor_tester = new tree::CNode(prior, action_num, ptr_node_pool, beta);

    int to_play = 0;
    int hidden_state_index_x = 0;
    int hidden_state_index_y = 0;
    float value_prefix = 1;
    const std::vector<float> policy_logits = {3.0, 4.0, 5.0};
    float value_prefix_uncertainty = 2;
    contructor_tester->expand(to_play, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits, value_prefix_uncertainty);
    printf("Expand smoke test passed \n");
}

void smoke_test_cbatch_back_propagate() {
    printf("Testing cbatch_back_propagate \n");
    int hidden_state_index_x = 0;
    int discount = 0.997;
    int beta = 100;
    std::vector<float> value_prefixs = {1.0, 2.0, 3.0};
    std::vector<float> values = {4.0, 5.0, 6.0};
    std::vector<std::vector<float>> policies = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    tools::CMinMaxStatsList min_max_stats_lst;
    tree::CSearchResults results;
    std::vector<int> is_reset_lst = {0, 0, 1};
    std::vector<float> value_prefixs_uncertainty = {0.1, 0.2, 0.3};
    std::vector<float> values_uncertainty = {0.4, 0.5, 0.6};

    // Call the function
    cbatch_back_propagate(hidden_state_index_x, discount, value_prefixs, values, policies, &min_max_stats_lst, results, is_reset_lst, value_prefixs_uncertainty, values_uncertainty, beta);
    printf("cbatch_back_propagate smoke test passed \n");
}

void smoke_test_cback_propagate() {
    printf("Smoke testing c_back_propagate \n");

    // init a node
    float prior = 0.1;
    int action_num = 1;
    float beta = 100.0;
    int hidden_state_index_x = 0;
    float value_prefix = 3.0;
    const std::vector<float> policy_logits = {5.0};
    float value_prefix_uncertainty = 0.7;
    int hidden_state_index_y = 0;


    tree::CNode *node_1 = new tree::CNode();    // init a child
    std::vector<tree::CNode> *ptr_node_pool = new std::vector<tree::CNode>();   // init a child-pool of size num.actions
    ptr_node_pool->push_back(*node_1);  // insert the node to the child pool

    tree::CNode *test_node = new tree::CNode(prior, action_num, ptr_node_pool, beta);   // create a new node

    // expand a node
    test_node->expand(0, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits, value_prefix_uncertainty);

    std::vector<tree::CNode*> search_path = {
            test_node
    };
    tools::CMinMaxStats min_max_stats;
    int to_play = 1;
    float value = 3.14;
    float discount = 0.9;
    float value_uncertainty = 0.2;

    // pass the node to cback_propagate
    tree::cback_propagate(search_path, min_max_stats, to_play, value, discount, value_uncertainty);

    // Verify the results
    printf("cback_propagate smoke test passed \n");
}

void test_value_uncertainty() {
    printf("Testing c_back_propagate \n");
    // init a node
    float prior = 0.1;
    int action_num = 1;
    float beta = 100.0;
    int hidden_state_index_x = 0;
    float value_prefix = 3.0;
    const std::vector<float> policy_logits = {5.0};
    float value_prefix_uncertainty = 5;
    int hidden_state_index_y = 0;


    tree::CNode *node_1 = new tree::CNode();    // init a child
    std::vector<tree::CNode> *ptr_node_pool = new std::vector<tree::CNode>();   // init a child-pool of size num.actions
    ptr_node_pool->push_back(*node_1);  // insert the node to the child pool

    tree::CNode *test_node = new tree::CNode(prior, action_num, ptr_node_pool, beta);   // create a new node

    // expand a node
    test_node->expand(0, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits, value_prefix_uncertainty);

    // pass it to cback_propagate so that the counter will increase
    std::vector<tree::CNode*> search_path = {
            test_node
    };
    tools::CMinMaxStats min_max_stats;
    int to_play = 1;
    float value = 3.14;
    float discount = 0.9;
    float value_uncertainty = 4.0;

    // pass the node to cback_propagate
    tree::cback_propagate(search_path, min_max_stats, to_play, value, discount, value_uncertainty);

    // test that value_uncertainty() is what it is expected to be
    float node_value_uncertainty = test_node->value_uncertainty();
    assert (value_uncertainty==node_value_uncertainty);

    printf("test_value_uncertainty passed \n");
}

void test_prepare_explore() {
    printf("Testing prepare_explore \n");
    // setup CRoot
    int root_num = 1;
    int action_num = 2;
    int pool_size = 1;
    tree::CRoots *test_roots = new tree::CRoots(root_num, action_num, pool_size);

    float root_exploration_fraction = 0.3;
    const std::vector<std::vector<float>> noises = {{15.0}};
    const std::vector<float> value_prefixs = {3.0};
    const std::vector<std::vector<float>> policies = {{0.5, 0.5}};
    const std::vector<float> value_prefixs_uncertainty = {7.0};

    // call
    test_roots->prepare_explore(root_exploration_fraction, noises, value_prefixs, policies, value_prefixs_uncertainty);
    printf("prepare_explore smoke test passed \n");
}

void test_CRoots() {
    printf("Testing CRoots \n");
    // setup CRoot
    int root_num = 1;
    int action_num = 2;
    int pool_size = 1;

    tree::CRoots *test_roots = new tree::CRoots(root_num, action_num, pool_size);
    printf("CRoots smoke test passed \n");
}

void test_get_mean_q_uncertainty() {
    // Not implemented yet
}

void test_cucb_score() {
    printf("Testing cucb_score \n");
    // setup
    // init a node
    float prior = 0;
    int action_num = 1;
    float beta = 1.0;
    int hidden_state_index_x = 0;
    float value_prefix = 1;
    const std::vector<float> policy_logits = {0};
    float value_prefix_uncertainty = 4;
    int hidden_state_index_y = 0;
    float discount = 0.9;

    tree::CNode *node_1 = new tree::CNode();    // init a child
    std::vector<tree::CNode> *ptr_node_pool = new std::vector<tree::CNode>();   // init a child-pool of size num.actions
    ptr_node_pool->push_back(*node_1);  // insert the node to the child pool

    tree::CNode *test_node = new tree::CNode(prior, action_num, ptr_node_pool, beta);   // create a new node

    // expand a node
    test_node->expand(0, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits, value_prefix_uncertainty);

    // pass it to cback_propagate so that the counter will increase
    std::vector<tree::CNode*> search_path = {
            test_node
    };
    tools::CMinMaxStats min_max_stats;
    int to_play = 1;
    float value = 5 / discount;// 5 / discount;

    float value_uncertainty = 5 / (discount * discount);

    tree::cback_propagate(search_path, min_max_stats, to_play, value, discount, value_uncertainty);

    // init additional params for ucb_score
    float parent_mean_q = -1;
    int is_reset = 0;
    float total_children_visit_counts = 1;
    float parent_value_prefix = 0;
    float pb_c_base = 1;
    float pb_c_init = 0;
    float parent_mean_q_uncertainty = 0;
    
    //call
    float ucb_score = cucb_score(test_node, min_max_stats, parent_mean_q, is_reset, total_children_visit_counts, parent_value_prefix, pb_c_base, pb_c_init, discount, parent_mean_q_uncertainty);

    printf("cucb_score smoke test passed \n");

    // Tested with prints in inside UCB value (otherwise a bit hard to test, because all values normalize to 0-1)

    printf("cucb_score value test passed \n");
}

void test_cselect_child() {
    printf("Testing cselect_child \n");
    // setup
    tools::CMinMaxStats min_max_stats;
    float pb_c_base = 1;
    float pb_c_init = 0;
    float discount = 0.9;
    float mean_q = 10;
    float mean_q_uncertainty = 9;

    // Setup a node to select on
    float prior = 0;
    int action_num = 1;
    float beta = 1.0;
    int hidden_state_index_x = 0;
    float value_prefix = 1;
    const std::vector<float> policy_logits = {0};
    float value_prefix_uncertainty = 4;
    int hidden_state_index_y = 0;

    tree::CNode *node_1 = new tree::CNode();    // init a child
    std::vector<tree::CNode> *ptr_node_pool = new std::vector<tree::CNode>();   // init a child-pool of size num.actions
    ptr_node_pool->push_back(*node_1);  // insert the node to the child pool

    tree::CNode *test_node = new tree::CNode(prior, action_num, ptr_node_pool, beta);   // create a new node

    // expand a node
    test_node->expand(0, hidden_state_index_x, hidden_state_index_y, value_prefix, policy_logits, value_prefix_uncertainty);

    // pass it to cback_propagate so that the counter will increase
    std::vector<tree::CNode*> search_path = {
            test_node
    };
    int to_play = 1;
    float value = 5 / discount;// 5 / discount;

    float value_uncertainty = 5 / (discount * discount);

    tree::cback_propagate(search_path, min_max_stats, to_play, value, discount, value_uncertainty);

    // test
    int child_id = cselect_child(test_node, min_max_stats, pb_c_base, pb_c_init, discount, mean_q, mean_q_uncertainty);

    printf("child_id = %d \n", child_id);

    printf("cselect_child smoke test passed \n");
}

void test_update_tree_q_original_code() {
    //TODO: Test that works as expected with regular nodes.
}

void test_update_tree_q_uncertainty_code() {
    //TODO: Test that works as expected with modified nodes.
}

int main(int argc, char *argv[]) {
    // To run this file:
    // navigate to dir: cd /home/yaniv/EfficientExplore/tests/ctree_copy_for_testing)
    // call make: make
    // call c_tests_for_mcts: ./c_tests_for_mcts

    printf("Running CPP functions tests \n");
    test_CNode_constructor();
    smoke_test_expand();
    smoke_test_cbatch_back_propagate();
    smoke_test_cback_propagate();
    test_value_uncertainty();
    test_prepare_explore();
    test_CRoots();
    test_get_mean_q_uncertainty();
    test_cucb_score();
    test_cselect_child();
    test_update_tree_q_original_code();
    test_update_tree_q_uncertainty_code();
}

