import numpy
import numpy as np
import tests.ctree_copy_for_testing.cytree as cytree

class Interfacing_C_Tests:
    def __init__(self):
        # Init the shared parameters of the tests
        # CRoots
        self.env_num = 2
        self.action_space_size = 3
        self.num_simulations = 5
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.discount = 0.95
        self.value_delta_max = 0.01
        self.beta = 350
        self.test_roots = cytree.Roots(self.env_num, self.action_space_size, self.num_simulations, self.beta)
        self.min_max_stats_lst = cytree.MinMaxStatsList(self.test_roots.num)
        self.results = cytree.ResultsWrapper(self.test_roots.num)


    def run_tests(self):
        self.test_prepare_explore()
        self.test_batch_traverse()
        self.test_uncertainty_batch_back_propagate()
        # self.test_exploratory_search()

    def test_prepare_explore(self):
        root_dirichlet_alpha = 0.3
        noises = [np.random.dirichlet([root_dirichlet_alpha] * self.action_space_size).astype(
            np.float32).tolist() for _ in range(self.env_num)]
        root_exploration_fraction = 0.25
        # This needs to be of size env_num, which is 2
        value_prefix_pool = [5, 17]
        value_prefixs_uncertainty_pool = [13, 5]

        # This needs to be of size env_num x action_space, which is 2 x 3
        policy_logits_pool = [[3, -5, 22], [14, -20, 0]]
        self.test_roots.prepare_explore(root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool,
                                        value_prefixs_uncertainty_pool, self.beta)
        print(f"Prepare explore smoke test passed \n")

    def test_batch_traverse(self):
        # Test batch_traverse
        ## Test that call doesn't crash
        self.min_max_stats_lst.set_delta(self.value_delta_max)

        # prepare a result wrapper to transport results between python and c++ parts
        # Prepared outside
        # traverse to select actions for each root
        # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
        # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
        # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
        hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = cytree.batch_traverse(self.test_roots, self.pb_c_base,
                                                                                               self.pb_c_init, self.discount,
                                                                                               self.min_max_stats_lst,
                                                                                               self.results)
        print(f"batch_traverse smoke test passed \n")
        ## Test that

    def test_uncertainty_batch_back_propagate(self):
        # Test batch_back_propagate:
        hidden_state_index_x = 0
        value_prefix_pool = [-2, 7]
        value_pool = [5, 3]
        policy_logits_pool = [[11, 17, -40], [31, -100, 55]]
        is_reset_lst = [0, 0]
        value_prefixs_uncertainty = [34.0, 52.1]
        values_uncertainty = [-15, 44.5]

        ## Test that call doesn't crash
        hidden_state_index_x += 1
        cytree.uncertainty_batch_back_propagate(hidden_state_index_x, self.discount,
                                  value_prefix_pool, value_pool, policy_logits_pool,
                                  self.min_max_stats_lst, self.results, is_reset_lst,
                                  value_prefixs_uncertainty, values_uncertainty)
        print(f"uncertainty_batch_back_propagate smoke test passed \n")
        ## Test that results update correctly

    def test_exploratory_search(self):
        #TODO: The purpose of this test is to test that prepare explore induces behavior of 1 exploit root MCTS, and the rest of the parallel roots explore MCTS.

        raise NotImplementedError


Interfacing_C_Tests().run_tests()