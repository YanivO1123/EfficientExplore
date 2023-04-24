import torch


def evaluate_unrolled(model, starting_observations, unroll_steps, env_size, step, model_type='catted'):
    with torch.no_grad():
        batch_size = starting_observations.shape[0]
        running_batch_size = batch_size
        device = starting_observations.device
        # get true actions right
        actions_true_right = model.dynamics_network.get_actions_right(starting_observations.squeeze()).reshape(-1, 1)

        # Setup starting states
        states = model.representation(starting_observations)
        reward_hidden = (torch.zeros(1, batch_size, model.lstm_hidden_size).to(device),
                             torch.zeros(1, batch_size, model.lstm_hidden_size).to(device))

        # Start unrolling
        for i in range(unroll_steps + 1):

            if i > 0:
                states, _, _ = model.dynamics_network(reward_hidden, states[:-1], actions_true_right[i:].long())
            else:
                states, _, _ = model.dynamics_network(reward_hidden, states, actions_true_right.long())
            running_batch_size -= 1
            reward_hidden = (torch.zeros(1, running_batch_size, model.lstm_hidden_size).to(device),
                             torch.zeros(1, running_batch_size, model.lstm_hidden_size).to(device))
            print(f"In evaluate_unrolled, at step {step}, unroll step {i}. Model type: {model_type}")
            if 'catted' in model_type:
                evaluate_row_col_cat_form(states, env_size)
            else:
                evaluate_states(states, env_size)


def evaluate_states(state_prediction, env_size):
    B = state_prediction.shape[0]
    N = env_size
    flattened_state_prediction = state_prediction.reshape(shape=(B, N * N))
    rows, cols = (flattened_state_prediction.argmax(dim=1) // N).long().squeeze(), (
            flattened_state_prediction.argmax(dim=1) % N).long().squeeze()

    state_prediction = state_prediction.reshape(B, 1, N, N)

    assert rows.shape == cols.shape and rows.shape[
        0] == B, \
        f"rows.shape = {rows.shape}, cols.shape = {cols.shape}, batch_size = {B}"

    for i in range(N - B, N):
        index = i - (N - B)
        print(f"({i}, {i}), action true right, predicted state: {rows[index].item(), cols[index].item()},"
              f" expected ({i + 1}, {i + 1}). "
              f"and value at that index: {state_prediction[index, 0, rows[index], cols[index]]} \n"
              , flush=True)


def evaluate_row_col_cat_form(flattened_state_batch, env_size):
    """
        Takes a tensor of shape [B, 2 * N] and evaluates how good it is as a state prediction
    """
    assert flattened_state_batch.shape == (flattened_state_batch.shape[0], 2 * env_size)
    B = flattened_state_batch.shape[0]
    N = env_size
    rows, cols = flattened_state_batch[:, :N].argmax(dim=-1).long().squeeze(), flattened_state_batch[:, N:].argmax(dim=-1).long().squeeze()

    assert rows.shape == cols.shape and rows.shape[
        0] == B, \
        f"rows.shape = {rows.shape}, cols.shape = {cols.shape}, batch_size = {B}"

    for i in range(N - B, N):
        index = i - (N - B)
        print(f"({i}, {i}), action true right, predicted state: {rows[index].item(), cols[index].item()},"
              f" expected ({i + 1}, {i + 1}). "
              f"and values at flattened row / column indexes: "
              f"{(flattened_state_batch[index, rows[index]].item(), flattened_state_batch[index, N + cols[index]].item())}. Expecting (10, 10)"
              , flush=True)
