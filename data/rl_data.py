import pandas as pd


def generate_rl_df(
        state: pd.DataFrame,
        action: pd.DataFrame | pd.Series,
        reward: pd.DataFrame | pd.Series,
        next_state: pd.DataFrame,
        done: pd.Series
    ) -> pd.DataFrame:
    index = state.index

    if (index != action.index).any() or \
       (index != reward.index).any() or \
       (index != next_state.index).any() or \
       (index != done.index).any():
        raise ValueError("Index mismatch")
    assert (state.columns == next_state.columns).all()

    columns = []

    for state_feature in state.columns:
        columns.append(("state", state_feature))
    
    if isinstance(action, pd.Series):
        columns.append(("action", ''))
    
    else:
        for action_feature in action.columns:
            columns.append(("action", action_feature))


    if isinstance(reward, pd.Series):
        columns.append(("reward", ''))
    
    else:
        for reward_feature in reward.columns:
            columns.append(("reward", reward_feature))


    for state_feature in next_state.columns:
        columns.append(("next_state", state_feature))

    columns.append(("done", ''))

    rl_data = pd.DataFrame(index=index, columns=pd.MultiIndex.from_tuples(columns))

    rl_data["state"]      = state
    rl_data["action"]     = action
    rl_data["reward"]     = reward
    rl_data["next_state"] = next_state
    rl_data["done"]       = done

    return rl_data
