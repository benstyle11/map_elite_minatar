import Agent



def nn_action(state):
    return torch.argmax(policy_net(s))

def play(network_action ,game = "breakout", display=False):
    env = Environment(game, sticky_action_prob=0.0, random_seed=0)
    env.reset()
    is_terminated = False
    total_reward = 0.0
    t = 0

    behaviour = np.zeros(env.num_actions()) #The behaviour of the agent

    while (not is_terminated) and t < NUM_FRAMES:
        s = get_state(env.state())
        with torch.no_grad():
            action = network_action(s)

        behaviour[action] += 1 ## add one to the corresponding behaviour

        reward, is_terminated = env.act(action)
        total_reward += reward
        t += 1
        if display:
            env.display_state(1)

    return total_reward, behaviour