from ppo import *
import gym

# TODO PARAMETERS
render = False
#render = True
env = gym.make('CartPole-v1')

# NOTE input of the original OpenAI Gym Environment
#print(env.observation_space) # Box
#print(env.action_space) # Discrete

# NOTE space...
obs_space = env.observation_space.shape
act_space = env.action_space.n

print(obs_space, act_space)

policy = PolicyWithValue(obs_space, act_space, 'policy')
old_policy = PolicyWithValue(obs_space, act_space, 'old_policy')

agent = PPOAgent(policy, old_policy, 32, 3e-4, 4, 32, 0.995, 0.95, 0.2, 1, 0.01)

# Initialize the agent
for e in range(2000):
    
    # Initialize OpenAI Gym Environment
    observation = env.reset()

    # Begin with score of 0
    score = 0
    for t in range(1000):
        if render:
            env.render()

        # Query the agent for its action decision
        action, value  = agent.action(observation)
        #print(action, value)

        # Execute the decision and retrieve the current performance score
        observation, reward, done, info = env.step(action)

        # Modify reward so that negative reward is given when it finishes too early
        reward = reward if not done or score >= 499 else -10

        # Pass feedback about performance (and termination) to the agent
        agent.observe_and_learn(reward=reward, terminal=done)
        
        # accumulate reward
        score += reward

        if done:
            print("Episode {} finished after {} timesteps".format(e+1, t+1))
            break
