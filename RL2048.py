"""
Dr. J, Spring 2024
Implementing Deep Q Network for Reinforcement learning solution to 2048.
"""
import random
import tensorflow as tf
import numpy as np
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.utils import common
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents import DqnAgent
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies import TFPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents import trajectories
import tensorflow.keras.optimizers.legacy as optimizers

from Env2048 import PyEnv2048

def compute_avg_return(
        env: PyEnvironment,
        pol: TFPolicy,
        num_episodes: int = 10,
) -> np.float32:
    """Runs episodes, calculates return.  Max return in CartPole-v0 is 200."""
    total_return = 0.0
    for i in range(num_episodes):
        episode_ret = 0.0
        time_step = env.reset()
        time_step = tf_batched_time_from_py_time(time_step)
        actions = []
        while not time_step.is_last():
            action_step = pol.action(time_step)
            actions.append(action_step.action.numpy()[0])
            while len(actions) > 10:
                actions.pop(0)
            repeat_check = sum([n == actions[0] for n in actions])
            if repeat_check == 10:
                break
            time_step = tf_batched_time_from_py_time(env.step(action_step.action))
            episode_ret += time_step.reward
        total_return += episode_ret
    if num_episodes > 0:
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    else:
        return None

def tf_batched_time_from_py_time(
        time_step: trajectories.TimeStep) -> trajectories.TimeStep:
    """Convert a PyEnv time step to a batched TFEnv time step.
    
    I am having a problem where, in the first time step after the reset()
    is called, the TFEnv is returning an observation with a single entry
    instead of the newly initialized board, but the PyEnv has the correct
    observation.  So, this will build the appropriate TFEnv time step from
    the PyEnv version, and give everything the appropriate batch dimension."""
    return trajectories.TimeStep(
        discount = tf.expand_dims(tf.constant(time_step.discount), axis=0),
        observation = tf.expand_dims(
                tf.constant(np.array(time_step.observation)), 
                axis=0,
        ),
        reward = tf.expand_dims(
                tf.constant(float(time_step.reward), dtype=np.float32), 
                axis=0,
        ),
        step_type = tf.expand_dims(tf.constant(time_step.step_type), axis=0),
    )

# Supress Warnings
tf.get_logger().setLevel('ERROR')

# 1. Environment
py_env = PyEnv2048()
tf_env = TFPyEnvironment(py_env)
py_env_2 = PyEnv2048()
eval_env = TFPyEnvironment(py_env_2)

# 2. Network/Model/Agent
model = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params = (1024, 256),
    # dropout_layer_params = (0.5, 0.5),
)

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.001, 10000, 0.1)
# lr = 1e-3
train_counter = tf.Variable(0)
agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network = model,
    optimizer = optimizers.Adam(learning_rate = lr),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter = train_counter,
)
agent.initialize()

# 3. Replay Buffer
# The first argument here is a description of the required format for the
#    description of one step of the agent in the environment. The pre-built
#    environments have an attribute that contains that information for us.
replay_buffer = TFUniformReplayBuffer (
    data_spec = agent.collect_data_spec,
    batch_size = 1,
    max_length = 10000, 
)

# 5. Policy
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())

# 7. Dataset
dataset = replay_buffer.as_dataset(
    sample_batch_size = 64,
    num_steps = 2, 
    num_parallel_calls = 10,
)
iterator = iter(dataset)
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

time_step = py_env.reset()
time_step = tf_batched_time_from_py_time(time_step)
action = random.randrange(4)
next_time_step = py_env.step(action)
next_time_step = tf_batched_time_from_py_time(next_time_step)
action = tf.expand_dims(tf.constant(action, dtype=np.int64), axis=0)
traj = trajectories.from_transition(
        time_step, 
        trajectories.PolicyStep(action = (action)), 
        next_time_step
)
replay_buffer.add_batch(traj)

print("Model Summary: ")
model.summary()

print("\nComputing Initial Average Return:")
print(compute_avg_return(py_env, eval_policy))

print("\nStarting Training:")
losses = []
actions = []
for i in range(20000):
    time_step = tf_batched_time_from_py_time(py_env.current_time_step())
    action_step = collect_policy.action(time_step)
    next_time_step = tf_batched_time_from_py_time(
            py_env.step(action_step.action)
    )
    traj = trajectories.from_transition(
            time_step, 
            trajectories.PolicyStep(action = (action)), 
            next_time_step
    )
    replay_buffer.add_batch(traj)
    experience, _ = next(iterator)
    loss = agent.train(experience)
    losses.append(loss.loss)
    while len(losses) > 500:
        losses.pop(0)
    # print(loss)
    if i % 200 == 199:
        if len(losses) > 0:
            loss = sum(losses) / len(losses)
        print(f"{i+1:5} {loss:.8f}", flush=True)
    if i % 1000 == 999:
        print(f"Average return after {int((i+1)/1000)}k steps: ", end="")
        print(compute_avg_return(py_env, eval_policy))

print()

print("Computing Final Average Return:")
print(compute_avg_return(py_env, eval_policy))
