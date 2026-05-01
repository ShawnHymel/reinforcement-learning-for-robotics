"""
Customized implementation of PPO. Originally taken from:
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py

Original author: Costa Huang
Changes by: Shawn Hymel
Date: April 30, 2026

Changes:
 * Replaced "__main__" into a train() function so this file can be imported into
   e.g. a Jupyter Notebook.
 * Added ability to import a custom environment (rather than having to register one in gymnasium).
 * Made the actor and critic network architectures adjustable hyperparameters (variable number of
   hidden layers and variable number of nodes per hidden layer).
 * Added value_clip parameter to control clipping in the value (critic) network rather than reusing
   clip_coef (which forced reward normalization in environments).
 * Moved approx. KL divergence detection to each minibatch loop, so that we can stop training
   early before waiting for the end of the epoch if the action distributions drift too much
 * Added feature to save model checkpoints (best current model) throughout the training process.
 * Switched to Path() for saving model files (instead of raw strings).
 * Replaced cleanrl_utils.evaluate with a self-contained evaluate() function to remove the external 
   dependency.
 * Reorganized and added comments to make the implementation easier to understand.
"""

# Import standard libraries
import os
import random
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

# Import third-party libraries
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

#-------------------------------------------------------------------------------
# Configuration (hyperparameters)

@dataclass
class PPOConfig:

    # General settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    checkpoint_interval: int | None = None
    """Iterations between checkpoints, None to disable checkpoint saves"""
    save_model: bool = False
    """whether to save the final model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    timestep: float | None = None
    """simulation timestep in seconds for real-time rendering"""

    # Network architectures
    actor_hidden_layers: int = 2
    """number of hidden layers in the actor network"""
    actor_hidden_size: int = 64
    """number of nodes in each hidden layer in the actor network"""
    critic_hidden_layers: int = 2
    """number of hidden layers in the critic network"""
    critic_hidden_size: int = 64
    """number of nodes in each hidden layer in the critic network"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient for each policy update"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    value_clip: float  = 10.0
    """Absolute bound on value prediction change per update"""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


#-------------------------------------------------------------------------------
# Agent

class Agent(nn.Module):
    """
    Wrapper for actor and critic networks.
    """
    def __init__(self, envs, config):
        """
        Initialize actor and critic networks.

        Args:
            envs (gym.vector.SyncVectorEnv): vectorized environment containing all parallel envs
            config (PPOConfig): settings
        """
        super().__init__()

        # Get observation and action space sizes
        obs_size = np.array(envs.single_observation_space.shape).prod()
        action_size = np.prod(envs.single_action_space.shape)

        # Critic: takes observation and outputs single number estimating the expected cumulative
        # reward starting from this particular state. Tanh keeps activations bounded [-1, 1].
        self.critic = build_mlp(
            input_size = obs_size,
            output_size = 1,
            num_hidden_layers = config.critic_hidden_layers,
            hidden_layer_size = config.critic_hidden_size,
            output_std = 1.0,
        )

        # Actor means: takes observation and outputs the means of Gaussian distributions over the 
        # various actions that should maximize the expected return from a given state (observation)
        self.actor_mean = build_mlp(
            input_size = obs_size,
            output_size = action_size,
            num_hidden_layers = config.actor_hidden_layers,
            hidden_layer_size = config.actor_hidden_size,
            output_std = 0.01,
        )
        
        # Actor log standard deviations: learnable log(standard deviation) of the action
        # distribution curves above. Use log() to keep the std_dev positive. High std_dev early
        # in the training process (explore new actions) then lower std_dev later in training
        # (focus on exploiting known-good actions). Start at 0 so exp(0) = 1.0 (wide exploration).
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_size))

    def get_value(self, x):
        """
        Use the critic network to predict the value: total future, discounted rewards in the current
        state given an observation (x).

        Args:
            x (torch.Tensor): observation vector used as input to the critic network

        Returns:
            Single element np.array with the value (total future, discounted rewards)
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Run the actor and critic networks on the given observation. Serves two purposes
        depending on whether an action is provided:
            1. During rollout collection (action=None): sample a new action from the actor's
               Gaussian distribution and return it along with its log probability, entropy,
               and the critic's value estimate.
            2. During PPO update (action provided): evaluate the log probability and entropy
               of a previously collected action under the current (updated) policy. This is
               what allows PPO to compute the probability ratio between old and new policies.

        Args:
            x (torch.Tensor): observation vector used as input to both networks
            action (torch.Tensor or None): if provided, evaluate this action under the current
                                           policy rather than sampling a new one

        Returns:
            action (torch.Tensor): sampled or provided action, shape (num_envs, action_dim)
            log_prob (torch.Tensor): log probability of the action under the current policy,
                                     used to compute the PPO clipping ratio
            entropy (torch.Tensor): entropy of the action distribution, used as exploration
                                    bonus in the loss function (higher = more exploratory)
            value (torch.Tensor): critic's estimate of total discounted future rewards from
                                  this observation, shape (num_envs, 1)
        """
        # Get predicted action means (one for each action) by feeding the observation array into
        # the actor network
        action_mean = self.actor_mean(x)

        # Expand log std to match the batch size of action_mean (broadcasting without copying memory)
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Get the standard deviations from the actor (separate, learnable parameters)
        action_std = torch.exp(action_logstd)

        # Construct probability distributions from the means and standard deviations
        probs = Normal(action_mean, action_std)

        # During rollout: select a probability from the distribution. Skip during the update process
        # so we can just get the log(probability of a given action) and entropy.
        if action is None:
            action = probs.sample()
        
        # Sum log probs and entropy across action dimensions. We assume each action dimension is
        # independent, so the joint probability of the complete action vector is the product of
        # individual probabilities (which becomes a sum in log space). This gives one log_prob
        # and one entropy value per environment rather than one per action dimension.
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


#-------------------------------------------------------------------------------
# Module-level functions

def make_env(env_id, idx, capture_video, run_name, gamma):
    """
    Factory function that returns a callable (thunk) which creates and configures a single
    gymnasium environment. SyncVectorEnv requires a list of callables rather than a list of
    env instances so that each parallel environment is created independently with its own state.
    Note that rewards are normalized in environments created with this function.
    
    Args:
        env_id (str): registered gymnasium environment ID (e.g. "BalanceBot-v0")
        idx (int): index of this environment in the parallel pool, used to determine
                   which env captures video (only idx=0 records)
        capture_video (bool): whether to record video of the first environment
        run_name (str): name of the current training run, used for video save path
        gamma (float): discount factor, passed to NormalizeReward for correct scaling

    Returns:
        callable: a thunk that creates and returns the configured environment when called
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear layer using orthogonal weight initialization, which PPO prefers
    over random initialization (e.g. LeCun, He) for more stable early training.

    Args:
        layer: the nn.Linear layer to initialize
        std (float): scaling factor for the orthogonal weight matrix. Use sqrt(2) for hidden layers,
                     1.0 for critic output, 0.01 for actor output (keeps initial actions near zero)
        bias_const (float): constant value to initialize all biases to
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    
    return layer

def build_mlp(input_size, output_size, num_hidden_layers, hidden_layer_size, output_std=1.0):
    """
    Helper function to build a simple multilayer perceptron (i.e. dense neural network).

    Args:
        input_size (int): number of inputs to the neural network
        output_size (int): number of outputs of the neural network
        num_hidden_layers (int): number of hidden layers
        hidden_layer_size (int): number of nodes in each hidden layer
        output_std (float): std scaling for the output layer weights
    """
    layers = []
    in_size = input_size

    # Build hidden layers
    for _ in range(num_hidden_layers):
        layers += [layer_init(nn.Linear(in_size, hidden_layer_size)), nn.Tanh()]
        in_size = hidden_layer_size

    # Add output layer
    layers += [layer_init(nn.Linear(in_size, output_size), std=output_std)]

    return nn.Sequential(*layers)

def evaluate(
    model_path,
    eval_episodes,
    Model,
    device,
    config,
    envs=None,
    make_env=None,
    env_id=None,
    run_name=None,
    gamma=None,
    max_steps=None,
):
    """
    Evaluate a saved policy by running it in the environment for a fixed number of episodes
    and returning the episodic returns. Used after training to measure final policy performance.

    Either pass in a pre-built vectorized env (envs) or provide make_env + env_id + run_name +
    gamma to have this function construct one automatically.

    Args:
        model_path (Path): path to the saved model file (.cleanrl_model)
        eval_episodes (int): number of complete episodes to run
        Model (nn.Module): agent class to instantiate (e.g. Agent)
        device (torch.device): device to run inference on (CPU or GPU)
        config (PPOConfig): configuration used to build the agent network architecture
        envs (SyncVectorEnv): optional pre-built vectorized env. If provided, make_env,
                              env_id, run_name, and gamma are ignored.
        make_env (callable): environment factory function (see make_env() above)
        env_id (str): registered gymnasium environment ID (e.g. "BalanceBot-v0")
        run_name (str): name of the current run, used for environment setup
        gamma (float): discount factor, passed to make_env for reward normalization
        max_steps (int): max number of steps per episode (if None, use config.num_steps)

    Returns:
        list[float]: episodic returns for each completed evaluation episode
    """
    # Use provided vectorized env or construct a single eval env from make_env
    owns_envs = False
    if envs is not None:
        eval_envs = envs
    elif make_env is not None:
        assert all(v is not None for v in [env_id, run_name, gamma]), \
            "env_id, run_name, and gamma are required when make_env is provided"
        eval_envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, run_name, gamma)])
        owns_envs = True
    else:
        raise ValueError("Either envs or make_env must be provided")

    # Define the max number of steps per episode
    max_steps = max_steps if max_steps is not None else config.num_steps

    # Load the saved weights into a fresh agent instance
    agent = Model(eval_envs, config).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Set agent to evaluation mode: disables dropout and batch normalization if present
    agent.eval()

    # Run until we have collected enough complete episodes
    episodic_returns = []
    obs, _ = eval_envs.reset()
    step = 0
    while len(episodic_returns) < eval_episodes:
        # Select actions without gradient tracking since we're not training
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))

        # Step the environment
        obs, _, _, _, infos = eval_envs.step(action.cpu().numpy())
        step += 1

        # Record the return when an episode finishes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_returns.append(info["episode"]["r"])

        # Force episode end if we've hit the step limit
        if step >= max_steps:
            # Mark episode as incomplete
            episodic_returns.append(float("nan"))

            # Reset environment
            obs, _ = eval_envs.reset()
            step = 0

    # Only close the env if we created it (don't close one the caller owns)
    if owns_envs:
        eval_envs.close()

    return episodic_returns

def train(config: PPOConfig, envs=None):
    """
    Train a PPO agent.

    Args:
        config (PPOConfig): Filled out PPOConfig object (see above for member documentation)
        envs (SyncVectorEnv): Pass in a vector of custom gym environments (if None, set 
                              config.env_id to choose an existing one)
    """
    # Compute batch size, minibatch size, and total number of iterations
    config.batch_size = int(config.num_envs * config.num_steps)
    config.minibatch_size = int(config.batch_size // config.num_minibatches)
    config.num_iterations = config.total_timesteps // config.batch_size

    # TEST
    print("START: train()")

    # Ensure that if a vector of custom environments is passed in, the num_envs parameter reflects
    # that number
    if envs is not None:
        assert envs.num_envs == config.num_envs, (
            f"Number of envs ({envs.num_envs}) does not match config.num_envs ({config.num_envs}). "
            f"Either update config.num_envs or reconstruct the vectorized env."
        )

    # Assign a name for the run
    run_name = f"{config.env_id}__{config.exp_name}__{config.seed}__{int(time.time())}"

    # If track is set, use Weights & Biases to log experiments
    if config.track:
        import wandb

        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TEST
    print("CHECKPOINT: config computed")

    # Initialize a log for TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in \
                                                 vars(config).items()])),
    )

    # TEST
    print("CHECKPOINT: writer initialized")

    # Initialize checkpoints
    checkpoint_dir = Path(f"runs/{run_name}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_mean_return = -float("inf")
    recent_returns = deque(maxlen=10)

    # Seed all random number generators for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Assign a device for computation (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    # If a pre-built vectorized env is provided, use it directly. Otherwise, construct parallel envs
    # using make_env() and the env_id in config.
    if envs is not None:
        render = True
        owns_envs = False
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env(config.env_id, i, config.capture_video, run_name, config.gamma)
             for i in range(config.num_envs)]
        )
        render = False
        owns_envs = True

    # Check that the environment supports a continuous action space
    assert isinstance(envs.single_action_space, gym.spaces.Box), \
        "only continuous action space is supported"

    # TEST
    print("CHECKPOINT: envs ready")

    # Create an agent (initialize actor/critic networks) and set optimizer function
    agent = Agent(envs, config).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # TEST
    print("CHECKPOINT: agent created")

    # Pre-allocate storage tensors for one full rollout of experience
    obs = torch.zeros((config.num_steps, config.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config.num_steps, config.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)

    # TEST
    print("CHECKPOINT: storage allocated")

    # Initialize training state: step counter, timer, and first observation
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(config.num_envs).to(device)

    # TEST
    print("CHECKPOINT: initial reset done")

    # %%%Test render directly before training loop
    if render:
        print("CHECKPOINT: attempting test render")
        print(f"  envs.envs[0] type: {type(envs.envs[0])}")
        print(f"  render_mode: {envs.envs[0].render_mode}")
        envs.envs[0].render()
        print("CHECKPOINT: test render complete")
        time.sleep(2)  # pause so we can see if the window appears
        print("CHECKPOINT: entering iteration loop")

    # Each iteration: run all environments for num_steps (restarting terminated or truncated
    # episodes as needed), collect data from observations (by having the agent execute actions
    # chosen by the actor network), and perform update_epochs backward passes to update the actor 
    # and critic networks
    for iteration in range(1, config.num_iterations + 1):
        # Annealing: gradually reduce the learning rate over the course of training. This allows for
        # large updates to the networks early in the learning process and avoid overshoots later on.
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / config.num_iterations
            lrnow = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # TEST
        print(f"CHECKPOINT: iteration {iteration} start")

        # Perform a rollout: collect experience by executing actions given by the actor network
        # in all of the simulated environments
        for step in range(0, config.num_steps):
            step_start = time.time()
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # TEST
            print(f"CHECKPOINT: step {step}")

            # Get actions (for each env) from the actor network
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step all environments forward one timestep with the chosen actions
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Render the current state if using a custom env with render_mode="human"
            if render:
                envs.envs[0].render()
                if config.timestep is not None:
                    slack = config.timestep - (time.time() - step_start)
                    if slack > 0:
                        time.sleep(slack)

            # Log information from final step if the episode was terminated/truncated
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                        # Track recent returns for best model checkpointing
                        if config.checkpoint_interval is not None:
                            recent_returns.append(episodic_return)

        # Compute returns for each timestep for each environment. Some terminology:
        #   value[t]: critic's predicted total, discounted rewards looking forward from step t
        #   return[t]: actual total, discounted rewards from step t onward (computed by walking
        #              backward through the rollout). Note: we use GAE here, which blends real
        #              rewards with critic estimates.
        #   advantage[t]: how much better the actual return was versus the critic's prediction
        #                 (return[t] - value[t])
        with torch.no_grad():

            # Bootstrap: use estimate from critic to get the estimated value from the next would-be
            # step (as we never actually executed that step in the rollout)
            next_value = agent.get_value(next_obs).reshape(1, -1)

            # Walk backward through the rollout to compute advantages at each step. An "advantage"
            # measures how much better the actual return was compared to the critic's predicted
            # value at that step.
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                # Use predicted value if at final step, otherwise use real value from rollout
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                # Temporal Difference (TD) error: actual reward at step t + discounted value of the
                # next state (as predicted by the critic) - what the critic predicted at step t
                # Simplified: estimate how off was the critic at predicting the value (expected, 
                # discounted total reward) versus the real thing at a given timestep. It's a one-
                # step approximation of the advantage.
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]

                # Compute estimated advantages by walking backward through the rollout. Use 
                # Generalized Advantage Estimate (GAE) to combine TD with Monte Carlo rollout
                lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam

            # Estimated discounted rewards from each step onward. We used GAE, so advantages were a 
            # mix of critic estimations and real rollout rewards (Monte Carlo). These are used as 
            # the critic's learning target (i.e. this is what you should have predicted) during 
            # backpropagation.
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Spend update_epochs number of epochs optimizing the actor and critic networks
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.update_epochs):

            # Go through samples in minibatches
            early_stop = False
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                # Get indices for the minibatch that slice the rollout tensors
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                # Get the log probabilities of the rollout actions and predicted values from the
                # critic (given the observations), under the new (currently being updated) policy.
                # Note that PyTorch is tracking the graph operations here, so the outputs can later
                # be used to compute a loss value and ultimately used to update the networks wrapped
                # in agent.
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], 
                    b_actions[mb_inds]
                )
                
                # Compute the ratios of the action probabilities under the new policy and those 
                # under the old policy. We are computing:
                # ratio = new_prob / old_prob = exp(log(new_prob) - log(old_prob))
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Disable gradient tracking in PyTorch (we're just doing evaluation)
                with torch.no_grad():
                    # Approximate KL Divergence (http://joschu.net/blog/kl-approx.html), which
                    # provides a measure of how different two probability distributions are
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()

                    # For logging: track what fraction of the minibatch had their ratio clipped
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                    # Stop if the update causes a too large change in action distributions to help
                    # prevent overshoot in training
                    if config.target_kl is not None and approx_kl > config.target_kl:
                        early_stop = True
                        break

                # Standardize the minibatch advantages to keep them on a similar scale for training
                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss: by changing the probabilities of the actions at given states, how
                # much advantage are we gaining or losing (given by the ratio). Note the negative
                # sign: we want to maximize the reward (PyTorch tries to minimize loss by default).
                pg_loss1 = -mb_advantages * ratio

                # Clipping: important in PPO. Limits the ratio to [1-clip_coef, 1+clip_coef], 
                # cutting off the gradient signal if the policy tries to change too aggressively. 
                # torch.max() picks the more conservative of the clipped and unclipped losses, 
                # preventing overshooting.
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: use mean squared error (MSE) between predicted values at given 
                # timesteps and returns at those steps (mix of critic estimations and real rollout 
                # rewards) as the loss function. Optionally clip the MSE to prevent large updates.
                # Unlike the policy/actor loss function, we want to minimize MSE here.
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    # Calculate raw square error between the predicted values and returns
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2

                    # Clipping: Prevent the critic from updating its parameters by limiting the
                    # old value to change by value_clip amount.
                    # CHANGED FROM CLEANRL! we use a separate value_clip parameter instead of the
                    # original clip_coef, which forced environments to normalize their rewards.
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.value_clip,
                        config.value_clip,
                    )

                    # Calculate MSE using the new clipped value
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # Without clipping, use the raw MSE as the loss function
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Averages the entropy (measure of action distribution spread) across the minibatch.
                # High entropy: actor is outputting wide distributions (exploration)
                # Low entropy: actor is outputting narrow distributions (confident, exploitation)
                entropy_loss = entropy.mean()

                # Compute a single loss for both actor and critic networks. It's a combination of
                # the policy loss (advantages), entropy (widths of action distributions), and value
                # loss (difference between predicted values and returns). The importance of entropy
                # and value loss can be adjusted by the ent_coef and vf_coef hyperparameters.
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                # Resets the gradients to zero
                optimizer.zero_grad()

                # Backpropagation: walk backward through the computation graph to compute gradients
                # for every parameter in both actor and critic networks
                loss.backward()

                # Clip (clamp) the gradients to prevent the norm (magnitude) from being larger than
                # the given max_grad_norm parameter amount.
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)

                # Perform gradient descent step: apply the parameter updates
                optimizer.step()

            # Break out of the full training loop of our action distributions will change too much
            if early_stop:
                break

        # Compute explained variance: measures how much the spread of the actual returns can be
        # accounted for by the critic's predictions (ideally start near 0 early in training and 
        # climb to 0.9+ for a well-trained critic).
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log training metrics to TensorBoard once per iteration
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Periodic checkpointing
        if config.checkpoint_interval is not None and iteration % config.checkpoint_interval == 0:

            # Save periodic checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_iter{iteration:04d}.cleanrl_model"
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"checkpoint saved to {checkpoint_path}")

            # Save best model if mean return has improved
            if len(recent_returns) > 0:
                mean_return = np.mean(recent_returns)
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    best_model_path = checkpoint_dir / "best_model.cleanrl_model"
                    torch.save(agent.state_dict(), best_model_path)
                    print(f"new best model saved (mean_return={mean_return:.2f}) to {best_model_path}")

            # Log mean return to TensorBoard
            if len(recent_returns) > 0:
                writer.add_scalar("charts/mean_return_10ep", np.mean(recent_returns), global_step)

    # Optionally save the final model
    if config.save_model:
        final_model_path = checkpoint_dir / f"{config.exp_name}_final.cleanrl_model"
        torch.save(agent.state_dict(), final_model_path)
        print(f"final model saved to {final_model_path}")

    # Print best model summary if checkpointing was enabled
    if config.checkpoint_interval is not None and len(recent_returns) > 0:
        print(f"best model achieved mean return of {best_mean_return:.2f}, "
              f"saved to {checkpoint_dir / 'best_model.cleanrl_model'}")

    # Prefer the best model (highest mean return during training) over the final model,
    # as the policy can sometimes degrade near the end of training due to overshooting.
    eval_model_path = None
    if config.checkpoint_interval is not None and len(recent_returns) > 0:
        eval_model_path = checkpoint_dir / "best_model.cleanrl_model"
    elif config.save_model:
        eval_model_path = checkpoint_dir / f"{config.exp_name}_final.cleanrl_model"

    # Evaluate the model by running it in the environment for eval_episodes episodes
    if eval_model_path is not None:
        # Evaluate using the same env setup as training
        if not owns_envs:
            # Use custom vectorized envs passed in by caller
            episodic_returns = evaluate(
                model_path = eval_model_path,
                eval_episodes = 10,
                Model = Agent,
                device = device,
                config = config,
                envs = envs,
            )
        else:
            # No custom env: construct a new one for evaluation
            episodic_returns = evaluate(
                model_path = eval_model_path,
                eval_episodes = 10,
                Model = Agent,
                device = device,
                config = config,
                make_env = make_env,
                env_id = config.env_id,
                run_name = f"{run_name}-eval",
                gamma = config.gamma,
            )

        # Log evaluation episode returns to TensorBoard
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        print(f"Evaluation complete: mean return = {np.mean(episodic_returns):.2f} "
            f"over {len(episodic_returns)} episodes")

        # Optionally upload the model to Hugging Face Hub (removed)
        if config.upload_model:
            print("WARNING: Hugging Face Hub upload is not supported in this implementation")
   
    # Clean up
    if owns_envs:
        envs.close()
    writer.close()

    # Return the best agent if checkpointing was enabled, otherwise return the final agent
    if config.checkpoint_interval is not None and len(recent_returns) > 0:
        best_model_path = checkpoint_dir / "best_model.cleanrl_model"
        agent.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        print(f"returning best model (mean_return={best_mean_return:.2f})")

    return agent

#-------------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    import tyro
    train(tyro.cli(PPOConfig))