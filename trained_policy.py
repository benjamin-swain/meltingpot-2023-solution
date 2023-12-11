from meltingpot.utils.policies.policy import Policy
from ray.rllib.policy.policy import Policy as RayPolicy
import dm_env
import numpy as np
import cv2
import random


def print(*args, **kwargs):
    pass

def downsample_observation(array: np.ndarray, scaled) -> np.ndarray:
    """Downsample image component of the observation.
    Args:
      array: RGB array of the observation provided by substrate
      scaled: Scale factor by which to downsaple the observation
    
    Returns:
      ndarray: downsampled observation  
    """
    
    frame = cv2.resize(
            array, (array.shape[0]//scaled, array.shape[1]//scaled), interpolation=cv2.INTER_AREA)
    return frame

def _downsample_single_timestep(timestep: dm_env.TimeStep, scale) -> dm_env.TimeStep:
    return timestep._replace(
        observation={k: downsample_observation(v, scale) if k == 'RGB' else v for k, v in timestep.observation.items()
        })


class TrainedPolicy(Policy):
    """
        Trained policy class where observation is a flattened 11x11x3 RGB image with values normalized between -1 and 1
    """
    def __init__(self, policy_id):
        policy_path = 'al_harvest_checkpoint/'
        self._policy = RayPolicy.from_checkpoint(policy_path)
        self.substrate_name = None
        self.policy_id = policy_id

    def initial_state(self):
        """ Called at the beginning of every episode """
        state = {}
        state['step_count'] = 0
        state['action'] = 0
        state['policy_state'] = self._policy.get_initial_state()
        state['last_actions'] = []

        return state
    
    def step(self, timestep, prev_state):
        """ Returns random actions according to spec """
        state = dict(prev_state)

        timestep = _downsample_single_timestep(timestep, 8)
        observation = timestep.observation
        obs = observation['RGB']


        def preprocess_image(image):
            return ((image / 127.5) - 1).flatten().astype(np.float32)
        
        obs = preprocess_image(obs)

        action, policy_state, _ = self._policy.compute_single_action(
            obs=obs,
            explore=True,
            # clip_action=False,
            # unsquash_action=True,
            # prev_action=prev_state['action'],
            # prev_reward=timestep.reward,
            # timestep=state['step_count'],
            # state=prev_state['policy_state']
            )
        
        # record historical data
        num_actions_to_store = 5
        state['last_actions'] = [action] + state['last_actions']
        state['last_actions'] = state['last_actions'][:num_actions_to_store]
        # prevent repetitive actions
        if len(state['last_actions']) == num_actions_to_store and len(set(state['last_actions'])) == 1:
            print('stopping repetitive action')
            repeated_action = state['last_actions'][0]
            possible_actions = [1, 2, 3, 4, 5, 6] # forward, backward, left, right
            if repeated_action in possible_actions:
                possible_actions.remove(repeated_action)
            action = random.choice(possible_actions)
            state['last_actions'][0] = action

        print('action', action)
        
        state['step_count'] += 1
        state['action'] = action
        state['policy_state'] = policy_state

        return action, state
    
    def close(self):
        """ Required by base class """
        pass
