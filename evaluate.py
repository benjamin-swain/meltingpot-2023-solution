from collections import defaultdict
import time
import numpy as np
import json
import os
import pandas as pd
import cv2
import uuid
from copy import deepcopy
import psutil
from meltingpot.utils.scenarios import population as population_lib
from meltingpot.utils.substrates import substrate as substrate_lib
import meltingpot
from meltingpot import substrate
from hard_code_al_harvest import HardCodeAlHarvestPolicy
from hard_code_pd_arena import HardCodePDArenaPolicy
from hard_code_clean_up import HardCodeCleanUpPolicy
from hard_code_territory import HardCodeTerritoryPolicy
from trained_policy import TrainedPolicy

# USER INPUT                                                                                       
scenario = 'allelopathic_harvest__open_0'                                   
render_world = True
report_score = True
downsample_render = False
# to record data for training MARWIL algo:
record_data = False
batch_output_dir = '/home/ben/Downloads/offline_data/'
num_agent_episodes = 1  
num_workers = 1


substrate_to_policy = {
    'allelopathic_harvest__open': HardCodeAlHarvestPolicy, #TrainedPolicy
    'prisoners_dilemma_in_the_matrix__arena': HardCodePDArenaPolicy, 
    'clean_up': HardCodeCleanUpPolicy,
    'territory__rooms': HardCodeTerritoryPolicy,
}

# allelopathic_harvest__open	
# allelopathic_harvest__open_0
# allelopathic_harvest__open_1
# allelopathic_harvest__open_2
# clean_up	
# clean_up_0
# clean_up_1
# clean_up_2
# clean_up_3
# clean_up_4
# clean_up_5
# clean_up_6
# clean_up_7
# clean_up_8
# prisoners_dilemma_in_the_matrix__arena	
# prisoners_dilemma_in_the_matrix__arena_0
# prisoners_dilemma_in_the_matrix__arena_1
# prisoners_dilemma_in_the_matrix__arena_2
# prisoners_dilemma_in_the_matrix__arena_3
# prisoners_dilemma_in_the_matrix__arena_4
# prisoners_dilemma_in_the_matrix__arena_5
# territory__rooms	
# territory__rooms_0
# territory__rooms_1
# territory__rooms_2
# territory__rooms_3

substrate_name = scenario[:-2] if scenario[len(scenario)-2] == '_' else scenario
roles = substrate.get_config(substrate_name).default_player_roles
policy_ids = [f"agent_{i}" for i in range(len(roles))]
hardcode_policy = substrate_to_policy[substrate_name]
names_by_role = defaultdict(list)
for i in range(len(policy_ids)):
    names_by_role[roles[i]].append(policy_ids[i])
factory = meltingpot.scenario.get_factory(scenario) if scenario in meltingpot.scenario.SCENARIOS else \
    meltingpot.substrate.get_factory(scenario)
population = {p_id: hardcode_policy(p_id) for p_id in policy_ids}
focal_population = population_lib.Population(
    policies=population,
    names_by_role=names_by_role,
    roles=factory.focal_player_roles()) if scenario in meltingpot.scenario.SCENARIOS else \
    population_lib.Population(
    policies=population,
    names_by_role=names_by_role,
    roles=factory.default_player_roles())
num_focal_agents = len(focal_population._roles)
num_episodes = num_agent_episodes // num_focal_agents
print(f'You asked for {num_agent_episodes} agent episodes. There are {num_focal_agents} focal agents per scenario episode, so running {num_episodes} scenario episodes')

def get_memory_usage():
    """Return the current process's memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

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

def _downsample_multi_timestep(timestep, scaled):
    return timestep._replace(
        observation=[{k: downsample_observation(v, scaled) if k == 'RGB' else v for k, v in observation.items()
        } for observation in timestep.observation])

def get_normalization_scores():
    mp2res = pd.read_feather('meltingpot-results-2.1.1.feather')
    substrate_names = [
        'clean_up',
        'territory__rooms',
        'prisoners_dilemma_in_the_matrix__arena',
        'allelopathic_harvest__open'
    ]
    res_df = mp2res[mp2res['substrate'].isin(substrate_names)]
    minmax_scores = {}
    for scenario in res_df['scenario'].unique():
        # if scenario in ['clean_up_0', 'clean_up_1']:
        #     continue
        # after update
        algo_results = res_df[res_df['scenario'] == scenario].groupby('mapla')['focal_per_capita_return'].agg(np.mean)
        max_score = algo_results.max()
        min_score = algo_results.min()
        # before update
        # max_score = scenario_results[scenario_results['mapla'] == 'exploiter_acb']['focal_per_capita_return'].mean()
        # min_score = scenario_results[scenario_results['mapla'] == 'random']['focal_per_capita_return'].mean()
        minmax_scores[scenario] = dict(
            max_score=max_score,
            min_score=min_score,
        )
    return minmax_scores

def print_scores(scenario_name, scores, minmax_scores):
  print(f"Results for {scenario_name=}")
  norm_scores = normalize_scores(scores, minmax_scores[scenario_name])
  mean_score = norm_scores.mean()
  print(f"{scores=} \n {mean_score=:0.3f} \n {norm_scores=}")
  return mean_score

def normalize_scores(scores, minmax_scores):
    max_score, min_score = minmax_scores['max_score'], minmax_scores['min_score']
    norm_scores = (scores - min_score) / ( max_score - min_score)
    return norm_scores

def add_to_sample_batch(sample_batches, timestep, actions, step_count, eps_id):
    """
    Sample batches are a list of dicts representing each agent's data which are saved to jsons 
    after every epside. Format:
    Key: type, always set to SampleBatch
    Key: obs, list of lists (collective_reward, ready_to_shoot, flattened image)
    Key: new_obs, current obs
    Key: actions, action taken to get from obs to new_obs
    Key: prev_actions, previous action (just assume 0 at the start)
    Key: rewards, current reward
    Key: prev_rewards, previous reward
    Key: terminateds, always False
    Key: truncateds, always False
    Key: eps_id, unix timestamp for the episode
    Key: t, step count
    """
    timestep = _downsample_multi_timestep(timestep, 8)
    _IGNORE_KEYS = ['WORLD.RGB', 'INTERACTION_INVENTORIES', 'NUM_OTHERS_WHO_CLEANED_THIS_STEP']
    observations = []
    for observation in timestep.observation:
        observations.append({
            key: value
            for key, value in observation.items()
            if key not in _IGNORE_KEYS
        })

    for agent_index, observation in enumerate(observations):

        obs = observation['RGB'].astype(int).tolist()

        # Append prev data before current data
        sample_batches[agent_index]['obs'].append(sample_batches[agent_index]['new_obs'][-1] if sample_batches[agent_index]['new_obs'] else obs)
        sample_batches[agent_index]['prev_actions'].append(sample_batches[agent_index]['actions'][-1] if sample_batches[agent_index]['actions'] else 0)
        sample_batches[agent_index]['prev_rewards'].append(sample_batches[agent_index]['rewards'][-1] if sample_batches[agent_index]['rewards'] else 0.0)

        sample_batches[agent_index]['new_obs'].append(obs)
        sample_batches[agent_index]['actions'].append(actions[agent_index])
        sample_batches[agent_index]['rewards'].append(timestep.reward[agent_index].item())
        sample_batches[agent_index]['terminateds'].append(False)
        sample_batches[agent_index]['truncateds'].append(False)
        sample_batches[agent_index]['eps_id'].append(eps_id)
        sample_batches[agent_index]['t'].append(step_count)

    assert set(sample_batches[0].keys()) == \
        set(["type", "obs", "new_obs", "actions", "prev_actions", "rewards", "prev_rewards", 
             "terminateds", "truncateds", "eps_id", "t"])

    return sample_batches


def render_rgb(rgb_frame, additional_frame=None, downsample=False):
    if downsample:
        rgb_frame = downsample_observation(rgb_frame, 8)
        if additional_frame is not None:
            additional_frame = downsample_observation(additional_frame, 8)
    
    rgb_frame_show = rgb_frame[:, :, ::-1]

    if additional_frame is not None:
        additional_frame_show = additional_frame[:, :, ::-1]
        # If the heights don't match
        if rgb_frame_show.shape[0] > additional_frame_show.shape[0]:
            diff = rgb_frame_show.shape[0] - additional_frame_show.shape[0]
            padding = ((diff // 2, diff - (diff // 2)), (0, 0), (0, 0))
            additional_frame_show = np.pad(additional_frame_show, padding, mode='constant', constant_values=0)
        elif rgb_frame_show.shape[0] < additional_frame_show.shape[0]:
            diff = additional_frame_show.shape[0] - rgb_frame_show.shape[0]
            padding = ((diff // 2, diff - (diff // 2)), (0, 0), (0, 0))
            rgb_frame_show = np.pad(rgb_frame_show, padding, mode='constant', constant_values=0)
        combined = np.concatenate((rgb_frame_show, additional_frame_show), axis=1) # Concatenate side by side
    else:
        combined = rgb_frame_show

    cv2.imshow('Combined Images', combined)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        assert 1==2

def run_episode(
    population: population_lib.Population,
    substrate: substrate_lib.Substrate) -> None:
    """Runs a population on a substrate for one episode."""
  
    start_time = time.time()  # Record the starting time
    step_count = 0  # Initialize step counter
  
    population.reset()
    timestep = substrate.reset()
    population.send_timestep(timestep)
    actions = population.await_action()

    focal_scores = np.zeros_like(timestep.reward)

    eps_id = int(time.time())
    initial_batch = {
        "type": 'SampleBatch',
        "obs": [],
        "new_obs": [],
        "actions": [],
        "prev_actions": [],
        "rewards": [],
        "prev_rewards": [],
        "terminateds": [],
        "truncateds": [],
        "eps_id": [],
        "t": [],
    }
    # Initialize sample batches (one per focal agent)
    sample_batches = [deepcopy(initial_batch) for _ in range(len(actions))]
  
    t_big = 0
    t_avg = -1
    max_mem = 0

    while not timestep.step_type.last():
        t1 = time.time()
        # timestep is a class containing reward & observation for each agent
        timestep = substrate.step(actions)
        population.send_timestep(timestep)
        # actions are a tuple of the agent actions
        actions = population.await_action()

        t2 = time.time()
        if t2-t1>t_big:
            t_big=t2-t1
        if t_avg==-1:
            t_avg = t2-t1
        else:
            t_avg = (t_avg + (t2-t1)) / 2

        mem_usage = get_memory_usage()
        if mem_usage > max_mem:
            max_mem = mem_usage

        if report_score:
            focal_scores += timestep.reward

        if record_data:
            sample_batches = add_to_sample_batch(sample_batches, timestep, actions, step_count, eps_id)

        if render_world:
            world_rgb = deepcopy(timestep.observation[0]['WORLD.RGB'])
            agent_rgb = deepcopy(timestep.observation[0]['RGB'])
            render_rgb(world_rgb, additional_frame=agent_rgb, downsample=downsample_render)
        
        step_count += 1  # Increment the step counter

    print('slowest step time', t_big)
    print('average step time', t_avg)
    print(f"highest memory usage {max_mem:.2f} MB")
    
    end_time = time.time()  # Record the ending time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Compute and print steps per second
    steps_per_second = step_count / elapsed_time
    print(f"Steps per second: {steps_per_second:.2f}") 
    print('Writing episode data...')
    print('num steps', step_count)

    if report_score:
        minmax_scores = get_normalization_scores()
        print_scores(scenario, focal_scores, minmax_scores)

    if not record_data:
        return None
    
    for agent_idx, batch in enumerate(sample_batches):
        json_filename = f"data_{scenario}_eps_{eps_id}_agent_{agent_idx}_uuid_{uuid.uuid4()}.json"
        output_json_path = os.path.join(batch_output_dir, scenario, json_filename)

        if not os.path.exists(os.path.dirname(output_json_path)):
            os.makedirs(os.path.dirname(output_json_path))
        
        # Write the JSON data directly to a file
        with open(output_json_path, 'w') as jsonf:
            json.dump(batch, jsonf)

    print(f'Epsiode data sample: {json_filename}')
    filtered_data = {key: value for key, value in sample_batches[0].items() if isinstance(value, list)}
    df = pd.DataFrame(filtered_data)
    print(df.head(100))
    

from concurrent.futures import ThreadPoolExecutor

def threaded_run_episode(n):
    roles = substrate.get_config(substrate_name).default_player_roles
    policy_ids = [f"agent_{i}" for i in range(len(roles))]
    if substrate_name == 'allelopathic_harvest__open':
        new_policy_ids = ['red_' + str(i) for i in range(int(len(policy_ids)/2))] + \
                         ['green_' + str(i) for i in range(int(len(policy_ids)/2))]
        policy_ids = new_policy_ids
    print('policy ids', policy_ids)
    print('roles', roles)
    hardcode_policy = substrate_to_policy[substrate_name]
    names_by_role = defaultdict(list)
    for i in range(len(policy_ids)):
        names_by_role[roles[i]].append(policy_ids[i])
    factory = meltingpot.scenario.get_factory(scenario) if scenario in meltingpot.scenario.SCENARIOS else \
        meltingpot.substrate.get_factory(scenario)
    population = {p_id: hardcode_policy(p_id) for p_id in policy_ids}
    print('names by role', names_by_role)
    print('roles:', factory.focal_player_roles() if scenario in meltingpot.scenario.SCENARIOS else factory.default_player_roles())

    focal_population = population_lib.Population(
        policies=population,
        names_by_role=names_by_role,
        roles=factory.focal_player_roles()) if scenario in meltingpot.scenario.SCENARIOS else \
        population_lib.Population(
        policies=population,
        names_by_role=names_by_role,
        roles=factory.default_player_roles())
    env = factory.build() if scenario in meltingpot.scenario.SCENARIOS else factory.build(factory.default_player_roles())
    run_episode(focal_population, env)
    print('%4d / %4d episodes completed...' % (n + 1, num_episodes))


start_memory_usage = get_memory_usage()

if record_data:
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(threaded_run_episode, range(num_episodes)))
else:   
    threaded_run_episode(1)

end_memory_usage = get_memory_usage()

print(f"Memory usage at start: {start_memory_usage:.2f} MB")
print(f"Memory usage at end: {end_memory_usage:.2f} MB")
