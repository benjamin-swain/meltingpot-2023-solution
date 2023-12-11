from meltingpot.utils.policies.policy import Policy
import cv2
import dm_env
import numpy as np
import random
import feature_detector_clean_up

def print(*args, **kwargs):
    pass

MEETUP_END_STEP = 35

DIRT_COLOR = (28, 152, 147)
APPLE_COLOR = (170, 153, 69)
WALL = (115, 115, 115)
WATER = (34, 129, 163)
WATER2 = (34, 129, 162)
WATER_DARK = (32, 121, 152)
WATER_NEAR_GRASS = (34, 129, 162)

COLOR_THRESHOLD = 5

AGENT_VIEW_SIZE = 11

# Create action mappings
ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

def get_neighbours(cell):
    """ Return the neighbours of the cell in the grid """
    i, j = cell
    return [(i + di, j + dj) for di, dj in ACTIONS.values() if 0 <= i + di < AGENT_VIEW_SIZE and 0 <= j + dj < AGENT_VIEW_SIZE]

def shortest_path_to_goal(start, goal, walls, bad_cells):
    """ Compute the shortest path from start to goal avoiding walls and bad cells """
    visited = set()
    queue = [(start, [])]
    while queue:
        (i, j), path = queue.pop(0)
        if (i, j) == goal:
            return path
        if (i, j) in visited:
            continue
        visited.add((i, j))
        for neighbour in get_neighbours((i, j)):
            if neighbour not in walls and neighbour not in bad_cells and neighbour not in visited:
                queue.append((neighbour, path + [neighbour]))
    return []

def get_direction_to_goal(goal, walls, bad_cells):
    start = (9, 5)
    path = shortest_path_to_goal(start, goal, walls, bad_cells)
    if not path:
        return None
    next_cell = path[0]
    for action, (di, dj) in ACTIONS.items():
        if (start[0] + di, start[1] + dj) == next_cell:
            return action
    return None
    
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

def direction_to_number(direction):
    mapping = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "same_location": 0
    }
    return mapping.get(direction, 0)

def get_nearest(pixel_positions):
    pixel_positions = [pixel for pixel in pixel_positions if pixel != (9, 5)]
    if len(pixel_positions) == 0:
        return None
    pixels = pixel_positions
    # Splitting the list into rows and cols arrays
    rows = np.array([pixel[0] for pixel in pixels])
    cols = np.array([pixel[1] for pixel in pixels])
    # Calculate distances to the point (9, 5)
    distances = np.sqrt((rows - 9)**2 + (cols - 5)**2)
    # Find the pixel with the minimum distance
    min_index = np.argmin(distances)
    # Extract the nearest pixel's coordinates
    nearest_pixel = pixels[min_index]
    return nearest_pixel
    
def adjust_apple_direction(turn_direction, apple_direction):
    
    # Define a mapping for the left turn (90 degrees CW)
    left_turn_map = {
        "left": "up",
        "up": "right",
        "right": "down",
        "down": "left",
        "unknown": "unknown"
    }
    
    # Define a mapping for the right turn
    right_turn_map = {
        "left": "down",
        "up": "left",
        "right": "up",
        "down": "right",
        "unknown": "unknown"
    }
    
    # Check the turn_direction and adjust the apple_direction accordingly
    if turn_direction == 5:
        return left_turn_map[apple_direction]
    elif turn_direction == 6:
        return right_turn_map[apple_direction]
    else:
        raise ValueError("Invalid turn_direction value. It must be either 5 or 6.")
    
def reverse_direction(direction):
    mapping = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
        "same_location": "same_location"
    }
    return mapping.get(direction, "same_location")

def is_within_range(a, b, threshold=COLOR_THRESHOLD):
    within_threshold = np.all(np.abs(a - b) <= threshold)
    return within_threshold

def any_within_range(color, color_list):
    for color_i in color_list:
        if is_within_range(color, color_i):
            return True
    return False

def change_goal_based_on_conditions(state, apples):
    if state['solo_focal']:
        # If haven't found apples for a while after looking in the grass, go clean dirt
        if state['goal_color'] == APPLE_COLOR and state['num_steps_without_finding_apples_in_grass'] > 3 and not apples \
            and state['step_count'] - state['last_goal_change_step'] > 15:
            # print('set goal=DIRT because too many steps without finding apple')
            state['goal_color'] = DIRT_COLOR
            state['last_goal_change_step'] = state['step_count']
        # If haven't found dirt for a while after looking in the water, go find apples
        if state['goal_color'] == DIRT_COLOR and state['num_steps_without_finding_dirt_in_water'] > 20 \
            and state['step_count'] - state['last_goal_change_step'] > 15:
            # print('set goal=APPLE because too many steps without finding dirt')
            state['goal_color'] = APPLE_COLOR
            state['last_goal_change_step'] = state['step_count']
        # If seen 2 or more other players cleaning in the past 20 steps, go find apples
        if state['goal_color'] == DIRT_COLOR and two_or_more_others_cleaning(state) \
            and state['step_count'] - state['last_goal_change_step'] > 15:
            # print('set goal=APPLE because 2 or more other players are cleaning')
            state['goal_color'] = APPLE_COLOR
            state['last_goal_change_step'] = state['step_count']
    return state['goal_color'], state['last_goal_change_step']

def two_or_more_others_cleaning(state):
    num_others_cleaning_past_20_steps = 0
    attrs_to_check = [
        'steps_since_yellow_seen_in_water', 
        'steps_since_purple_seen_in_water', 
        'steps_since_red_seen_in_water',
        'steps_since_orange_seen_in_water', 
        'steps_since_green_seen_in_water', 
        'steps_since_teal_seen_in_water',
        'steps_since_pink_seen_in_water', 
        'steps_since_light_purple_seen_in_water', 
        'steps_since_mint_seen_in_water',
        'steps_since_lime_green_seen_in_water', 
        'steps_since_dark_blue_seen_in_water', 
        'steps_since_light_red_seen_in_water',
        'steps_since_brown_seen_in_water', 
        'steps_since_dark_orange_seen_in_water', 
        'steps_since_dark_teal_seen_in_water'
    ]
    for attr in attrs_to_check:
        if state[attr] <= 20:
            num_others_cleaning_past_20_steps += 1
    if num_others_cleaning_past_20_steps >= state['num_other_cleaners_needed_to_harvest']:
        return True
    return False

def update_steps_without_finding_x(state):
    if state['solo_focal']:
        if state['goal_color'] == DIRT_COLOR:
            state['num_steps_without_finding_apples_in_grass'] = 0
        else:
            state['num_steps_without_finding_dirt_in_water'] = 0
        state['steps_since_yellow_seen_in_water'] += 1
        state['steps_since_purple_seen_in_water'] += 1
        state['steps_since_red_seen_in_water'] += 1
        state['steps_since_orange_seen_in_water'] += 1
        state['steps_since_green_seen_in_water'] += 1
        state['steps_since_teal_seen_in_water'] += 1
        state['steps_since_pink_seen_in_water'] += 1
        state['steps_since_light_purple_seen_in_water'] += 1
        state['steps_since_mint_seen_in_water'] += 1
        state['steps_since_lime_green_seen_in_water'] += 1
        state['steps_since_dark_blue_seen_in_water'] += 1
        state['steps_since_light_red_seen_in_water'] += 1
        state['steps_since_brown_seen_in_water'] += 1
        state['steps_since_dark_orange_seen_in_water'] += 1
        state['steps_since_dark_teal_seen_in_water'] += 1

    return state

def apple_direction_to_meetup_direction(apple_direction):
    if apple_direction == 'left':
        meetup_direction = 'up'
    elif apple_direction == 'up':
        meetup_direction = 'right'
    elif apple_direction == 'right':
        meetup_direction = 'down'
    elif apple_direction == 'down':
        meetup_direction = 'left'
    else:
        meetup_direction = 'up'
    return meetup_direction

def get_meetup_agent_spot(img, sand):
    """given RGB image, identify WALL pixels with a SAND pixel below it. return the one with the max column value"""
    max_col = -1
    max_row = -1

    # Iterate through the image
    for row in range(img.shape[0] - 1):  # We subtract 1 to avoid checking the last row as there's no row below it
        for col in range(img.shape[1]):
            current_pixel = img[row, col]
            below_pixel_loc = (row + 1, col)

            # If pixel is wall and below pixel is sand
            if (np.array_equal(current_pixel, WALL) and below_pixel_loc in sand):
                if col > max_col:
                    max_col = col
                    max_row = row

    if max_col == -1:
        return None
    # Return the sand pixel nearest to water
    return (max_row+1, max_col)

def get_apple_meetup_agent_spot(sand, grass, water):
    """given RGB image, identify SAND pixels with a GRASS pixel above it and a WATER pixel to the left of it. 
    return the one with the minimum column value"""
    # Convert pixel location lists to sets for faster lookup
    sand_set = set(sand)
    grass_set = set(grass)
    water_set = set(water)
    
    # Initialize variables to store the result
    result = None
    min_column = float('inf')
    
    # Iterate through each pixel in the sand list
    for x, y in sand_set:
        # Check if there is a grass pixel directly above the current sand pixel
        if (x-1, y) in grass_set:
            # Check if there are no water pixels on any column of the same row to the right of the current sand pixel
            if all((x, col) not in water_set for col in range(y+1, 11)):
                # Update result if current sand pixel has a smaller column value
                if y < min_column:
                    result = (x, y)
                    min_column = y
    
    return result
    
def agent_at_meetup_spot(img, sand):
    # Check the pixel at (8, 5)
    pixel_8_5 = img[8, 5]
    if not np.array_equal(pixel_8_5, WALL):
        return False
    # Check the pixel at (9, 6)
    if (9, 6) in sand:
        return False
    return True

def agent_at_apple_meetup_spot(sand, grass):
    # Check the pixel in front 
    if not (8, 5) in grass:
        return False
    # Check the pixel to left
    if (9, 4) in sand:
        return False
    return True

def average_pixel_change(images):
    """Average change in images. Must be 2 or more images to prevent error"""
    total_changes = 0

    # Iterate through successive pairs of images
    for i in range(1, len(images)):
        # Compute the absolute difference between the two images
        diff = np.abs(images[i] - images[i-1])

        # Count the number of pixels that have changed
        changed_pixels = np.sum(diff != 0) / 3  # Dividing by 3 because of the 3 color channels

        # Update the total changes
        total_changes += changed_pixels

    # Calculate the average percentage of pixel changes
    avg_percentage = (total_changes / (11 * 11 * (len(images) - 1))) * 100
    return avg_percentage

def detect_friendlies(players, player_colors, player_directions, walls, sand):
    friendly_colors = []
    for i, pos in enumerate(players):
        if player_directions[i] == 'up' and \
            (pos[0]-1, pos[1]) in walls and \
                (pos[0]+1, pos[1]) in sand:
            friendly_colors.append(player_colors[i])
    return friendly_colors

def get_death_zones_v2(players, player_colors, player_directions, friendly_colors):
    """Define locations where enemy zaps can reach"""
    death_zones = []
    for i, loc in enumerate(players):
        dir = player_directions[i]
        color = player_colors[i]
        if color in friendly_colors:
            continue
        if dir == 'unknown':
            continue
        zap_obstacles = players
        zap_locs = get_expected_zap_locations(loc, dir, zap_obstacles, include_off_screen=True)
        death_zones = death_zones + zap_locs
    return death_zones

def get_expected_zap_locations(player_pos, direction, zap_obstacles, include_off_screen=False):
    """Return the pixels expected to have the zap color based on player pos, direction, and obstacles"""
    expected_zap_locations = []
    if direction == 'left':
        # first list right beam, then left beam, then center beam
        pixels_to_check = [(1, 0), (1, -1), (1, -2),
                           (-1, 0), (-1, -1), (-1, -2), 
                           (0, -1), (0, -2), (0, -3)]
    elif direction == 'right':
        pixels_to_check = [(1, 0), (1, 1), (1, 2),
                           (-1, 0), (-1, 1), (-1, 2),
                           (0, 1), (0, 2), (0, 3)]
    elif direction == 'up':
        pixels_to_check = [(0, 1), (-1, 1), (-2, 1),
                           (0, -1), (-1, -1), (-2, -1),
                           (-1, 0), (-2, 0), (-3, 0)]
    elif direction == 'down':
        pixels_to_check = [(0, -1), (1, -1), (2, -1),
                           (0, 1), (1, 1), (2, 1),
                           (1, 0), (2, 0), (3, 0)]
    else:
        return []

    expected_locs = [(player_pos[0]+dy, player_pos[1]+dx) for dy, dx in pixels_to_check]

    # right beam
    expected_zap_locations.append(expected_locs[0])
    if expected_locs[0] not in zap_obstacles:
        expected_zap_locations.append(expected_locs[1])
        if expected_locs[1] not in zap_obstacles:
            expected_zap_locations.append(expected_locs[2])
    # left beam
    expected_zap_locations.append(expected_locs[3])
    if expected_locs[3] not in zap_obstacles:
        expected_zap_locations.append(expected_locs[4])
        if expected_locs[4] not in zap_obstacles:
            expected_zap_locations.append(expected_locs[5])
    # center beam
    expected_zap_locations.append(expected_locs[6])
    if expected_locs[6] not in zap_obstacles:
        expected_zap_locations.append(expected_locs[7])
        if expected_locs[7] not in zap_obstacles:
            expected_zap_locations.append(expected_locs[8])

    if not include_off_screen:
        expected_zap_locations = [loc for loc in expected_zap_locations if loc[0] >= 0 and \
                                  loc[0] <= 10 and loc[1] >= 0 and loc[1] <= 10]
    return expected_zap_locations

def get_nearest_list(pixel_positions, from_pos=(9, 5)):
    # Filter out the point (9, 5) if it's in the list
    pixel_positions = [pixel for pixel in pixel_positions if pixel != from_pos]
    
    if len(pixel_positions) == 0:
        return None
    
    # Splitting the list into rows and cols arrays
    rows = np.array([pixel[0] for pixel in pixel_positions])
    cols = np.array([pixel[1] for pixel in pixel_positions])
    
    # Calculate distances to the point (9, 5)
    distances = np.sqrt((rows - from_pos[0])**2 + (cols - from_pos[1])**2)
    
    # Sort pixels by distance
    sorted_indices = np.argsort(distances)
    sorted_pixel_positions = [pixel_positions[i] for i in sorted_indices]
    
    return sorted_pixel_positions

def stepping_into_death(action, death_zones):
    if action_to_enum(action) == 'FORWARD' and (8, 5) in death_zones:
        return True
    if action_to_enum(action) == 'BACKWARD' and (10, 5) in death_zones:
        return True
    if action_to_enum(action) == 'STEP_LEFT' and (9, 4) in death_zones:
        return True
    if action_to_enum(action) == 'STEP_RIGHT' and (9, 6) in death_zones:
        return True
    if action_to_enum(action) in \
        ['NOOP', 'FIRE_CLEAN', 'TURN_LEFT', 'TURN_RIGHT'] and (9, 5) in death_zones:
        return True
    return False

def action_to_enum(action):
    action_set = (
    'NOOP',
    'FORWARD',
    'BACKWARD',
    'STEP_LEFT',
    'STEP_RIGHT',
    'TURN_LEFT',
    'TURN_RIGHT',
    'FIRE_ZAP',
    'FIRE_CLEAN'
    )
    return action_set[action]

def evade_enemy_action(action, death_zones, obstacles):
    potential_actions = [1, 3, 4, 2, 0] # forward, left, right, backward, noop
    if action in potential_actions:
        potential_actions.remove(action)
    for act in potential_actions:
        if not stepping_into_death(act, death_zones) and not stepping_into_death(act, obstacles):
            return act
    return None

def player_in_grass(grass, apples):
    """If the player is in grass, the player should have either grass or apple on two sides"""
    front_grass = (8, 5) in grass or (8, 5) in apples
    left_grass = (9, 4) in grass or (9, 4) in apples
    right_grass = (9, 6) in grass or (9, 6) in apples
    back_grass = (10, 5) in grass or (10, 5) in apples
    if sum([front_grass, left_grass, right_grass, back_grass]) >= 2:
        return True
    return False

def get_directional_goal(direction):
    if direction == 'left': 
        return (9, 2)
    if direction == 'right':
        return (9, 8)
    if direction == 'up':
        return (6, 5)
    return (10, 5)

    
class HardCodeCleanUpPolicy(Policy):
    """
        Hardcoded (rule-based) policy for clean_up substrate
    """
    def __init__(self, policy_id):
        self.substrate_name = None
        self.policy_id = policy_id

    def initial_state(self):
        """ Called at the beginning of every episode """
        state = {
            'step_count': 0,
            'last_goal_change_step': 0,
            'goal_color': DIRT_COLOR,
            'num_steps_without_finding_apples_in_grass': 0,
            'num_steps_without_finding_dirt_in_water': 0,
            'apple_direction': 'unknown',
            'last_actions': [],
            'last_images': [],
            'solo_focal': False,  
            'steps_since_yellow_seen_in_water': 999,
            'steps_since_purple_seen_in_water': 999,
            'steps_since_red_seen_in_water': 999,
            'steps_since_orange_seen_in_water': 999,
            'steps_since_green_seen_in_water': 999,
            'steps_since_teal_seen_in_water': 999,
            'steps_since_pink_seen_in_water': 999,
            'steps_since_light_purple_seen_in_water': 999,
            'steps_since_mint_seen_in_water': 999,
            'steps_since_lime_green_seen_in_water': 999,
            'steps_since_dark_blue_seen_in_water': 999,
            'steps_since_light_red_seen_in_water': 999,
            'steps_since_brown_seen_in_water': 999,
            'steps_since_dark_orange_seen_in_water': 999,
            'steps_since_dark_teal_seen_in_water': 999,
            'total_reward': 0,
            'num_other_cleaners_needed_to_harvest': 2,
            'friendly_colors': [],
            'steps_since_turn_toward_enemy': 0,
            'death_zones': [],
            'players': [],
            'walls': []
        }
        return state
    
    def step(self, timestep, prev_state):
        """ Returns random actions according to spec """
        state = dict(prev_state)
        state['step_count'] += 1

        print('\ntimestep', state['step_count'])

        state['steps_since_turn_toward_enemy'] += 1

        raw_timestep = timestep
        raw_observation = raw_timestep.observation['RGB']

        timestep = _downsample_single_timestep(timestep, 8)
        observation = timestep.observation['RGB']

        action, state = self.custom_step(observation, timestep, raw_observation, state)

        if stepping_into_death(action, state['death_zones']) and state['step_count'] > 100:
            print('overriding action', action_to_enum(action), 'because it could lead to death')
            # choose the first action that is not obstacle or death zone
            new_action = evade_enemy_action(action, state['death_zones'], state['players']+state['walls'])
            if new_action:
                print('new action', action_to_enum(new_action))
                action = new_action
            else:
                print('no hope')

        # record historical data
        num_actions_to_store = 30
        state['last_actions'] = [action] + state['last_actions']
        state['last_actions'] = state['last_actions'][:num_actions_to_store]
        num_images_to_store = 5
        state['last_images'] = [timestep.observation['RGB']] + state['last_images']
        state['last_images'] = state['last_images'][:num_images_to_store]

        # If image hasn't changed much in the past 5 steps, send a random action to get unstuck
        if state['step_count'] > 150 and len(state['last_images']) == num_images_to_store and \
            all(isinstance(item, np.ndarray) for item in state['last_images']):
            avg_pixel_change = average_pixel_change(state['last_images'])
            if avg_pixel_change <= 1.0:
                repeated_action = state['last_actions'][0]
                possible_actions = [1, 2, 3, 4] # forward, backward, left, right
                if repeated_action in possible_actions:
                    possible_actions.remove(repeated_action)
                action = random.choice(possible_actions)
                state['last_actions'][0] = action

        # adjust apple direction if actor turns or is killed (black screen)
        if action in [5, 6]: 
            state['apple_direction'] = adjust_apple_direction(action, state['apple_direction'])
        if np.all(observation == [0, 0, 0]):
            state['apple_direction'] = 'unknown'

        return action, state
    
    
    def custom_step(self, observation, timestep, raw_observation, state):

        state['total_reward'] = state['total_reward'] + int(timestep.observation['COLLECTIVE_REWARD'])
        
        feature_dict = feature_detector_clean_up.process_observation(observation, raw_observation)
        # Retrieve each feature list from the dictionary
        apples = feature_dict['apples']
        dirt = feature_dict['dirt']
        walls = feature_dict['walls']
        sand = feature_dict['sand']
        grass = feature_dict['grass']
        water = feature_dict['water']
        players = feature_dict['players']
        player_colors = feature_dict['player_colors']
        player_directions = feature_dict['player_directions']
        water_player_colors = feature_dict['water_player_colors']
        apple_direction = feature_dict['apple_direction']
        away_from_wall_direction = feature_dict['away_from_wall_direction']
        
        print('player colors', player_colors)
        print('player direct', player_directions)
        print('player positi', players)

        if 'yellow' in water_player_colors:
            state['steps_since_yellow_seen_in_water'] = 0
        if 'purple' in water_player_colors:
            state['steps_since_purple_seen_in_water'] = 0
        if 'red' in water_player_colors:
            state['steps_since_red_seen_in_water'] = 0
        if 'orange' in water_player_colors:
            state['steps_since_orange_seen_in_water'] = 0
        if 'green' in water_player_colors:
            state['steps_since_green_seen_in_water'] = 0
        if 'teal' in water_player_colors:
            state['steps_since_teal_seen_in_water'] = 0
        if 'pink' in water_player_colors:
            state['steps_since_pink_seen_in_water'] = 0
        if 'light_purple' in water_player_colors:
            state['steps_since_light_purple_seen_in_water'] = 0
        if 'mint' in water_player_colors:
            state['steps_since_mint_seen_in_water'] = 0
        if 'lime_green' in water_player_colors:
            state['steps_since_lime_green_seen_in_water'] = 0
        if 'dark_blue' in water_player_colors:
            state['steps_since_dark_blue_seen_in_water'] = 0
        if 'light_red' in water_player_colors:
            state['steps_since_light_red_seen_in_water'] = 0
        if 'brown' in water_player_colors:
            state['steps_since_brown_seen_in_water'] = 0
        if 'dark_orange' in water_player_colors:
            state['steps_since_dark_orange_seen_in_water'] = 0
        if 'dark_teal' in water_player_colors:
            state['steps_since_dark_teal_seen_in_water'] = 0

        death_zones = get_death_zones_v2(players, player_colors, player_directions, state['friendly_colors'])

        state['death_zones'] = death_zones
        state['players'] = players
        state['walls'] = walls
        
        state['goal_color'], state['last_goal_change_step'] = \
            change_goal_based_on_conditions(state, apples)
        
        state = update_steps_without_finding_x(state)

        if apple_direction != 'unknown':
            state['apple_direction'] = apple_direction

        # Navigate to corner at start to count number of focals and divide tasks
        if not state['solo_focal'] and state['step_count'] < MEETUP_END_STEP:
            if state['apple_direction'] != 'unknown':
                print('going to initial meetup')
                meetup_wall_direction = apple_direction_to_meetup_direction(state['apple_direction'])
                # turn to face meetup direction
                if meetup_wall_direction == 'left':
                    return 5, state
                if meetup_wall_direction == 'down':
                    return 5, state
                if meetup_wall_direction == 'right':
                    return 6, state
                if agent_at_meetup_spot(observation, sand):
                    return 0, state
                # get agent spot along wall
                agent_meetup_spot = get_meetup_agent_spot(observation, sand)
                if agent_meetup_spot:
                    nonsand_pixels = [(i, j) for i in range(observation.shape[0]) for j in range(observation.shape[1]) if (i, j) not in sand]
                    direction = get_direction_to_goal(agent_meetup_spot, walls+nonsand_pixels+players, [])
                    if direction:
                        return direction_to_number(direction), state
                # navigate until wall is visible
                return direction_to_number(meetup_wall_direction), state
            else:
                print('failed to detect apple direction. cannot navigate to focal meetup')
        if not state['solo_focal'] and state['step_count'] == MEETUP_END_STEP:
            if agent_at_meetup_spot(observation, sand):
                # If player is next to water, set permanent goal to clean
                if any_within_range(observation[9, 6], [WATER, WATER2, WATER_DARK, WATER_NEAR_GRASS, DIRT_COLOR]):
                    state['solo_focal'] = True
                    state['goal_color'] = DIRT_COLOR
                elif any_within_range(observation[9, 7], [WATER, WATER2, WATER_DARK, WATER_NEAR_GRASS, DIRT_COLOR]):
                    print('adding 2nd cleaner')
                    state['solo_focal'] = True
                    state['goal_color'] = DIRT_COLOR
                    state['num_other_cleaners_needed_to_harvest'] = 3
                # otherwise go harvest
                else:
                    state['goal_color'] = APPLE_COLOR
                # record friendly colors
                state['friendly_colors'] = detect_friendlies(players, player_colors, player_directions, walls, sand)
            else:
                # player did not make it to the meetup spot, so operate independently
                state['solo_focal'] = True
                state['goal_color'] = DIRT_COLOR

        enemies = [pos for i, pos in enumerate(players) if player_colors[i] not in state['friendly_colors']]
            
        # If you're in grass, face the sand to prepare for incoming enemies
        if not enemies and player_in_grass(grass, apples) and \
            (not apples or any(apple_y >= 4 and apple_x >= 4 and apple_x <=6 for apple_y, apple_x in apples)):
            if state['apple_direction'] == 'up':
                return 6, state
            if state['apple_direction'] == 'left':
                return 6, state
            if state['apple_direction'] == 'right':
                return 5, state

        # Clean dirt
        if state['goal_color'] == DIRT_COLOR:
            if len(dirt) > 0:
                print('Going to dirt')
                nearest_dirt = get_nearest(dirt)
                # If dirt in front of you (8, 5), clean it
                if nearest_dirt == (8, 5):
                    state['num_steps_without_finding_dirt_in_water'] = 0
                    return 8, state
                # If dirt to left of you (9, 4), turn left
                if nearest_dirt == (9, 4):
                    return 5, state
                # If dirt to right of you (9, 6), turn right
                if nearest_dirt == (9, 6):
                    return 6, state
                # if dirt behind you, turn around
                if nearest_dirt == (10, 5):
                    return 6, state
                # If dirt behind and to the left, turn left
                if nearest_dirt[0] == 10 and nearest_dirt[1] < 5:
                    return 5, state
                # If dirt behind and to the right, turn right
                if nearest_dirt[0] == 10 and nearest_dirt[1] > 5:
                    return 6, state
                
                # To get the player just in front of the dirt, 
                nearest_dirt = (nearest_dirt[0]+1, nearest_dirt[1])
                direction = get_direction_to_goal(nearest_dirt, walls+players, [])
                if direction:
                    return direction_to_number(direction), state
                
            # print('Cannot see dirt')
            
            # record how many steps player is searching for dirt while in water
            nearest_water = get_nearest(water)
            dist_to_nearest_water = np.sqrt((nearest_water[0] - 9)**2 + (nearest_water[1] - 5)**2) if nearest_water else None
            if nearest_water:
                if dist_to_nearest_water < 1.5:
                    state['num_steps_without_finding_dirt_in_water'] += 1
   
            # if apple direction is known, go in opposite direction
            if state['apple_direction'] != 'unknown':
                if dist_to_nearest_water == None or dist_to_nearest_water > 1.5:
                    # print('Going toward water to find dirt')
                    goal_loc = get_directional_goal(reverse_direction(state['apple_direction']))
                    direction = get_direction_to_goal(goal_loc, [obst for obst in walls+players if obst != goal_loc], [])
                    if direction:
                        return direction_to_number(direction), state
        
        # Harvest apples
        if state['goal_color'] == APPLE_COLOR:

            nearest_apples = get_nearest_list(apples)
            if nearest_apples:
                for nearest_apple in nearest_apples:
                    state['num_steps_without_finding_apples_in_grass'] = 0
                    direction = get_direction_to_goal(nearest_apple, walls+players, [])
                    if direction:
                        return direction_to_number(direction), state

            # record how many steps player is searching for apples while in grass
            nearest_grass = get_nearest(grass)
            dist_to_nearest_grass = np.sqrt((nearest_grass[0] - 9)**2 + (nearest_grass[1] - 5)**2) if nearest_grass else None
            if nearest_grass:
                if dist_to_nearest_grass < 1.5:
                    state['num_steps_without_finding_apples_in_grass'] += 1
  
            # if cannot find apples, line up at the corner of grass, water, and sand
            if state['apple_direction'] != 'unknown':
                # turn to face meetup direction
                if state['apple_direction'] == 'left':
                    print('turning to face apple meetup direction')
                    return 5, state
                if state['apple_direction'] == 'down':
                    print('turning to face apple meetup direction')
                    any_walls_on_left = any(wall_y >= 9 and wall_x < 5 for wall_y, wall_x in walls)
                    if any_walls_on_left:
                        return 6, state # turn right
                    return 5, state # turn left
                if state['apple_direction'] == 'right':
                    print('turning to face apple meetup direction')
                    return 6, state
                if agent_at_apple_meetup_spot(sand, grass):
                    if state['step_count'] == 100 and state['total_reward'] == 0 and (9, 4) in water:
                        print('adding cleaner because no apples to harvest')
                        state['solo_focal'] = True
                        state['goal_color'] = DIRT_COLOR
                        state['num_other_cleaners_needed_to_harvest'] = 3
                        return 5, state
                    # print('agent at apple meetup spot')
                    # test
                    return 0, state
                # get agent spot along wall
                agent_meetup_spot = get_apple_meetup_agent_spot(sand, grass, water)
                if agent_meetup_spot:
                    nonsand_pixels = [(i, j) for i in range(observation.shape[0]) for j in range(observation.shape[1]) if (i, j) not in sand]
                    direction = get_direction_to_goal(agent_meetup_spot, grass+nonsand_pixels+players, [])
                    if direction:
                        # print('going to apple meetup', agent_meetup_spot)
                        return direction_to_number(direction), state
                # navigate until border of grass and sand is visible
                # print('navigating to grass because apple meetup spot not visible')
                return direction_to_number(state['apple_direction']), state

            # if apple direction is known and agent is not in grass, go in that direction
            if state['apple_direction'] != 'unknown':
                if dist_to_nearest_grass == None or dist_to_nearest_grass > 1.5:
                    goal_loc = get_directional_goal(state['apple_direction'])
                    direction = get_direction_to_goal(goal_loc, [obst for obst in walls+players if obst != goal_loc], [])
                    if direction:
                        return direction_to_number(direction), state
        
        if away_from_wall_direction != 'unknown':
            # print('going away from wall', away_from_wall_direction, ' hoping to find', ('APPLE' if state['goal_color'] == APPLE_COLOR else 'DIRT'), ' - Apple direction:', apple_direction)
            if away_from_wall_direction == 'down':
                print('turning away from wall')
                return 5, state
            return direction_to_number(away_from_wall_direction), state

        # print('exploring, hoping to find', ('APPLE' if state['goal_color'] == APPLE_COLOR else 'DIRT'), ' - Apple direction:', apple_direction)
        action = random.choice([1, 5, 6])
        print('random action')
        if action == 1 and is_within_range(observation[8, 5], WALL):
            action = random.choice([5, 6])

        return action, state

    def close(self):
        """ Required by base class """
        cv2.destroyAllWindows()
        pass
