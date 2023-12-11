from meltingpot.utils.policies.policy import Policy
import cv2
import dm_env
import numpy as np
import random

def print(*args, **kwargs):
    pass

EMPTY_SPACE = [56, 56, 56]

RIPE_BERRY_COLORS = [
        [82, 43, 43],
        [43, 82, 43],
        [43, 43, 82]
    ]

UNRIPE_NONRED_COLORS = [
        [46, 62, 46],
        [46, 46, 62],
    ]

UNRIPE_NONGREEN_COLORS = [
        [62, 46, 46],
        [46, 46, 62],
]

BLUE_PLAYER_RAW = (10, 10, 200)
GREEN_PLAYER_RAW = (10, 200, 10)
RED_PLAYER_RAW = (200, 10, 10)
GRAY_PLAYER_RAW = (125, 125, 125)
RAW_PLAYER_COLOR_MAP = {
    BLUE_PLAYER_RAW: 'blue',
    GREEN_PLAYER_RAW: 'green',
    RED_PLAYER_RAW: 'red',
    GRAY_PLAYER_RAW: 'gray'
}
EYE_COLOR = (60, 60, 60)

GREEN_PLAYER = (37, 110, 33) # greatest diff 12
ZAPPED_GREEN_PLAYER = (28, 94, 32)
RED_PLAYER = (113, 35, 35) # greatest diff 11
ZAPPED_RED_PLAYER = (97, 31, 31)
BLUE_PLAYER = (44, 44, 114) # greatest diff 12
ZAPPED_BLUE_PLAYER = (32, 28, 94)
GRAY_PLAYER = (79, 79, 83) # greatest diff 7


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

def is_within_range(a, b, threshold=12):
    within_threshold = np.all(np.abs(a - b) <= threshold)
    return within_threshold

def any_within_range(color, color_list):
    for color_i in color_list:
        if is_within_range(color, color_i):
            return True
    return False

def detect_in_claim_range(rgb_data, target_colors):
    within_claim_range = False
    if list(rgb_data[8, 5]) in target_colors: 
        within_claim_range = True
    elif list(rgb_data[7, 5]) in target_colors and list(rgb_data[8, 5]) == EMPTY_SPACE:
        within_claim_range = True
    elif list(rgb_data[6, 5]) in target_colors and list(rgb_data[8, 5]) == EMPTY_SPACE and \
    list(rgb_data[7, 5]) == EMPTY_SPACE:
        within_claim_range = True
    return within_claim_range

def find_nearest_berry(rgb_data, target_colors, offset_in_front_of_player=False):    
    combined_mask = np.zeros((11, 11), dtype=bool)
    
    # Create a mask for each target color and combine them
    for color in target_colors:
        mask = np.all(rgb_data == color, axis=-1)
        combined_mask = np.logical_or(combined_mask, mask)
    
    # Get row and column indices of pixels matching any of the target colors
    rows, cols = np.where(combined_mask)
    
    # If there's no matching pixel, return None
    if len(rows) == 0:
        return None
    
    origin_pos = (9, 5)
    if offset_in_front_of_player:
        origin_pos = (8, 5)

    # Calculate the Euclidean distance for each matching pixel to the point (10, 6)
    distances = np.sqrt((rows - origin_pos[0])**2 + (cols - origin_pos[1])**2)  # Subtracting 1 because of 0-based indexing
    
    # Find the pixel with the minimum distance
    min_index = np.argmin(distances)
    nearest_pixel = (rows[min_index], cols[min_index])
    
    return nearest_pixel

def detect_player(raw_rgb, rgb, i, j):
    """Return is_player, player_color, player_dir"""
    # this is a planting beam and not a player
    if tuple(rgb[i//8, j//8]) in RAW_PLAYER_COLOR_MAP:
        return False, '', ''
    # this is the focal agent- we only need to record other players
    if i == 72 and j == 40:
        return False, '', ''
    # used j+5 because j+4 includes berries and injured players
    rgb_val = tuple(raw_rgb[i+5, j+5])
    if rgb_val in RAW_PLAYER_COLOR_MAP:
        color = RAW_PLAYER_COLOR_MAP[rgb_val]
        direction = get_player_direction(raw_rgb, i, j)
        return True, color, direction
    return False, '', ''

def get_player_direction(raw_rgb, i, j):
    """Get player direction based on eye position
    Eye position based on direction (Eye color: (60, 60, 60))
    - Down: (3, 2), (3, 5)
    - Left: (3, 2), (3, 4)
    - Right (3, 3), (3, 5)
    - Up: No eyes visible
    """
    if tuple(raw_rgb[i+3, j+2]) == EYE_COLOR and tuple(raw_rgb[i+3, j+5]) == EYE_COLOR:
        return 'down'
    elif tuple(raw_rgb[i+3, j+2]) == EYE_COLOR and tuple(raw_rgb[i+3, j+4]) == EYE_COLOR:
        return 'left'
    elif tuple(raw_rgb[i+3, j+3]) == EYE_COLOR and tuple(raw_rgb[i+3, j+5]) == EYE_COLOR:
        return 'right'
    return 'up'

def direction_from_reference(nearest_pixel):
    reference_pixel = (9, 5)
    
    # Determine the direction
    if nearest_pixel[0] < reference_pixel[0]:  # Checking row value
        return "up"
    elif nearest_pixel[0] > reference_pixel[0]:
        return "down"
    elif nearest_pixel[1] < reference_pixel[1]:  # Checking column value
        return "left"
    elif nearest_pixel[1] > reference_pixel[1]:
        return "right"
    else:
        return "same_location"
    
def direction_to_number(direction):
    mapping = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "same_location": 0
    }
    return mapping.get(direction, 0)

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


def enemy_in_range(rgb, prefer_color):
    if prefer_color == 'red':
        friendly_colors = [RED_PLAYER, GRAY_PLAYER]
        enemy_colors = [GREEN_PLAYER, BLUE_PLAYER]
    elif prefer_color == 'green': 
        friendly_colors = [GREEN_PLAYER, GRAY_PLAYER]
        enemy_colors = [RED_PLAYER, BLUE_PLAYER]
    zap_positions = [(9, 4), (9, 6), (8, 4), (8, 5), (8, 6), (7, 5)]
    enemy_exists = False
    friendly_exists = False
    for zap_pos in zap_positions:
        if any_within_range(rgb[zap_pos[0], zap_pos[1]], enemy_colors):
            enemy_exists = True
        if any_within_range(rgb[zap_pos[0], zap_pos[1]], friendly_colors):
            friendly_exists = True
    if enemy_exists and not friendly_exists:
        return True
    return False

def zapped_enemy_in_range(rgb, prefer_color):
    if prefer_color == 'red':
        friendly_colors = [RED_PLAYER, GRAY_PLAYER]
        enemy_colors = [ZAPPED_GREEN_PLAYER, ZAPPED_BLUE_PLAYER]
    else: 
        friendly_colors = [GREEN_PLAYER, GRAY_PLAYER]
        enemy_colors = [ZAPPED_RED_PLAYER, ZAPPED_BLUE_PLAYER]
    zap_positions = [(9, 4), (9, 6), (8, 4), (8, 5), (8, 6), (7, 5)]
    enemy_exists = False
    friendly_exists = False
    for zap_pos in zap_positions:
        if any_within_range(rgb[zap_pos[0], zap_pos[1]], enemy_colors):
            enemy_exists = True
        if any_within_range(rgb[zap_pos[0], zap_pos[1]], friendly_colors):
            friendly_exists = True
    if enemy_exists and not friendly_exists:
        return True
    return False

def get_death_zones(players, player_directions, player_colors, prefer_color):
    """Define locations where enemy zaps can reach"""
    if prefer_color == 'red':
        friendly_colors = ['red']
    else: 
        friendly_colors = ['green']
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
        ['NOOP', 'FIRE_ONE', 'FIRE_TWO', 'FIRE_THREE', 'TURN_LEFT', 'TURN_RIGHT'] and (9, 5) in death_zones:
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
    'FIRE_ONE',
    'FIRE_TWO',
    'FIRE_THREE',
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


class HardCodeAlHarvestPolicy(Policy):
    """
        Hardcoded (rule-based) policy for allelopathic_harvest__open substrate
    """
    def __init__(self, policy_id):
        self.substrate_name = None
        self.policy_id = policy_id

    def initial_state(self):
        """ Called at the beginning of every episode """
        state = {
            'step_count': 0,
            'last_actions': [],
            'last_images': [],
            'prefer_color': 'red', # if 'green' in self.policy_id else 'red', #'red',
            'death_zones': [],
            'players': [],
        }
        return state
    
    def step(self, timestep, prev_state):
        """ Returns random actions according to spec """
        state = dict(prev_state)
        state['step_count'] += 1
        print('\ntimestep', state['step_count'])

        raw_timestep = timestep

        timestep = _downsample_single_timestep(timestep, 8)

        action, state = self.custom_step(timestep, raw_timestep, state)

        # record historical data
        num_actions_to_store = 5
        state['last_actions'] = [action] + state['last_actions']
        state['last_actions'] = state['last_actions'][:num_actions_to_store]
        num_images_to_store = 5
        state['last_images'] = [timestep.observation['RGB']] + state['last_images']
        state['last_images'] = state['last_images'][:num_images_to_store]

        # If image hasn't changed much in the past 5 steps, send a random action to get unstuck
        if len(state['last_images']) == num_images_to_store and all(isinstance(item, np.ndarray) for item in state['last_images']):
            avg_pixel_change = average_pixel_change(state['last_images'])
            if avg_pixel_change <= 10.0:
                repeated_action = state['last_actions'][0]
                possible_actions = [1, 2, 3, 4] # forward, backward, left, right
                if repeated_action in possible_actions:
                    possible_actions.remove(repeated_action)
                action = random.choice(possible_actions)
                state['last_actions'][0] = action

        if stepping_into_death(action, state['death_zones']):
            print('overriding action', action_to_enum(action), 'because it could lead to death')
            # choose the first action that is not obstacle or death zone
            new_action = evade_enemy_action(action, state['death_zones'], state['players'])
            if new_action:
                print('new action', action_to_enum(new_action))
                action = new_action
            else:
                print('no hope')

        return action, state


    def custom_step(self, timestep, raw_timestep, state):

        rgb_data = timestep.observation['RGB']
        raw_rgb_data = raw_timestep.observation['RGB']
        ready_to_shoot = timestep.observation['READY_TO_SHOOT'] == 1.0

        # determine non-injured player positions
        players = []
        player_colors = []
        player_directions = []
        for i in range(0, 88, 8):
            for j in range(0, 88, 8):
                is_player, player_color, player_dir = detect_player(raw_rgb_data, rgb_data, i, j)
                if is_player:
                    players.append((i//8, j//8))
                    player_colors.append(player_color)
                    player_directions.append(player_dir)

        death_zones = get_death_zones(players, player_directions, player_colors, state['prefer_color'])
        state['death_zones'] = death_zones
        state['players'] = players

        print('players', players)
        print('player colors', player_colors)
        print('player_directions', player_directions)
        print('last actions', state['last_actions'])
        print('prefer color', state['prefer_color'])
        
        # If enemy in shooting range and no friendly fire, zap
        if enemy_in_range(rgb_data, state['prefer_color']) and ready_to_shoot:
            return 7, state
        if zapped_enemy_in_range(rgb_data, state['prefer_color']): 
            if not ready_to_shoot:
                return 0, state
            else:
                return 7, state

        preffered_planting_colors = UNRIPE_NONRED_COLORS if state['prefer_color'] == 'red' else UNRIPE_NONGREEN_COLORS
        unripe_within_claim_range = detect_in_claim_range(rgb_data, preffered_planting_colors)
        if unripe_within_claim_range:
            if state['prefer_color'] == 'red':
                return 8, state
            else:
                return 9, state
        # If ripe berry is visible, go harvest it
        nearest_berry = find_nearest_berry(rgb_data, RIPE_BERRY_COLORS)
        nearest_unripe = find_nearest_berry(rgb_data, preffered_planting_colors)
        # if (nearest_berry and state['step_count'] > 200) or (nearest_berry and not nearest_unripe):
        if nearest_berry:
            direction = direction_from_reference(nearest_berry)
            return direction_to_number(direction), state
        
        # If unripe non-red berry is visble, go change the color to red
        if nearest_unripe:
            # if in front, plant red
            if nearest_unripe == (8, 5):
                if state['prefer_color'] == 'red':
                    return 8, state
                else:
                    return 9, state
            # if on left side, turn left
            if nearest_unripe == (9, 4):
                return 5, state
            # if on right side, turn right
            if nearest_unripe == (9, 6):
                return 6, state
            # if behind, turn around
            if nearest_unripe == (10, 5):
                return 6, state
            
            # Find closest berry to the position in front of player (where the zapper is)
            nearest_unripe = find_nearest_berry(rgb_data, preffered_planting_colors, offset_in_front_of_player=True)
            # ensure player navigates in front of berry
            nearest_unripe = (nearest_unripe[0]+1, nearest_unripe[1])
            direction = direction_from_reference(nearest_unripe)
            return direction_to_number(direction), state


        # Explore (because there are no berries to plant or harvest)
        return random.choice([1, 5, 6]), state
    
    def close(self):
        """ Required by base class """
        pass
