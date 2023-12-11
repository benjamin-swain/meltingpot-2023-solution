from meltingpot.utils.policies.policy import Policy
import cv2
import dm_env
import numpy as np
import random

def print(*args, **kwargs):
    pass

ZAP_COLOR = (252, 252, 106)

RED_CARD_COLOR = (129, 34, 53)
BLUE_CARD_COLOR = (34, 129, 109)
EMPTY_SPACE = (0, 0, 0)
WALL = (115, 115, 115)
AGENT_VIEW_SIZE = 11

GREY_CAP_COLOR = (204, 203, 200)

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

def interactable_player_in_zap_range(interactable_players, walls, players, goal_cells):
    """
    Interactable players is a list of tuples representing their position. If any are in zap range, return True. Otherwise False
    """
    for player in interactable_players:
        if player == (8,5):
            return True
        if player == (9,4):
            return True
        if player == (9,6):
            return True
        if player == (7,4) and not (set(walls+players+goal_cells) & set([(9,4), (8,4)])):
            return True
        if player == (8,4) and not (set(walls+players+goal_cells) & set([(9,4)])):
            return True
        if player == (7,6) and not (set(walls+players+goal_cells) & set([(8,6), (9,6)])):
            return True
        if player == (8,6) and not (set(walls+players+goal_cells) & set([(9,6)])):
            return True
        if player == (6,5) and not (set(walls+players+goal_cells) & set([(7, 5), (8,5)])):
            return True
        if player == (7,5) and not (set(walls+players+goal_cells) & set([(8,5)])):
            return True
        
def get_map_corner_direction(walls, empty_space):
    map_corner = detect_map_corner(walls, empty_space)
    if not map_corner:
        return None
    if map_corner[1] <= 5:
        if map_corner[0] <= 9:
            return 'northwest'
        else:
            return 'southwest'
    else:
        if map_corner[0] <= 9:
            return 'northeast'
        else:
            return 'southeast'

def detect_map_corner(walls, empty_space):
    walls_set = set(walls)
    empty_space_set = set(empty_space)

    def count_consecutive_neighbors(point, axis, direction, point_set, max_consecutive_to_check = 3):
        """ Count the number of consecutive neighbors for the point in the specified axis and direction, up to 3 """
        count = 0
        for i in range(1, max_consecutive_to_check + 1):
            next_point = (point[0] + i * direction if axis == 'x' else point[0],
                          point[1] + i * direction if axis == 'y' else point[1])
            if next_point in point_set:
                count += 1
            else:
                break
        return count

    for point in walls_set:
        x_pos_walls = count_consecutive_neighbors(point, 'x', 1, walls_set)
        x_neg_walls = count_consecutive_neighbors(point, 'x', -1, walls_set)
        y_pos_walls = count_consecutive_neighbors(point, 'y', 1, walls_set)
        y_neg_walls = count_consecutive_neighbors(point, 'y', -1, walls_set)

        # Check if there are >2 consecutive neighbors in either direction for both axes
        is_corner_extending_by_greater_than_2_in_both_directions = (x_pos_walls > 2 or x_neg_walls > 2) and (y_pos_walls > 2 or y_neg_walls > 2)
        if is_corner_extending_by_greater_than_2_in_both_directions:
            return point
        
        x_pos_space = count_consecutive_neighbors(point, 'x', 1, empty_space_set, max_consecutive_to_check=1)
        x_neg_space = count_consecutive_neighbors(point, 'x', -1, empty_space_set, max_consecutive_to_check=1)
        y_pos_space = count_consecutive_neighbors(point, 'y', 1, empty_space_set, max_consecutive_to_check=1)
        y_neg_space = count_consecutive_neighbors(point, 'y', -1, empty_space_set, max_consecutive_to_check=1)

        # Only check corners if one side extends by greater than 1 and the other side extends greater than 0
        is_corner = ((x_pos_walls > 1 or x_neg_walls > 1) and (y_pos_walls > 0 or y_neg_walls > 0)) or \
                    ((x_pos_walls > 0 or x_neg_walls > 0) and (y_pos_walls > 1 or y_neg_walls > 1))
        # Check if there is at least one consecutive neighbor in both directions for both axes in both sets
        if is_corner and (x_pos_space > 0 or x_neg_space > 0) and (y_pos_space > 0 or y_neg_space > 0):
            return point

    return None

def adjust_map_corner_direction(turn_direction, map_corner_direction):
    if not map_corner_direction:
        return None
    
    # Define a mapping for the left turn (90 degrees CW)
    left_turn_map = {
        "northwest": "northeast",
        "southwest": "northwest",
        "northeast": "southeast",
        "southeast": "southwest",
    }
    
    # Define a mapping for the right turn
    right_turn_map = {
        "northwest": "southwest",
        "southwest": "southeast",
        "northeast": "northwest",
        "southeast": "northeast",
    }
    
    # Check the turn_direction and adjust the apple_direction accordingly
    if turn_direction == 5:
        return left_turn_map[map_corner_direction]
    else:
        return right_turn_map[map_corner_direction]
    
def action_num_to_str(action_num):
    action_map = [
        'NOOP',
        'FORWARD',
        'BACKWARD',
        'STEP_LEFT',
        'STEP_RIGHT',
        'TURN_LEFT',
        'TURN_RIGHT',
        'INTERACT']
    return action_map[action_num]


class HardCodePDArenaPolicy(Policy):
    """
        Hardcoded (rule-based) policy for prisoners_dilemma_in_the_matrix__arena substrate
    """
    def __init__(self, policy_id):
        self.substrate_name = None
        self.policy_id = policy_id

    def initial_state(self):
        """ Called at the beginning of every episode """
        state = {
            'step_count': 0,
            'last_interaction_step': 0,
            'steps_since_turn_toward_player': 0,
            'map_corner_direction': None,
            'steps_since_ready_to_interact': 0
        }
        return state
    
    def step(self, timestep, prev_state):
        """ Returns random actions according to spec """
        state = dict(prev_state)

        raw_observation = timestep.observation['RGB']

        timestep = _downsample_single_timestep(timestep, 8)
        observation = timestep.observation['RGB']

        inventory = timestep.observation['INVENTORY']

        action, state = self.custom_step(observation, raw_observation, inventory, state)

        if np.all(observation == [0, 0, 0]):
            state['map_corner_direction'] = None
            state['steps_since_ready_to_interact'] = 0
        elif action in [5, 6]: 
            state['map_corner_direction'] = adjust_map_corner_direction(action, state['map_corner_direction'])
        
        return action, state


    def custom_step(self, observation, raw_observation, inventory, state):

        state['step_count'] += 1
        
        red_card_cells = []
        blue_card_cells = []
        walls = []
        empty_space = []
        players = []
        zaps = []
        interactable_players = []
        noninteractable_players = []
        for i in range(AGENT_VIEW_SIZE):
            for j in range(AGENT_VIEW_SIZE):
                if np.array_equal(observation[i, j], RED_CARD_COLOR):
                    red_card_cells.append((i, j))
                elif np.array_equal(observation[i, j], BLUE_CARD_COLOR):
                    blue_card_cells.append((i, j))
                elif np.array_equal(observation[i, j], WALL):
                    walls.append((i, j))
                elif np.array_equal(observation[i, j], EMPTY_SPACE):
                    empty_space.append((i, j))
                elif np.array_equal(observation[i, j], ZAP_COLOR):
                    zaps.append((i, j))
                else:
                    players.append((i, j))
                    if np.array_equal(raw_observation[(i*8)+1, (j*8)+2], GREY_CAP_COLOR) and (i, j) != (9, 5) and (i, j) != (10, 5):
                        interactable_players.append((i, j))
                    else:
                        noninteractable_players.append((i,j))

        map_corner_direction = get_map_corner_direction(walls, empty_space)
        if map_corner_direction:
            state['map_corner_direction'] = map_corner_direction
        print('map corner dir', state['map_corner_direction'])
        print('inventory', inventory)

        state['steps_since_turn_toward_player'] += 1

        required_inventory_before_seek_interact = 2
        # go interact
        if (inventory[0] > required_inventory_before_seek_interact or inventory[1] > required_inventory_before_seek_interact) and \
            interactable_players:
            print('going to interact')
            nearest_player = get_nearest(interactable_players)
            # If player within zap range, zap! (8, 5)
            if interactable_player_in_zap_range(interactable_players, walls, players, blue_card_cells+red_card_cells):
                state['last_interaction_step'] = state['step_count']
                return 7, state
            # If player to left (and not so far forward that it will be out of view), turn left
            if nearest_player[1] < 4 and nearest_player[0] > 3 and state['steps_since_turn_toward_player'] > 10:
                state['steps_since_turn_toward_player'] = 0
                return 5, state
            # If player to right (and not so far forward that it will be out of view), turn right
            if nearest_player[1] > 6 and nearest_player[0] > 3 and state['steps_since_turn_toward_player'] > 10:
                state['steps_since_turn_toward_player'] = 0
                return 6, state
            
            nearest_player = (nearest_player[0]+1, nearest_player[1])
            direction = get_direction_to_goal(nearest_player, walls+noninteractable_players, [])
            if direction:
                return direction_to_number(direction), state

        # If you've just picked up enough cards for interaction, spin to find players
        if (inventory[0] > required_inventory_before_seek_interact or inventory[1] > required_inventory_before_seek_interact) and state['steps_since_ready_to_interact'] < 4:
            state['steps_since_ready_to_interact'] += 1
            print('spinning after inventory ready')
            return 6, state

        # If red card is visible
        if red_card_cells:
            # print('picking up cards. players:', (True if len(players)>0 else False), 'interactable:', (True if len(interactable_players)>0 else False), 'steps since interact:', steps_since_interact)
            print('picking up red card')
            nearest_goal = get_nearest(red_card_cells)
            direction = get_direction_to_goal(nearest_goal, walls+players, [])
            if direction:
                return direction_to_number(direction), state

        # If blue card is visible
        if blue_card_cells:
            # print('picking up cards. players:', (True if len(players)>0 else False), 'interactable:', (True if len(interactable_players)>0 else False), 'steps since interact:', steps_since_interact)
            print('picking up blue card')
            nearest_goal = get_nearest(blue_card_cells)
            direction = get_direction_to_goal(nearest_goal, walls+players, [])
            if direction:
                return direction_to_number(direction), state
              
        # If you have cards, just rotate in place to hopefully see someone to interact with
        if (inventory[0] > required_inventory_before_seek_interact or inventory[1] > required_inventory_before_seek_interact):
            print('spinning to find interactable player')
            return 6, state

        # Explore away from map corner
        if state['map_corner_direction']:
            # Turn away from map corner
            if state['map_corner_direction'] == 'northwest':
                print('turning away from corner')
                return 6, state
            if state['map_corner_direction'] == 'northeast':
                print('turning away from corner')
                return 5, state
            if state['map_corner_direction'] == 'southwest':
                temp_goal_pos = (6, 8)
                temp_obstacles = [x for x in walls+players if x != temp_goal_pos]
                direction = get_direction_to_goal(temp_goal_pos, temp_obstacles, [])
                if direction:
                    print('navigating away from corner')
                    return direction_to_number(direction), state
                else:
                    print('no route away from corner')
            if state['map_corner_direction'] == 'southeast':
                temp_goal_pos = (6, 2)
                temp_obstacles = [x for x in walls+players if x != temp_goal_pos]
                direction = get_direction_to_goal(temp_goal_pos, temp_obstacles, [])
                if direction:
                    print('navigating away from corner')
                    return direction_to_number(direction), state
                else:
                    print('no route away from corner')

        # If no goal cell is visible, explore
        action = random.choice([1, 1, 1, 5, 6])
        if action == 1 and np.array_equal(observation[8, 5], WALL):
            action = random.choice([5, 6])
        # print('exploring. inventory:', inventory, ' steps_since_interact:', steps_since_interact, 'action:', action_num_to_str(action))
        return action, state
    
    def close(self):
        """ Required by base class """
        pass
