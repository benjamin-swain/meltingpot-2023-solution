from meltingpot.utils.policies.policy import Policy
import cv2
import dm_env
import numpy as np
import random
import feature_detector_territory

def print(*args, **kwargs):
    pass
  
# PLAYER COLORS FROM RAW IMAGE
BLUE_PLAYER = (45, 110, 220)
PINK_PLAYER = (205, 5, 165)
PURPLE_PLAYER = (125, 50, 200)
RED_PLAYER = (245, 65, 65)
TEAL_PLAYER = (35, 185, 175)
GREEN_PLAYER = (125, 185, 65)
LIGHT_PURPLE_PLAYER = (160, 15, 200)
YELLOW_PLAYER = (195, 180, 0)
ORANGE_PLAYER = (245, 130, 0)

PLAYER_COLOR_LIST = ['blue', 'pink', 'purple', 'red', 'teal', 'green', 'light_purple', 'yellow', 'orange']

# MISC
PAINT_HANDLE_GRAY_COLOR = (70, 70, 70)
PAINT_HANDLE_BROWN_COLOR = (117, 79, 61)

LIGHT_PURPLE_CLAIMING_BEAM = (160, 15, 200)
BLUE_CLAIMING_BEAM = (45, 110, 220)
PURPLE_CLAIMING_BEAM = (125, 50, 200)
ORANGE_CLAIMING_BEAM = (245, 130, 0)
PINK_CLAIMING_BEAM = (205, 5, 165)
GREEN_CLAIMING_BEAM = (125, 185, 65)
YELLOW_CLAIMING_BEAM = (195, 180, 0)
RED_CLAIMING_BEAM = (245, 65, 65)
TEAL_CLAIMING_BEAM = (35, 185, 175)

BEAM_COLOR_MAP = {
    'light_purple': LIGHT_PURPLE_CLAIMING_BEAM,
    'blue': BLUE_CLAIMING_BEAM,
    'purple': PURPLE_CLAIMING_BEAM,
    'orange': ORANGE_CLAIMING_BEAM,
    'pink': PINK_CLAIMING_BEAM,
    'green': GREEN_CLAIMING_BEAM,
    'yellow': YELLOW_CLAIMING_BEAM,
    'red': RED_CLAIMING_BEAM,
    'teal': TEAL_CLAIMING_BEAM
}

AGENT_VIEW_SIZE = 11

ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1)
}

 
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

def get_player_color(my_color):
    """
    player colors:
    green
    purple
    light_purple
    pink
    blue
    teal
    red
    yellow
    orange
    """
    if tuple(my_color) == GREEN_PLAYER:
        return 'green'
    if tuple(my_color) == PURPLE_PLAYER:
        return 'purple'
    if tuple(my_color) == LIGHT_PURPLE_PLAYER:
        return 'light_purple'
    if tuple(my_color) == PINK_PLAYER:
        return 'pink'
    if tuple(my_color) == BLUE_PLAYER:
        return 'blue'
    if tuple(my_color) == TEAL_PLAYER:
        return 'teal'
    if tuple(my_color) == RED_PLAYER:
        return 'red'
    if tuple(my_color) == YELLOW_PLAYER:
        return 'yellow'
    if tuple(my_color) == ORANGE_PLAYER:
        return 'orange'
    return None
                
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

def get_neighbours(cell):
    """ Return the neighbours of the cell in the grid """
    i, j = cell
    return [(i + di, j + dj) for di, dj in ACTIONS.values() if 0 <= i + di < AGENT_VIEW_SIZE and 0 <= j + dj < AGENT_VIEW_SIZE]

def shortest_path_to_goal(start, goal, obstacles):
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
            if (neighbour not in obstacles or neighbour == goal) and neighbour not in visited:
                queue.append((neighbour, path + [neighbour]))
    return []

def adjust_offscreen_goal(goal):
    """return the closest onscreen pixel to the offscreen goal. offscreen goals
    are always only 1 pixel off screen"""
    if not goal:
        return goal
    new_goal = list(goal)
    for i, goal_val in enumerate(goal):
        if goal_val == -1:
            new_goal[i] = 0
        if goal_val == 11:
            new_goal[i] = 10
    return tuple(new_goal)

def get_direction_to_goal(goal, obstacles):
    start = (9, 5)
    goal = adjust_offscreen_goal(goal)
    path = shortest_path_to_goal(start, goal, obstacles)
    if not path:
        return None
    next_cell = path[0]
    for action, (di, dj) in ACTIONS.items():
        if (start[0] + di, start[1] + dj) == next_cell:
            return action
    return None


def player_in_zap_range(players, friendly_players, obstacles):
    """
    Enemy players is a list of tuples representing their position. If any are in zap range, return True. Otherwise False
    """
    for player in friendly_players:
        if player == (8,5):
            return False
        if player == (9,4):
            return False
        if player == (9,6):
            return False
        if player == (8,4) and not (set(obstacles) & set([(9,4)])):
            return False
        if player == (8,6) and not (set(obstacles) & set([(9,6)])):
            return False
        if player == (7,5) and not (set(obstacles) & set([(8,5)])):
            return False
    for player in players:
        if player == (8,5):
            return True
        if player == (9,4):
            return True
        if player == (9,6):
            return True
        if player == (8,4) and not (set(obstacles) & set([(9,4)])):
            return True
        if player == (8,6) and not (set(obstacles) & set([(9,6)])):
            return True
        if player == (7,5) and not (set(obstacles) & set([(8,5)])):
            return True

def direction_to_number(direction):
    mapping = {
        "up": 1,
        "down": 2,
        "left": 3,
        "right": 4,
        "same_location": 0
    }
    return mapping.get(direction, 0)

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
    'FIRE_CLAIM'
    )
    return action_set[action]

def detect_enemy_at_start(players, expect_zap, claiming_beams, player_colors):
    """Return a list of player colors and their direction who are not at the expected starting location 
    or who are not firing claiming beam at expected timestep"""
    focal_location = (2, 5)
    enemy_colors = []
    friendly_colors = []
    if not players:
        return enemy_colors, friendly_colors
    for i, player in enumerate(players):
        if player != focal_location or (expect_zap and player == focal_location and not is_player_claiming(claiming_beams)):
            enemy_color = player_colors[i]
            enemy_colors.append(enemy_color)
        else:
            friendly_colors.append(player_colors[i])
    return enemy_colors, friendly_colors

def is_player_claiming(claiming_beams):
    pixels_to_check = [(1, 5), (2, 4), (2, 6), (3, 5)]
    for color in claiming_beams:
        for pos in claiming_beams[color]:
            if pos in pixels_to_check:
                return True
    return False

def get_evade_goals(image, players, obstacles):
    """
    Given an image as a numpy array and a list of player positions,
    return a list of all pixels with the target color, sorted from farthest to nearest based on their distances to the players.

    :param image: A numpy array of shape (H, W, 3) representing an image
    :param players: A list of tuples representing the (x, y) positions of the players
    :return: A list of tuples representing the (x, y) positions of the pixels with the target color,
             sorted from farthest to nearest based on their distances to the players
    """
    # Initialize a list to store pixels with the target color and their distances
    target_pixels = []

    # Iterate through each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if the color of the current pixel matches the target color
            if (y, x) not in obstacles:
                # Calculate the distance from this pixel to each player position
                distances = [euclidean_distance((y, x), player) for player in players]
                
                if not distances:
                    print('empty distances, players', players)
                    continue
                # Find the minimum distance to any player
                min_distance_to_players = min(distances)
                
                # Add the pixel and its distance to the list
                target_pixels.append(((y, x), min_distance_to_players))

    # Sort the list by distance in descending order
    target_pixels.sort(key=lambda x: -x[1])

    evade_goals = [pixel[0] for pixel in target_pixels]

    # Return the sorted list of pixels
    return evade_goals
       
def find_paint_brushes(image):
    brown_pixels = np.column_stack(np.where(np.all(image == PAINT_HANDLE_BROWN_COLOR, axis=-1)))
    
    t_centers = []
    t_orientations = []
    brush_colors = []

    t_patterns = {
        'down': [(1, -1), (1, 0), (1, 1)],
        'up': [(-1, -1), (-1, 0), (-1, 1)],
        'right': [(-1, 1), (0, 1), (1, 1)],
        'left': [(-1, -1), (0, -1), (1, -1)]
    }
    brush_color_patterns = {
        'down': (4, 1),
        'up': (-4, -1),
        'left': (-1, -4),
        'right': (-1,  4)
    }
    
    for y, x in brown_pixels:
        # ignore focal player
        if y==73 and x==46:
            continue
        if y > 0 and y < image.shape[0] - 1 and x > 0 and x < image.shape[1] - 1:
            for orientation, pattern in t_patterns.items():
                if all(np.array_equal(image[y + dy, x + dx], PAINT_HANDLE_GRAY_COLOR) for dy, dx in pattern):
                    t_centers.append((y + pattern[-1][1], x + pattern[-1][0]))
                    t_orientations.append(orientation)
                    brush_color_y = y + brush_color_patterns[orientation][0]
                    brush_color_x = x + brush_color_patterns[orientation][1]
                    brush_colors.append(get_player_color(image[brush_color_y, brush_color_x]))
                    break

    # convert brush location to player location
    new_t_centers = []
    for i, dir in enumerate(t_orientations):
        y = t_centers[i][0] // 8
        x = t_centers[i][1] // 8
        if dir == 'left':
            new_t_centers.append((y, x+1))
        elif dir == 'right':
            new_t_centers.append((y, x-1))
        elif dir == 'up':
            new_t_centers.append((y, x))
        elif dir == 'down':
            new_t_centers.append((y-1, x))

    return new_t_centers, t_orientations, brush_colors

def get_death_zones(players, player_directions, player_colors, zap_obstacles, state):
    """Define locations where enemy zaps can reach"""
    death_zones = []
    for i, loc in enumerate(players):
        dir = player_directions[i]
        color = player_colors[i]
        if color in state['friendly_colors']:
            continue
        if dir == 'unknown':
            continue
        # Enemies can zap when step-since-zap reaches 4
        if (color and state[color+'_steps_since_zap'] != None and state[color+'_steps_since_zap'] <= 2):
            continue
        # It takes 2 steps to get out of an enemy's center zap area, so start moving away at step 3
        if (color and state[color+'_steps_since_zap'] != None and state[color+'_steps_since_zap'] == 3):
            zap_locs = get_front_zap_locations(loc, dir, zap_obstacles)
        else:
            zap_locs = get_expected_zap_locations(loc, dir, zap_obstacles, include_off_screen=True)
        death_zones = death_zones + zap_locs
    return death_zones

def evade_enemy_direction(observation, players, injured_players, static_walls, 
                          obstacles, death_zones, active_enemy_players, friendly_players,
                          zap_obstacles):
    all_obstacles = list(set(static_walls+obstacles+injured_players+players+death_zones))
    evade_goals = get_evade_goals(observation, active_enemy_players, all_obstacles)
    direction = None
    for evade_goal in evade_goals:
        direction = get_direction_to_goal(evade_goal, all_obstacles)
        if direction:
            print('evade goal', evade_goal)
            break     
    print('evade direction', direction)
    if direction == 'left' and (9, 4) in obstacles and (9, 4) not in static_walls and \
        not player_in_zap_range(friendly_players, [], zap_obstacles):
        return 7
    if direction == 'right' and (9, 6) in obstacles and (9, 6) not in static_walls and \
        not player_in_zap_range(friendly_players, [], zap_obstacles):
        return 7
    if direction == 'down' and (10, 5) in obstacles and (10, 5) not in static_walls:
        return 6
    if direction == 'up' and (8, 5) in obstacles and (8, 5) not in static_walls and \
        not player_in_zap_range(friendly_players, [], zap_obstacles):
        return 7
    if direction:
        return direction_to_number(direction)
    if player_in_zap_range(active_enemy_players, friendly_players, zap_obstacles):
        print('no where to evade to, but zapping anyway')
        return 7
    print('no where to evade to')
    return 0

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def stepping_into_death(action, death_zones):
    if action_to_enum(action) == 'FORWARD' and (8, 5) in death_zones:
        return True
    if action_to_enum(action) == 'BACKWARD' and (10, 5) in death_zones:
        return True
    if action_to_enum(action) == 'STEP_LEFT' and (9, 4) in death_zones:
        return True
    if action_to_enum(action) == 'STEP_RIGHT' and (9, 6) in death_zones:
        return True
    return False
    
def get_front_zap_locations(player_pos, direction, zap_obstacles):
    """Return the zap locations surrounding the player because they could rotate"""
    final_locs = []
    pixels_to_check = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    expected_locs = [(player_pos[0]+dy, player_pos[1]+dx) for dy, dx in pixels_to_check]
    for expected_loc in expected_locs:
        if expected_loc not in zap_obstacles:
            final_locs.append(expected_loc)
    return final_locs

def get_expected_zap_locations(player_pos, direction, zap_obstacles, include_off_screen=False):
    """Return the pixels expected to have the zap color based on player pos, direction, and obstacles"""
    expected_zap_locations = []
    if direction == 'left':
        # first list right beam, then left beam, then center beam
        pixels_to_check = [(-1, 0), (-1, -1), (1, 0), (1, -1), (0, -1), (0, -2)]
    elif direction == 'right':
        pixels_to_check = [(1, 0), (1, 1), (-1, 0), (-1, 1), (0, 1), (0, 2)]
    elif direction == 'up':
        pixels_to_check = [(0, 1), (-1, 1), (0, -1), (-1, -1), (-1, 0), (-2, 0)]
    elif direction == 'down':
        pixels_to_check = [(0, -1), (1, -1), (0, 1), (1, 1), (1, 0), (2, 0)]
    else:
        return []

    expected_locs = [(player_pos[0]+dy, player_pos[1]+dx) for dy, dx in pixels_to_check]

    # right beam
    if expected_locs[0] not in zap_obstacles:
        expected_zap_locations.append(expected_locs[0])
        if expected_locs[1] not in zap_obstacles:
            expected_zap_locations.append(expected_locs[1])
    # left beam
    if expected_locs[2] not in zap_obstacles:
        expected_zap_locations.append(expected_locs[2])
        if expected_locs[3] not in zap_obstacles:
            expected_zap_locations.append(expected_locs[3])
    # center beam
    if expected_locs[4] not in zap_obstacles:
        expected_zap_locations.append(expected_locs[4])
        if expected_locs[5] not in zap_obstacles:
            expected_zap_locations.append(expected_locs[5])

    if not include_off_screen:
        expected_zap_locations = [loc for loc in expected_zap_locations if loc[0] >= 0 and \
                                  loc[0] <= 10 and loc[1] >= 0 and loc[1] <= 10]
    return expected_zap_locations


def update_enemy_steps_since_zap(state, players, player_colors, player_directions, zaps, zap_obstacles):
    """Given a list of pixel positions of players, zaps, and zap obstacles, and the player colors,
    detect if a player is zapping at the current step and set their steps-since-zap to 0 if they are zapping"""
    players_near_zaps = []
    directions = []
    colors = []
    for i, player in enumerate(players):
        adjacent_pixels = [
            (player[0] - 1, player[1]),  # above
            (player[0] + 1, player[1]),  # below
            (player[0], player[1] - 1),  # left
            (player[0], player[1] + 1),  # right
        ]
        for adjacent_pixel in adjacent_pixels:
            if adjacent_pixel in zaps and player not in players_near_zaps:
                players_near_zaps.append(player)
                directions.append(player_directions[i])
                colors.append(player_colors[i])
    # Calculate the expected zap locations if the player fired their zapper
    for i, player in enumerate(players_near_zaps):
        direction = directions[i]
        color = colors[i]
        expected_zap_locs = get_expected_zap_locations(player, direction, zap_obstacles)
        if all(item in zaps for item in expected_zap_locs) and color in PLAYER_COLOR_LIST:
            state[color+'_steps_since_zap'] = 0
    return state
    
def update_enemy_steps_since_injured(state, prev_state, injured_player_colors):
    """If player is ok in the previous state but injured in the current state, set the steps-since-injured to 0.
    When a player's steps-since-injured is less than 25, the player will be added to normal players list"""
    for color in PLAYER_COLOR_LIST:
        # only consider players who were uninjured in the previous timestep
        if not color in prev_state['player_colors']:
            continue
        idx = prev_state['player_colors'].index(color)
        player_pos = prev_state['players'][idx]
        # only consider players who were onscreen in the previous timestep (because sometimes injured
        # players are detected as normal players if they are offscreen)
        if player_pos[0] < 0 or player_pos[0] > 10 or player_pos[1] < 0 or player_pos[1] > 10:
            continue
        # only consider players who are injured in the current timestep
        if color not in injured_player_colors:
            continue
        state[color+'_steps_since_injured'] = 0
    return state

def update_active_players(state, players, player_colors, player_directions, 
                                  injured_players, injured_player_directions, injured_player_colors):
    """Active players are all players who can move. This will be players + active injured players"""
    active_players = players.copy()
    active_player_directions = player_directions.copy()
    active_player_colors = player_colors.copy()
    # add injured players to active players if their steps-since-injured >= 24
    for i, color in enumerate(injured_player_colors):
        steps_since_injured = state[color+'_steps_since_injured']
        if steps_since_injured >= 24:
            active_players.append(injured_players[i])
            active_player_directions.append(injured_player_directions[i])
            active_player_colors.append(color)
    return active_players, active_player_directions, active_player_colors

def get_mystery_players(player_colors, injured_player_colors, players, injured_players, 
                        player_directions, injured_player_directions, state):
    """get mystery player info- these are players who are not determined to be focal or background players yet.
       They must also not be at the edge of your vision"""
    colors = []
    directions = []
    positions = []
    for i, player_color in enumerate(player_colors):
        if player_color not in state['enemy_colors'] and player_color not in state['friendly_colors']:
            colors.append(player_color)
            directions.append(player_directions[i])
            positions.append(players[i])
    for i, player_color in enumerate(injured_player_colors):
        if player_color not in state['enemy_colors'] and player_color not in state['friendly_colors']:
            colors.append(player_color)
            directions.append(injured_player_directions[i])
            positions.append(injured_players[i])
    return colors, positions, directions

def filter_to_both_players_visible(colors, positions, directions):
    """Given a list of colors (ex: ['red']), positions (ex: [(4,5)]), and directions (ex: ['up'])
    representing players, reduce the lists to only players who are within each other's FOVs (must be
    at least 1 pixel inside from the edge of the FOV so the claiming beam will be visible even when
    players face toward the outside of the FOV)"""
    target_cell = (9, 5)
    visible_players_colors = []
    visible_players_positions = []
    visible_players_directions = []
    
    for color, position, direction in zip(colors, positions, directions):
        y, x = position
        if y < 1 or y > 9 or x < 1 or x > 9:
            continue
        visible_area = []
        if direction == 'up':
            visible_area = [(y + dy, x + dx) for dy in range(-8, 1) for dx in range(-4, 5)]
        elif direction == 'down':
            visible_area = [(y + dy, x + dx) for dy in range(0, 9) for dx in range(-4, 5)]
        elif direction == 'left':
            [(y + dy, x + dx) for dy in range(-4, 5) for dx in range(-8, 1)]
        elif direction == 'right':
            [(y + dy, x + dx) for dy in range(-4, 5) for dx in range(0, 9)]

        if target_cell in visible_area:
            visible_players_colors.append(color)
            visible_players_positions.append(position)
            visible_players_directions.append(direction)

    return visible_players_colors, visible_players_positions, visible_players_directions

def get_claiming_beams(array):
    """Return a dictionary where key is color and value is a list of claiming beam positions
    in the inputted 11x11x3 image"""
    positions_found = {}
    for color_name, color_value in BEAM_COLOR_MAP.items():
        mask = (array == color_value).all(axis=-1)
        if mask.any():
            positions = list(zip(*np.where(mask)))
            positions_found[color_name] = positions

    return positions_found  

def get_space_from_corner(pos, orientation):
    space = []
    if orientation == 'top_left':
        space = [(pos[0]+dy, pos[1]+dx) for dy in range(0, 7) for dx in range(0, 7)]
    elif orientation == 'top_right':
        space = [(pos[0]+dy, pos[1]+dx) for dy in range(0, 7) for dx in range(-6, 1)]
    elif orientation == 'bottom_left':
        space = [(pos[0]+dy, pos[1]+dx) for dy in range(-6, 1) for dx in range(0, 7)]
    elif orientation == 'bottom_right':
        space = [(pos[0]+dy, pos[1]+dx) for dy in range(-6, 1) for dx in range(-6, 1)]
    return space

def get_random_move(friendly_claimed_walls):
    possible_choices = [1, 2, 3, 4, 5, 6, 0]
    if (7, 5) in friendly_claimed_walls:
        possible_choices.remove(1)
    if (8, 4) in friendly_claimed_walls:
        possible_choices.remove(3)
    if (8, 6) in friendly_claimed_walls:
        possible_choices.remove(4)
    if (9, 6) in friendly_claimed_walls:
        possible_choices.remove(6)
    if (9, 4) in friendly_claimed_walls:
        possible_choices.remove(5)
    return random.choice(possible_choices)

def am_i_closest(target, friendly_players):
    """target is a tuple like (9, 4). friendly_players is a list of tuples like [(7, 5), (8, 2)].
    If the target is closer to (9, 5) than it is to any location in friendly_players, return True.
    Otherwise, return False"""
    def distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    my_location = (9, 5)
    distance_to_me = distance(target, my_location)
    for player_location in friendly_players:
        if distance(target, player_location) <= distance_to_me:
            return False
    # If no friendly player is closer to the target than my_location, return True
    return True

def get_northwest_corner(pos, static_walls, raw_rgb):
    """Given a position (pos) within one of the territory spaces, define the northwest corner
    as the static wall with 2 bright edges at its corner. Return the northwest corner position, and 
    its orientation from the focal agent's perspective"""
    nearest_static_walls = get_nearest_list(static_walls, from_pos=pos)
    if not nearest_static_walls:
        return pos
    # detect if the nearest static wall is northeast, southeast, southwest, or northwest corner
    nearest_static_wall = nearest_static_walls[0]
    static_wall_type = get_static_wall_type(nearest_static_wall, raw_rgb)
    if not static_wall_type:
        return pos
    if static_wall_type == 'northwest_top_left':
        return nearest_static_wall, 'bottom_right'
    if static_wall_type == 'northwest_top_right':
        return nearest_static_wall, 'bottom_left'
    if static_wall_type == 'northwest_bottom_right':
        return nearest_static_wall, 'top_left'
    if static_wall_type == 'northwest_bottom_left':
        return nearest_static_wall, 'top_right'
    
    if static_wall_type == 'southeast_top_left':
        return (nearest_static_wall[0]-6, nearest_static_wall[1]-6), 'bottom_right'
    if static_wall_type == 'southeast_top_right':
        return (nearest_static_wall[0]-6, nearest_static_wall[1]+6), 'bottom_left'
    if static_wall_type == 'southeast_bottom_right':
        return (nearest_static_wall[0]+6, nearest_static_wall[1]+6), 'top_left'
    if static_wall_type == 'southeast_bottom_left':
        return (nearest_static_wall[0]+6, nearest_static_wall[1]-6), 'top_right'
    
    if static_wall_type == 'southwest_top_left':
        return (nearest_static_wall[0], nearest_static_wall[1]-6), 'bottom_right'
    if static_wall_type == 'southwest_top_right':
        return (nearest_static_wall[0]-6, nearest_static_wall[1]), 'bottom_left'
    if static_wall_type == 'southwest_bottom_right':
        return (nearest_static_wall[0], nearest_static_wall[1]+6), 'top_left'
    if static_wall_type == 'southwest_bottom_left':
        return (nearest_static_wall[0]+6, nearest_static_wall[1]), 'top_right'
    
    if static_wall_type == 'northeast_top_left':
        return (nearest_static_wall[0]-6, nearest_static_wall[1]), 'bottom_right'
    if static_wall_type == 'northeast_top_right':
        return (nearest_static_wall[0], nearest_static_wall[1]+6), 'bottom_left'
    if static_wall_type == 'northeast_bottom_right':
        return (nearest_static_wall[0]+6, nearest_static_wall[1]), 'top_left'
    if static_wall_type == 'northeast_bottom_left':
        return (nearest_static_wall[0], nearest_static_wall[1]-6), 'top_right'
    

def get_static_wall_type(static_wall_pos, raw_rgb):
    top_left_1 = (static_wall_pos[0]*8+1, static_wall_pos[1]*8)
    top_left_2 = (static_wall_pos[0]*8, static_wall_pos[1]*8+1)

    top_right_1 = (static_wall_pos[0]*8, static_wall_pos[1]*8+6)
    top_right_2 = (static_wall_pos[0]*8+1, static_wall_pos[1]*8+7)

    bottom_right_1 = (static_wall_pos[0]*8+6, static_wall_pos[1]*8+7)
    bottom_right_2 = (static_wall_pos[0]*8+7, static_wall_pos[1]*8+6)

    bottom_left_1 = (static_wall_pos[0]*8+7, static_wall_pos[1]*8+1)
    bottom_left_2 = (static_wall_pos[0]*8+6, static_wall_pos[1]*8)

    cw_corner_positions = [[top_left_1, top_left_2], [top_right_1, top_right_2], 
                           [bottom_right_1, bottom_right_2], [bottom_left_1, bottom_left_2]]
    corner_names = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

    bright_edge = (88, 84, 82)
    dark_edge = (53, 49, 47)

    for i, corners in enumerate(cw_corner_positions):
        corner_1 = corners[0]
        corner_2 = corners[1]
        if np.array_equal(raw_rgb[corner_1[0], corner_1[1]], bright_edge) and \
        np.array_equal(raw_rgb[corner_2[0], corner_2[1]], dark_edge):
            return 'northeast_' + corner_names[i]

        if np.array_equal(raw_rgb[corner_1[0], corner_1[1]], dark_edge) and \
        np.array_equal(raw_rgb[corner_2[0], corner_2[1]], dark_edge):
            return 'southeast_' + corner_names[i]

        if np.array_equal(raw_rgb[corner_1[0], corner_1[1]], dark_edge) and \
        np.array_equal(raw_rgb[corner_2[0], corner_2[1]], bright_edge):
            return 'southwest_' + corner_names[i]

        if np.array_equal(raw_rgb[corner_1[0], corner_1[1]], bright_edge) and \
        np.array_equal(raw_rgb[corner_2[0], corner_2[1]], bright_edge):
            return 'northwest_' + corner_names[i]
        
    return None
    

class HardCodeTerritoryPolicy(Policy):
    """
        Hardcoded (rule-based) policy for territory__rooms substrate
    """
    def __init__(self, policy_id):
        self.substrate_name = None
        self.policy_id = policy_id
        
    def initial_state(self):
        """ Called at the beginning of every episode """
        state = {'step_count': 0,
                 'zapped': False,
                 'steps_since_turn_toward_player': 11,
                 'last_zap_step': 0,
                 'friendly_colors': [],
                 'enemy_colors': [],
                 'observation': np.array([]),
                 'active_players': [],
                 'players': [],
                 'player_colors': [],
                 'injured_players': [],
                 'injured_player_colors': [],
                 'static_walls': [],
                 'obstacles': [],
                 'death_zones': [],
                 'is_injured': False,
                 'blue_steps_since_zap': 5,
                 'pink_steps_since_zap': 5,
                 'purple_steps_since_zap': 5,
                 'red_steps_since_zap': 5,
                 'teal_steps_since_zap': 5,
                 'green_steps_since_zap': 5,
                 'light_purple_steps_since_zap': 5,
                 'yellow_steps_since_zap': 5,
                 'orange_steps_since_zap': 5,
                 # it takes ~25 steps for an injured player to move again,
                 # after which the injured player should be treated as a normal player
                 'blue_steps_since_injured': 30,
                 'pink_steps_since_injured': 30,
                 'purple_steps_since_injured': 30,
                 'red_steps_since_injured': 30,
                 'teal_steps_since_injured': 30,
                 'green_steps_since_injured': 30,
                 'light_purple_steps_since_injured': 30,
                 'yellow_steps_since_injured': 30,
                 'orange_steps_since_injured': 30,
                 'handshaking_with': [],
                 'steps_since_start_handshake': 10,
                 'num_blue_claims': 0,
                 'num_pink_claims': 0,
                 'num_purple_claims': 0,
                 'num_red_claims': 0,
                 'num_teal_claims': 0,
                 'num_green_claims': 0,
                 'num_light_purple_claims': 0,
                 'num_yellow_claims': 0,
                 'num_orange_claims': 0}
        return state
    
    def step(self, timestep, prev_state):
        state = dict(prev_state)

        print('\nstep', state['step_count']+1)

        state['step_count'] += 1
        state['zapped'] = False
        state['steps_since_turn_toward_player'] += 1
        state['steps_since_start_handshake'] += 1
        for color in PLAYER_COLOR_LIST:
            state[color+'_steps_since_zap'] += 1
            state[color+'_steps_since_injured'] += 1

        action, state = self.custom_step(timestep, state, prev_state)
        
        if stepping_into_death(action, state['death_zones']):
            print('overriding action', action_to_enum(action), 'because it will lead to death')
            action = evade_enemy_direction(state['observation'], 
                                           state['active_players'], 
                                           state['injured_players'], 
                                           state['static_walls'], 
                                           state['obstacles'],
                                           state['death_zones'],
                                           state['active_enemy_players'],
                                           state['friendly_players'],
                                           state['zap_obstacles'])
            if stepping_into_death(action, state['death_zones']):
                print('there is no hope')
                action = 0

        print('friendly colors', state['friendly_colors'])
        print('enemy colors', state['enemy_colors'])
        print('action', action_to_enum(action))

        # If player has fired claiming beam greater than the number of other players, must be enemy because
        # focals only fire claiming beams once per other player
        for color in PLAYER_COLOR_LIST:
            claim_count_key = 'num_'+color+'_claims'
            # print(claim_count_key, state[claim_count_key])
            if state[claim_count_key] > 8:
                state['friendly_colors'] = [val for val in state['friendly_colors'] if val != color]
                if color not in state['enemy_colors']:
                    state['enemy_colors'].append(color)

        return action, state
    
    def custom_step(self, timestep, state, prev_state):

        raw_timestep = timestep

        timestep = _downsample_single_timestep(timestep, 8)
        observation = timestep.observation['RGB']

        raw_observation = raw_timestep.observation['RGB']

        ready_to_shoot = timestep.observation['READY_TO_SHOOT'] == 1.0

        feature_dict = feature_detector_territory.\
            process_observation(observation, raw_observation, state['enemy_colors'], state['friendly_colors'])
        unclaimed_walls = feature_dict['unclaimed_walls']
        inactive_claimed_walls = feature_dict['inactive_claimed_walls']
        active_claimed_walls = feature_dict['active_claimed_walls']
        enemy_claimed_walls = feature_dict['enemy_claimed_walls']
        friendly_claimed_walls = feature_dict['friendly_claimed_walls']
        any_walls = feature_dict['any_walls']
        static_walls = feature_dict['static_walls']
        damaged_walls = feature_dict['damaged_walls']
        zaps = feature_dict['zaps']
        bug_obstacles = feature_dict['bug_obstacles'] # sometimes the dark X will remain after a background player dies and block the path
        players = feature_dict['players']
        player_colors = feature_dict['player_colors']
        player_directions = feature_dict['player_directions']
        injured_players = feature_dict['injured_players']
        injured_player_colors = feature_dict['injured_player_colors']
        injured_player_directions = feature_dict['injured_player_directions']
        state['is_injured'] = feature_dict['self_is_injured']
        player_color = feature_dict['player_color']

        death_zones = []
        reachable_enemy_players = []
        reachable_injured_enemy_players = []
        reachable_unclaimed_or_enemy_claimed_walls = []
        claiming_beams = get_claiming_beams(observation)

        # Add player positions/directions detected from paint brushes
        paint_brush_locations, paint_brush_directions, paint_brush_colors = find_paint_brushes(raw_observation)
        for i, brush_loc in enumerate(paint_brush_locations):
            if brush_loc not in players and brush_loc not in injured_players:
                players.append(brush_loc)
                player_directions.append(paint_brush_directions[i])
                player_colors.append(paint_brush_colors[i])
        for i, player_loc in enumerate(players):
            if player_loc in paint_brush_locations and player_directions[i] == 'unknown':
                player_directions[i] = paint_brush_directions[paint_brush_locations.index(player_loc)]
        for i, player_loc in enumerate(injured_players):
            if player_loc in paint_brush_locations and injured_player_directions[i] == 'unknown':
                injured_player_directions[i] = paint_brush_directions[paint_brush_locations.index(player_loc)]
        
        obstacles = list(set(unclaimed_walls + inactive_claimed_walls + active_claimed_walls + \
            enemy_claimed_walls + players + static_walls + damaged_walls + bug_obstacles + any_walls))
        zap_obstacles = [t for t in obstacles if t not in damaged_walls]
        state = update_enemy_steps_since_injured(state, prev_state, injured_player_colors)
        # active players are normal players + injured players who can move
        active_players, active_player_directions, active_player_colors = \
            update_active_players(state, players, player_colors, player_directions, 
                                  injured_players, injured_player_directions, injured_player_colors)
        active_enemy_players = [player for i, player in enumerate(active_players) if \
                                active_player_colors[i] in state['enemy_colors']]
        injured_enemy_players = [player for i, player in enumerate(injured_players) if \
                                 injured_player_colors[i] in state['enemy_colors']]
        friendly_players = [player for i, player in enumerate(players) if player_colors[i] in state['friendly_colors']]
        state = update_enemy_steps_since_zap(state, active_players, active_player_colors, active_player_directions, zaps, zap_obstacles)
        death_zones = get_death_zones(active_players, active_player_directions, active_player_colors, zap_obstacles, state)

        for i, player in enumerate(active_enemy_players):
            direction = get_direction_to_goal(player, obstacles)
            if direction:
                reachable_enemy_players.append(player)
        for player in injured_enemy_players:
            direction = get_direction_to_goal(player, obstacles)
            if direction:
                reachable_injured_enemy_players.append(player)
        for wall in unclaimed_walls+enemy_claimed_walls:
            direction = get_direction_to_goal(wall, obstacles)
            if direction:
                reachable_unclaimed_or_enemy_claimed_walls.append(wall)
        
        # Update state with observation info
        state['observation'] = observation
        state['active_players'] = active_players
        state['players'] = players
        state['player_colors'] = player_colors
        state['injured_players'] = injured_players
        state['static_walls'] = static_walls
        state['obstacles'] = obstacles
        state['death_zones'] = death_zones
        state['active_enemy_players'] = active_enemy_players
        state['friendly_players'] = friendly_players
        state['zap_obstacles'] = zap_obstacles

        # print('active players:', active_players)    
        # print('players:', players)
        # print('colors', player_colors)
        # print('directions:', player_directions)
        # print('injured players:', injured_players)
        # print('injured colors', injured_player_colors)
        # print('injured directions:', injured_player_directions)
        # print('ready to shoot:', ready_to_shoot)

        # Rotate at the start to spot invading enemies
        if state['step_count'] == 1:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=False, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 8, state
        if state['step_count'] == 2:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=True, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 6, state
        if state['step_count'] == 3:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=False, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 8, state
        if state['step_count'] == 4:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=True, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 6, state
        if state['step_count'] == 5:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=False, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 8, state
        if state['step_count'] == 6:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=True, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 6, state
        if state['step_count'] == 7:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=False, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            return 8, state
        if state['step_count'] == 8:
            enemy_colors, friendly_colors = \
                detect_enemy_at_start(active_players, expect_zap=True, claiming_beams=claiming_beams, 
                                      player_colors=active_player_colors)
            state['enemy_colors'] = list(set(state['enemy_colors'] + enemy_colors))
            state['friendly_colors'] = list(set(state['friendly_colors'] + friendly_colors))
            # At the final step, remove friendly colors from state['friendly_colors'] if they are in state['enemy_colors']
            state['friendly_colors'] = [color for color in state['friendly_colors'] if color not in state['enemy_colors']]
            return 6, state
        
        # Count number of claims each player has made
        if state['step_count'] > 8:
            for color in claiming_beams:
                state['num_'+color+'_claims'] += 1
        
        # Detect if handshake returned
        if state['steps_since_start_handshake'] < 2:
            print(player_color, 'waiting for handshake from', state['handshaking_with'], 'currently sees', claiming_beams.keys())
            for expected_return_color in state['handshaking_with']:
                if expected_return_color in claiming_beams:
                    state['friendly_colors'].append(expected_return_color)
            return 8, state
        if state['steps_since_start_handshake'] == 3:
                for expected_return_color in state['handshaking_with']:
                    if expected_return_color not in state['friendly_colors']:
                        print(player_color, 'did not receive handshake from', expected_return_color)
                        state['enemy_colors'].append(expected_return_color)
                state['handshaking_with'] = []

        # Handshake initiate (version where you initiate when both players see each other for the first time)
        # if (9, 5) not in death_zones and not state['handshaking_with']:
        mystery_player_colors, mystery_player_positions, mystery_player_directions = \
            get_mystery_players(player_colors, injured_player_colors, players, injured_players, 
                                player_directions, injured_player_directions, state)
        # determine if they can see you based on their position/direction
        handshake_player_colors, _, _ = \
            filter_to_both_players_visible(mystery_player_colors, mystery_player_positions, mystery_player_directions)
        num_mystery_players_who_can_see_you = len(handshake_player_colors)
        
        # initiate handshake 
        if num_mystery_players_who_can_see_you > 0 and set(handshake_player_colors) - set(state['handshaking_with']):
            print(player_color, 'initating handshake with', handshake_player_colors)
            state['handshaking_with'] = handshake_player_colors
            state['steps_since_start_handshake'] = 0
            return 8, state
            
        # If you have recently zapped, and can't see any additional enemies, rotate (until 4 rotations without seeing an enemy)
        if prev_state['zapped'] and not active_enemy_players and state['step_count'] - state['last_zap_step'] < 4:
            print('rotate after zapping')
            state['zapped'] = True
            return 5, state
        
        # Run from enemies if not ready to shoot, in death zone, or injured
        if reachable_enemy_players and (not ready_to_shoot or \
            (9, 5) in death_zones or \
            state['is_injured']):
            print('evading')
            return evade_enemy_direction(observation, 
                                        active_players, 
                                        injured_players, 
                                        static_walls, 
                                        obstacles,
                                        death_zones,
                                        active_enemy_players,
                                        friendly_players,
                                        zap_obstacles), state

        # Zapping enemies is top priority
        if player_in_zap_range(active_enemy_players, friendly_players, zap_obstacles) and ready_to_shoot:
            print('zapping enemy')
            state['last_zap_step'] = state['step_count']
            state['zapped'] = True
            return 7, state
        # Go toward enemies
        if active_enemy_players:
            nearest_player = get_nearest(reachable_enemy_players)
            direction = get_direction_to_goal(nearest_player, obstacles+death_zones) 
            print('nearest enemy', nearest_player, 'direction', direction)
            print('reachable enemies:', reachable_enemy_players)
            print('obstacles + death zones:', obstacles+death_zones)
            if direction:
                print('moving to get enemy in zap range. steps since last turn', state['steps_since_turn_toward_player'])
                # Turn toward enemy if on left
                if nearest_player[1] < 4 and nearest_player[0] > 3 and state['steps_since_turn_toward_player'] > 4:
                    state['steps_since_turn_toward_player'] = 0
                    return 5, state
                # Turn toward enemy if on right
                if nearest_player[1] > 6 and nearest_player[0] > 3 and state['steps_since_turn_toward_player'] > 4:
                    state['steps_since_turn_toward_player'] = 0
                    return 6, state
                # Turn around if enemy behind you
                if nearest_player[1] == 5 and nearest_player[0] > 9:
                    return 6, state
                return direction_to_number(direction), state
            
        # Zap injured enemies
        if player_in_zap_range(injured_enemy_players, friendly_players, zap_obstacles) and ready_to_shoot:
            print('zapping injured enemy')
            state['last_zap_step'] = state['step_count']
            state['zapped'] = True
            return 7, state
        if reachable_injured_enemy_players:
            nearest_player = get_nearest(reachable_injured_enemy_players)
            direction = get_direction_to_goal(nearest_player, obstacles+death_zones)
            if direction:
                print('moving to get injured enemy in zap range. steps since last turn', state['steps_since_turn_toward_player'])
                # Turn toward enemy if on left
                if nearest_player[1] < 4 and nearest_player[0] > 3 and state['steps_since_turn_toward_player'] > 4:
                    state['steps_since_turn_toward_player'] = 0
                    return 5, state
                # Turn toward enemy if on right
                if nearest_player[1] > 6 and nearest_player[0] > 3 and state['steps_since_turn_toward_player'] > 4:
                    state['steps_since_turn_toward_player'] = 0
                    return 6, state
                # Turn around if enemy behind you
                if nearest_player[1] == 5 and nearest_player[0] > 9:
                    return 6, state
                return direction_to_number(direction), state
                
        # Claim nearest reachable wall
        if reachable_unclaimed_or_enemy_claimed_walls:
            print('claiming stuff in my space')
            nearest_unclaimed = get_nearest(reachable_unclaimed_or_enemy_claimed_walls)
            if nearest_unclaimed == (9, 4):
                if (8, 4) in static_walls:
                    return 6, state
                return 5, state
            if nearest_unclaimed == (9, 6):
                if (8, 6) in static_walls:
                    return 5, state
                return 6, state
            if nearest_unclaimed == (10, 5):
                if (10, 6) in static_walls:
                    return 5, state
                return 6, state

            direction = get_direction_to_goal(nearest_unclaimed, obstacles+death_zones)
            if direction:
                return direction_to_number(direction), state

        # If it has been a while and the neighbor has not claimed their walls, destroy the wall and claim them
        unclaimed_or_enemy_claimed_walls = unclaimed_walls+enemy_claimed_walls
        if state['step_count'] > 70 and unclaimed_or_enemy_claimed_walls:
            print('claiming stuff after 70 steps')
            nearest_unclaimed = get_nearest(unclaimed_or_enemy_claimed_walls)
            
            # only continue if there are no friendly players closer to the target than you
            northwest_corner, orientation = get_northwest_corner(nearest_unclaimed, static_walls, raw_observation)
            space = get_space_from_corner(northwest_corner, orientation)
            im_closest = am_i_closest(northwest_corner, friendly_players)
            friendly_already_in_space = set(friendly_players) & set(space)
            # if there are friendlies and northwest corner is out of view, assume you aren't closest
            if friendly_players and (northwest_corner[0] < 0 or northwest_corner[0] > 10 or northwest_corner[1] < 0 or northwest_corner[1] > 10):
                im_closest = False
            print(player_color, 'closest', im_closest, ' northwest corner', northwest_corner)
            if im_closest and not friendly_already_in_space:
                if nearest_unclaimed == (9, 4):
                    return 5, state
                if nearest_unclaimed == (9, 6):
                    return 6, state
                if nearest_unclaimed == (10, 5):
                    return 6, state
                if nearest_unclaimed[0] == 9 and nearest_unclaimed[1] < 5:
                    return 5, state
                if nearest_unclaimed[0] == 9 and nearest_unclaimed[1] > 5:
                    return 6, state
                if nearest_unclaimed[0] > 9 and nearest_unclaimed[1] == 5:
                    return 6, state
                # AND obstacles is not a STATIC WALL
                destroyable_walls = unclaimed_walls + inactive_claimed_walls + active_claimed_walls + \
                    enemy_claimed_walls + damaged_walls + any_walls
                direction = get_direction_to_goal(nearest_unclaimed, static_walls+death_zones+friendly_players) 
                if direction == 'right' and (9, 6) in destroyable_walls:
                    return 6, state
                if direction == 'left' and (9, 4) in destroyable_walls:
                    return 5, state
                if direction == 'up' and (8, 5) in destroyable_walls and not player_in_zap_range(friendly_players, [], zap_obstacles):
                    return 7, state
                if direction == 'down':
                    return 6, state
                if direction:
                    # print('going to claim neighbor wall')
                    return direction_to_number(direction), state
                
        print('returning random')
        return get_random_move(friendly_claimed_walls), state

    def close(self):
        """ Required by base class """
        pass

