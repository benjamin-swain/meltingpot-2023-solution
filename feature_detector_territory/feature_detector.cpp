#include <Python.h>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <utility> // For std::pair
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

struct Color {
    int r, g, b;

    bool operator==(const Color& other) const {
        return r == other.r && g == other.g && b == other.b;
    }

    bool operator<(const Color& other) const {
        return std::tie(r, g, b) < std::tie(other.r, other.g, other.b);
    }
};

const int AGENT_VIEW_SIZE = 11;
const Color UNCLAIMED_WALL = {68, 68, 68};
const Color DAMAGED_UNCLAIMED_WALL = {47, 47, 47};

// Walls
const Color INACTIVE_GREEN_WALL_MIN = {79, 97, 62};
const Color INACTIVE_GREEN_WALL_MAX = {97, 114, 75};
const Color ACTIVE_GREEN_WALL_MIN = {90, 129, 51};
const Color ACTIVE_GREEN_WALL_MAX = {106, 139, 66};
const Color INACTIVE_PURPLE_WALL_MIN = {79, 57, 101};
const Color INACTIVE_PURPLE_WALL_MAX = {98, 73, 118};
const Color ACTIVE_PURPLE_WALL_MIN = {90, 42, 139};
const Color ACTIVE_PURPLE_WALL_MAX = {107, 59, 150};
const Color INACTIVE_LIGHT_PURPLE_WALL_MIN = {90, 47, 101};
const Color INACTIVE_LIGHT_PURPLE_WALL_MAX = {109, 63, 116};
const Color ACTIVE_LIGHT_PURPLE_WALL_MIN = {114, 19, 134};
const Color ACTIVE_LIGHT_PURPLE_WALL_MAX = {127, 41, 149};
const Color INACTIVE_PINK_WALL_MIN = {103, 44, 91};
const Color INACTIVE_PINK_WALL_MAX = {123, 61, 107};
const Color ACTIVE_PINK_WALL_MIN = {142, 12, 116};
const Color ACTIVE_PINK_WALL_MAX = {157, 31, 128};
const Color INACTIVE_BLUE_WALL_MIN = {56, 75, 107};
const Color INACTIVE_BLUE_WALL_MAX = {75, 91, 120};
const Color ACTIVE_BLUE_WALL_MIN = {38, 80, 146};
const Color ACTIVE_BLUE_WALL_MAX = {61, 95, 162};
const Color INACTIVE_TEAL_WALL_MIN = {53, 97, 94};
const Color INACTIVE_TEAL_WALL_MAX = {69, 115, 110};
const Color ACTIVE_TEAL_WALL_MIN = {32, 129, 123};
const Color ACTIVE_TEAL_WALL_MAX = {51, 143, 134};
const Color INACTIVE_RED_WALL_MIN = {115, 62, 62};
const Color INACTIVE_RED_WALL_MAX = {136, 78, 75};
const Color ACTIVE_RED_WALL_MIN = {168, 51, 51};
const Color ACTIVE_RED_WALL_MAX = {182, 68, 65};
const Color INACTIVE_YELLOW_WALL_MIN = {100, 95, 42};
const Color INACTIVE_YELLOW_WALL_MAX = {118, 112, 55};
const Color ACTIVE_YELLOW_WALL_MIN = {136, 126, 9};
const Color ACTIVE_YELLOW_WALL_MAX = {147, 138, 28};
const Color INACTIVE_ORANGE_WALL_MIN = {115, 81, 42};
const Color INACTIVE_ORANGE_WALL_MAX = {136, 98, 56};
const Color ACTIVE_ORANGE_WALL_MIN = {165, 94, 9};
const Color ACTIVE_ORANGE_WALL_MAX = {178, 107, 29};

// Static walls
const Color STATIC_WALL_BOTTOM_RIGHT = {67, 63, 61};
const Color STATIC_WALL_BOTTOM_LEFT = {63, 59, 57};
const Color STATIC_WALL_TOP_LEFT = {59, 55, 53};
const Color STATIC_WALL_TOP_RIGHT = {63, 59, 57};

const Color BLUE_PLAYER = {45, 110, 220};
const Color PINK_PLAYER = {205, 5, 165};
const Color PURPLE_PLAYER = {125, 50, 200};
const Color RED_PLAYER = {245, 65, 65};
const Color TEAL_PLAYER = {35, 185, 175};
const Color GREEN_PLAYER = {125, 185, 65};
const Color LIGHT_PURPLE_PLAYER = {160, 15, 200};
const Color YELLOW_PLAYER = {195, 180, 0};
const Color ORANGE_PLAYER = {245, 130, 0};
std::vector<Color> PLAYER_COLORS = {
    BLUE_PLAYER, PINK_PLAYER, PURPLE_PLAYER, RED_PLAYER, TEAL_PLAYER, 
    GREEN_PLAYER, LIGHT_PURPLE_PLAYER, YELLOW_PLAYER, ORANGE_PLAYER
};

const Color PAINT_BRUSH_COLOR = {199, 176, 135};

const Color PAINT_HANDLE_BROWN_COLOR = {117, 79, 61};
const Color YELLOW_ARM = {146, 135, 0};
const Color GREEN_ARM = {93, 138, 48};
const Color RED_ARM = {183, 48, 48};
const Color TEAL_ARM = {26, 138, 131};
const Color LIGHT_PURPLE_ARM = {120, 11, 150};
const Color BLUE_ARM = {33, 82, 165};
const Color PURPLE_ARM = {93, 37, 150};
const Color ORANGE_ARM = {183, 97, 0};
const Color PINK_ARM = {153, 3, 123};
std::vector<Color> ARM_COLORS = {
    YELLOW_ARM, GREEN_ARM, RED_ARM, TEAL_ARM, LIGHT_PURPLE_ARM, 
    BLUE_ARM, PURPLE_ARM, ORANGE_ARM, PINK_ARM
};

const Color ZAP_COLOR = {252, 252, 106};
const Color BLACK = {0, 0, 0};

// Function to convert a Python object to std::array<std::array<Color, SIZE>, SIZE>
template <size_t SIZE>
std::array<std::array<Color, SIZE>, SIZE> convert_to_color_array(PyObject* pyArray) {
    std::array<std::array<Color, SIZE>, SIZE> result;

    if (!PyArray_Check(pyArray)) {
        throw std::invalid_argument("Input is not a numpy array");
    }

    auto* npArray = reinterpret_cast<PyArrayObject*>(pyArray);

    if (PyArray_NDIM(npArray) != 3 || PyArray_DIM(npArray, 0) != SIZE || 
        PyArray_DIM(npArray, 1) != SIZE || PyArray_DIM(npArray, 2) != 3) {
        throw std::invalid_argument("Input array has incorrect dimensions");
    }

    for (size_t  i = 0; i < SIZE; ++i) {
        for (size_t  j = 0; j < SIZE; ++j) {
            auto* pixel = reinterpret_cast<uint8_t*>(PyArray_GETPTR3(npArray, i, j, 0));
            result[i][j] = Color{static_cast<int>(pixel[0]), static_cast<int>(pixel[1]), static_cast<int>(pixel[2])};
        }
    }

    return result;
}

bool within_range(const Color& rgb_value, const Color& rgb_min, const Color& rgb_max) {
    if (rgb_value.r < rgb_min.r || rgb_value.g < rgb_min.g || rgb_value.b < rgb_min.b) {
        return false;
    }
    if (rgb_value.r > rgb_max.r || rgb_value.g > rgb_max.g || rgb_value.b > rgb_max.b) {
        return false;
    }
    return true;
}

bool is_enemy_claimed_wall(const Color& color, const std::vector<std::string>& enemy_colors) {
    if ((within_range(color, INACTIVE_GREEN_WALL_MIN, INACTIVE_GREEN_WALL_MAX) ||
         within_range(color, ACTIVE_GREEN_WALL_MIN, ACTIVE_GREEN_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "green") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_PURPLE_WALL_MIN, INACTIVE_PURPLE_WALL_MAX) ||
         within_range(color, ACTIVE_PURPLE_WALL_MIN, ACTIVE_PURPLE_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "purple") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_LIGHT_PURPLE_WALL_MIN, INACTIVE_LIGHT_PURPLE_WALL_MAX) ||
         within_range(color, ACTIVE_LIGHT_PURPLE_WALL_MIN, ACTIVE_LIGHT_PURPLE_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "light_purple") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_PINK_WALL_MIN, INACTIVE_PINK_WALL_MAX) ||
         within_range(color, ACTIVE_PINK_WALL_MIN, ACTIVE_PINK_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "pink") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_BLUE_WALL_MIN, INACTIVE_BLUE_WALL_MAX) ||
         within_range(color, ACTIVE_BLUE_WALL_MIN, ACTIVE_BLUE_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "blue") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_TEAL_WALL_MIN, INACTIVE_TEAL_WALL_MAX) ||
         within_range(color, ACTIVE_TEAL_WALL_MIN, ACTIVE_TEAL_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "teal") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_RED_WALL_MIN, INACTIVE_RED_WALL_MAX) ||
         within_range(color, ACTIVE_RED_WALL_MIN, ACTIVE_RED_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "red") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_YELLOW_WALL_MIN, INACTIVE_YELLOW_WALL_MAX) ||
         within_range(color, ACTIVE_YELLOW_WALL_MIN, ACTIVE_YELLOW_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "yellow") != enemy_colors.end()) {
        return true;
    }
    if ((within_range(color, INACTIVE_ORANGE_WALL_MIN, INACTIVE_ORANGE_WALL_MAX) ||
         within_range(color, ACTIVE_ORANGE_WALL_MIN, ACTIVE_ORANGE_WALL_MAX)) &&
         std::find(enemy_colors.begin(), enemy_colors.end(), "orange") != enemy_colors.end()) {
        return true;
    }

    return false;
}

bool is_any_wall(const Color& color) {
    if (within_range(color, INACTIVE_GREEN_WALL_MIN, INACTIVE_GREEN_WALL_MAX) ||
        within_range(color, ACTIVE_GREEN_WALL_MIN, ACTIVE_GREEN_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_PURPLE_WALL_MIN, INACTIVE_PURPLE_WALL_MAX) ||
        within_range(color, ACTIVE_PURPLE_WALL_MIN, ACTIVE_PURPLE_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_LIGHT_PURPLE_WALL_MIN, INACTIVE_LIGHT_PURPLE_WALL_MAX) ||
        within_range(color, ACTIVE_LIGHT_PURPLE_WALL_MIN, ACTIVE_LIGHT_PURPLE_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_PINK_WALL_MIN, INACTIVE_PINK_WALL_MAX) ||
        within_range(color, ACTIVE_PINK_WALL_MIN, ACTIVE_PINK_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_BLUE_WALL_MIN, INACTIVE_BLUE_WALL_MAX) ||
        within_range(color, ACTIVE_BLUE_WALL_MIN, ACTIVE_BLUE_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_TEAL_WALL_MIN, INACTIVE_TEAL_WALL_MAX) ||
        within_range(color, ACTIVE_TEAL_WALL_MIN, ACTIVE_TEAL_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_RED_WALL_MIN, INACTIVE_RED_WALL_MAX) ||
        within_range(color, ACTIVE_RED_WALL_MIN, ACTIVE_RED_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_YELLOW_WALL_MIN, INACTIVE_YELLOW_WALL_MAX) ||
        within_range(color, ACTIVE_YELLOW_WALL_MIN, ACTIVE_YELLOW_WALL_MAX)) {
        return true;
    }
    if (within_range(color, INACTIVE_ORANGE_WALL_MIN, INACTIVE_ORANGE_WALL_MAX) ||
        within_range(color, ACTIVE_ORANGE_WALL_MIN, ACTIVE_ORANGE_WALL_MAX)) {
        return true;
    }
    return false;
}

bool color_within_range(const Color& a, const Color& b, int threshold = 2) {
    return std::abs(a.r - b.r) <= threshold &&
           std::abs(a.g - b.g) <= threshold &&
           std::abs(a.b - b.b) <= threshold;
}

bool is_static_wall(const Color& color) {
    if (color_within_range(color, STATIC_WALL_BOTTOM_LEFT) ||
        color_within_range(color, STATIC_WALL_BOTTOM_RIGHT) ||
        color_within_range(color, STATIC_WALL_TOP_LEFT) ||
        color_within_range(color, STATIC_WALL_TOP_RIGHT)) {
        return true;
    }
    return false;
}

bool is_player(const Color& raw_color, const Color& downsampled_color) {
    auto it_raw = std::find(PLAYER_COLORS.begin(), PLAYER_COLORS.end(), raw_color);
    auto it_downsampled = std::find(PLAYER_COLORS.begin(), PLAYER_COLORS.end(), downsampled_color);

    // If the raw color is a player color and the downsampled color is not a player color
    if (it_raw != PLAYER_COLORS.end() && it_downsampled == PLAYER_COLORS.end()) {
        return true;
    }
    return false;
}

bool is_injured(const Color& raw_color) {
    return raw_color.r == 0 && raw_color.g == 0 && raw_color.b == 0;
}

bool is_color_in_vector(const Color& color, const std::vector<Color>& colors) {
    for (const auto& c : colors) {
        if (c.r == color.r && c.g == color.g && c.b == color.b) {
            return true;
        }
    }
    return false;
}

std::string try_detect_player_direction(const std::array<std::array<Color, 88>, 88>& raw_observation, int y, int x) {
    if (raw_observation[y + 1][x + 3] == PAINT_HANDLE_BROWN_COLOR) {
        return "down";
    }
    if (is_color_in_vector(raw_observation[y - 1][x - 4], ARM_COLORS) &&
        is_color_in_vector(raw_observation[y - 1][x - 3], ARM_COLORS)) {
        return "left";
    }
    if (is_color_in_vector(raw_observation[y - 1][x + 3], ARM_COLORS) &&
        is_color_in_vector(raw_observation[y - 1][x + 2], ARM_COLORS)) {
        return "right";
    }
    if (is_color_in_vector(raw_observation[y - 1][x - 3], ARM_COLORS) &&
        is_color_in_vector(raw_observation[y][x - 3], ARM_COLORS)) {
        return "down";
    }
    if (is_color_in_vector(raw_observation[y - 2][x + 1], ARM_COLORS) &&
        is_color_in_vector(raw_observation[y - 2][x + 2], ARM_COLORS)) {
        return "up";
    }
    return "unknown";
}

std::string get_player_color(const Color& my_color) {
    if (my_color == GREEN_PLAYER) {
        return "green";
    }
    if (my_color == PURPLE_PLAYER) {
        return "purple";
    }
    if (my_color == LIGHT_PURPLE_PLAYER) {
        return "light_purple";
    }
    if (my_color == PINK_PLAYER) {
        return "pink";
    }
    if (my_color == BLUE_PLAYER) {
        return "blue";
    }
    if (my_color == TEAL_PLAYER) {
        return "teal";
    }
    if (my_color == RED_PLAYER) {
        return "red";
    }
    if (my_color == YELLOW_PLAYER) {
        return "yellow";
    }
    if (my_color == ORANGE_PLAYER) {
        return "orange";
    }
    return "";
}

bool is_damaged_wall(int raw_i, int raw_j, const Color& raw_observation, const std::vector<std::pair<int, int>>& static_walls) {
    for (const auto& static_wall : static_walls) {
        if (raw_i / 8 == static_wall.first && raw_observation == Color{0, 0, 0}) {
            return true;
        }
        if (raw_j / 8 == static_wall.second && raw_observation == Color{0, 0, 0}) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> convert_pylist_to_vector(PyObject* pyList) {
    std::vector<std::string> result;
    if (PyList_Check(pyList)) {
        Py_ssize_t size = PyList_Size(pyList);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject* item = PyList_GetItem(pyList, i);
            if (PyUnicode_Check(item)) {
                result.push_back(PyUnicode_AsUTF8(item));
            }
        }
    }
    return result;
}

auto convert_vector_to_pylist = [](const std::vector<std::pair<int, int>>& vec) {
    PyObject* pyList = PyList_New(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", vec[i].first, vec[i].second);
        PyList_SetItem(pyList, i, pyCoord);  // Reference to pyCoord is stolen here
    }
    return pyList;
};

std::tuple<Color, Color, Color, Color> get_my_wall_colors(const std::string& player_color) {
    if (player_color == "green") {
        return {INACTIVE_GREEN_WALL_MIN, INACTIVE_GREEN_WALL_MAX, ACTIVE_GREEN_WALL_MIN, ACTIVE_GREEN_WALL_MAX};
    } else if (player_color == "purple") {
        return {INACTIVE_PURPLE_WALL_MIN, INACTIVE_PURPLE_WALL_MAX, ACTIVE_PURPLE_WALL_MIN, ACTIVE_PURPLE_WALL_MAX};
    } else if (player_color == "light_purple") {
        return {INACTIVE_LIGHT_PURPLE_WALL_MIN, INACTIVE_LIGHT_PURPLE_WALL_MAX, ACTIVE_LIGHT_PURPLE_WALL_MIN, ACTIVE_LIGHT_PURPLE_WALL_MAX};
    } else if (player_color == "pink") {
        return {INACTIVE_PINK_WALL_MIN, INACTIVE_PINK_WALL_MAX, ACTIVE_PINK_WALL_MIN, ACTIVE_PINK_WALL_MAX};
    } else if (player_color == "blue") {
        return {INACTIVE_BLUE_WALL_MIN, INACTIVE_BLUE_WALL_MAX, ACTIVE_BLUE_WALL_MIN, ACTIVE_BLUE_WALL_MAX};
    } else if (player_color == "teal") {
        return {INACTIVE_TEAL_WALL_MIN, INACTIVE_TEAL_WALL_MAX, ACTIVE_TEAL_WALL_MIN, ACTIVE_TEAL_WALL_MAX};
    } else if (player_color == "red") {
        return {INACTIVE_RED_WALL_MIN, INACTIVE_RED_WALL_MAX, ACTIVE_RED_WALL_MIN, ACTIVE_RED_WALL_MAX};
    } else if (player_color == "yellow") {
        return {INACTIVE_YELLOW_WALL_MIN, INACTIVE_YELLOW_WALL_MAX, ACTIVE_YELLOW_WALL_MIN, ACTIVE_YELLOW_WALL_MAX};
    } else if (player_color == "orange") {
        return {INACTIVE_ORANGE_WALL_MIN, INACTIVE_ORANGE_WALL_MAX, ACTIVE_ORANGE_WALL_MIN, ACTIVE_ORANGE_WALL_MAX};
    }

    // Default return value if no match is found
    return {Color{0, 0, 0}, Color{0, 0, 0}, Color{0, 0, 0}, Color{0, 0, 0}};
}



// Wrapper function for Python
static PyObject* process_observation_wrapper(PyObject* self, PyObject* args) {
    PyObject *pyObservation, *pyRawObservation, *pyEnemyColors, *pyFriendlyColors;

    std::array<std::array<Color, AGENT_VIEW_SIZE>, AGENT_VIEW_SIZE> observation;
    std::array<std::array<Color, 88>, 88> raw_observation;
    std::vector<std::string> enemy_colors, friendly_colors;

    // Parse the Python arguments to C types
    if (!PyArg_ParseTuple(args, "OOOO", &pyObservation, &pyRawObservation, &pyEnemyColors, &pyFriendlyColors)) {
        return NULL;
    }

    // Convert Python objects to C++ types (You need to write this part)
    try {
        observation = convert_to_color_array<AGENT_VIEW_SIZE>(pyObservation);
        raw_observation = convert_to_color_array<88>(pyRawObservation);
        enemy_colors = convert_pylist_to_vector(pyEnemyColors);
        friendly_colors = convert_pylist_to_vector(pyFriendlyColors);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    Color inactive_claimed_wall_color_min, inactive_claimed_wall_color_max;
    Color active_claimed_wall_color_min, active_claimed_wall_color_max;
    std::string player_color = get_player_color(raw_observation[77][44]);
    std::tie(inactive_claimed_wall_color_min, inactive_claimed_wall_color_max, 
         active_claimed_wall_color_min, active_claimed_wall_color_max) = get_my_wall_colors(player_color);

    std::vector<std::pair<int, int>> unclaimed_walls;
    std::vector<std::pair<int, int>> inactive_claimed_walls;
    std::vector<std::pair<int, int>> active_claimed_walls;
    std::vector<std::pair<int, int>> enemy_claimed_walls;
    std::vector<std::pair<int, int>> friendly_claimed_walls;
    std::vector<std::pair<int, int>> any_walls;
    std::vector<std::pair<int, int>> players;
    std::vector<std::string> player_directions;
    std::vector<std::string> player_colors;
    std::vector<std::pair<int, int>> injured_players;
    std::vector<std::string> injured_player_directions;
    std::vector<std::string> injured_player_colors;
    std::vector<std::pair<int, int>> static_walls;
    std::vector<std::pair<int, int>> damaged_walls;
    std::vector<std::pair<int, int>> zaps;
    std::vector<std::pair<int, int>> bug_obstacles;
    bool self_is_injured = false;

    for (int i = 0; i < 11; ++i) {
        for (int j = 0; j < 11; ++j) {
            if (i == 9 && j == 5) {
                continue;
            }
            const Color& current_color = observation[i][j];
            if (current_color == UNCLAIMED_WALL || current_color == DAMAGED_UNCLAIMED_WALL) {
                unclaimed_walls.emplace_back(i, j);
            } else if (within_range(current_color, inactive_claimed_wall_color_min, inactive_claimed_wall_color_max)) {
                inactive_claimed_walls.emplace_back(i, j);
            } else if (within_range(current_color, active_claimed_wall_color_min, active_claimed_wall_color_max)) {
                active_claimed_walls.emplace_back(i, j);
            }
            if (is_enemy_claimed_wall(current_color, enemy_colors)) {
                enemy_claimed_walls.emplace_back(i, j);
            }
            if (is_enemy_claimed_wall(current_color, friendly_colors)) {
                friendly_claimed_walls.emplace_back(i, j);
            }
            if (is_any_wall(current_color)) {
                any_walls.emplace_back(i, j);
            }
            if (is_static_wall(current_color)) {
                static_walls.emplace_back(i, j);
            }
        }
    }
    for (int i = 5; i < 86; i += 8) {
        for (int j = 4; j < 85; j += 8) {
            if (i != 77 || j != 44) {
                const Color& raw_color = raw_observation[i][j];
                const Color& downsampled_color = observation[i / 8][j / 8];

                if (is_player(raw_color, downsampled_color)) {
                    if (is_injured(raw_observation[i - 1][j])) {
                        injured_players.emplace_back(i / 8, j / 8);
                        injured_player_directions.push_back(try_detect_player_direction(raw_observation, i, j));
                        injured_player_colors.push_back(get_player_color(raw_color));
                    } else {
                        players.emplace_back(i / 8, j / 8);
                        player_directions.push_back(try_detect_player_direction(raw_observation, i, j));
                        player_colors.push_back(get_player_color(raw_color));
                    }
                }
            }

            if (is_damaged_wall(i, j, raw_observation[i][j], static_walls)) {
                damaged_walls.emplace_back(i / 8, j / 8);
            }

            if ((i == 77 && j == 44) && is_injured(raw_observation[i - 1][j])) {
                self_is_injured = true;
            }
        }
    }
    for (int j = 6; j < 94; j += 8) {
        if (raw_observation[87][j - 1] == PAINT_BRUSH_COLOR &&
            raw_observation[87][j] == PAINT_BRUSH_COLOR &&
            raw_observation[87][j + 1] == PAINT_BRUSH_COLOR &&
            raw_observation[86][j] == PAINT_BRUSH_COLOR) {
            players.emplace_back(11, j / 8);
            player_directions.push_back("up");
            player_colors.push_back(get_player_color(raw_observation[85][j - 1]));
        }
    }
    for (int i = 0; i < 88; i += 8) {
        for (int j = 0; j < 88; j += 8) {
            if (raw_observation[i][j] == ZAP_COLOR) {
                zaps.emplace_back(i / 8, j / 8);
            }
            if (raw_observation[i + 3][j + 3] == BLACK &&
                raw_observation[i + 4][j + 3] == BLACK &&
                raw_observation[i + 3][j + 4] == BLACK &&
                raw_observation[i + 4][j + 4] == BLACK) {
                bug_obstacles.emplace_back(i / 8, j / 8);
            }
        }
    }

    // Convert C++ results back to Python objects
    PyObject* pyResult = PyDict_New();

    // Convert each vector to a Python list and add to the dictionary
    // Unclaimed Walls
    PyObject* pyUnclaimedWalls = convert_vector_to_pylist(unclaimed_walls);
    PyDict_SetItemString(pyResult, "unclaimed_walls", pyUnclaimedWalls);
    Py_DECREF(pyUnclaimedWalls);
    // Inactive Claimed Walls
    PyObject* pyInactiveClaimedWalls = convert_vector_to_pylist(inactive_claimed_walls);
    PyDict_SetItemString(pyResult, "inactive_claimed_walls", pyInactiveClaimedWalls);
    Py_DECREF(pyInactiveClaimedWalls);
    // Active Claimed Walls
    PyObject* pyActiveClaimedWalls = convert_vector_to_pylist(active_claimed_walls);
    PyDict_SetItemString(pyResult, "active_claimed_walls", pyActiveClaimedWalls);
    Py_DECREF(pyActiveClaimedWalls);
    // Enemy Claimed Walls
    PyObject* pyEnemyClaimedWalls = convert_vector_to_pylist(enemy_claimed_walls);
    PyDict_SetItemString(pyResult, "enemy_claimed_walls", pyEnemyClaimedWalls);
    Py_DECREF(pyEnemyClaimedWalls);
    // Friendly Claimed Walls
    PyObject* pyFriendlyClaimedWalls = convert_vector_to_pylist(friendly_claimed_walls);
    PyDict_SetItemString(pyResult, "friendly_claimed_walls", pyFriendlyClaimedWalls);
    Py_DECREF(pyFriendlyClaimedWalls);
    // Any Walls
    PyObject* pyAnyWalls = convert_vector_to_pylist(any_walls);
    PyDict_SetItemString(pyResult, "any_walls", pyAnyWalls);
    Py_DECREF(pyAnyWalls);
    // Static Walls
    PyObject* pyStaticWalls = convert_vector_to_pylist(static_walls);
    PyDict_SetItemString(pyResult, "static_walls", pyStaticWalls);
    Py_DECREF(pyStaticWalls);
    // Damaged Walls
    PyObject* pyDamagedWalls = convert_vector_to_pylist(damaged_walls);
    PyDict_SetItemString(pyResult, "damaged_walls", pyDamagedWalls);
    Py_DECREF(pyDamagedWalls);
    // Zaps
    PyObject* pyZaps = convert_vector_to_pylist(zaps);
    PyDict_SetItemString(pyResult, "zaps", pyZaps);
    Py_DECREF(pyZaps);
    // Bug Obstacles
    PyObject* pyBugObstacles = convert_vector_to_pylist(bug_obstacles);
    PyDict_SetItemString(pyResult, "bug_obstacles", pyBugObstacles);
    Py_DECREF(pyBugObstacles);

    // Convert players to Python lists
    PyObject* pyPlayers = PyList_New(players.size());
    PyObject* pyPlayerColors = PyList_New(player_colors.size());
    PyObject* pyPlayerDirections = PyList_New(player_directions.size());
    for (size_t i = 0; i < players.size(); ++i) {
        PyObject* pyPlayer = Py_BuildValue("(ii)", players[i].first, players[i].second);
        PyList_SetItem(pyPlayers, i, pyPlayer); // Reference stolen

        PyObject* pyColor = PyUnicode_FromString(player_colors[i].c_str());
        PyList_SetItem(pyPlayerColors, i, pyColor); // Reference stolen

        PyObject* pyDirection = PyUnicode_FromString(player_directions[i].c_str());
        PyList_SetItem(pyPlayerDirections, i, pyDirection); // Reference stolen
    }
    // Add the lists to the dictionary
    PyDict_SetItemString(pyResult, "players", pyPlayers);
    Py_DECREF(pyPlayers);
    PyDict_SetItemString(pyResult, "player_colors", pyPlayerColors);
    Py_DECREF(pyPlayerColors);
    PyDict_SetItemString(pyResult, "player_directions", pyPlayerDirections);
    Py_DECREF(pyPlayerDirections);

    PyObject* pyInjuredPlayers = PyList_New(injured_players.size());
    PyObject* pyInjuredPlayerColors = PyList_New(injured_player_colors.size());
    PyObject* pyInjuredPlayerDirections = PyList_New(injured_player_directions.size());
    for (size_t i = 0; i < injured_players.size(); ++i) {
        PyObject* pyInjuredPlayer = Py_BuildValue("(ii)", injured_players[i].first, injured_players[i].second);
        PyList_SetItem(pyInjuredPlayers, i, pyInjuredPlayer);  // Reference stolen

        PyObject* pyInjuredColor = PyUnicode_FromString(injured_player_colors[i].c_str());
        PyList_SetItem(pyInjuredPlayerColors, i, pyInjuredColor);  // Reference stolen

        PyObject* pyInjuredDirection = PyUnicode_FromString(injured_player_directions[i].c_str());
        PyList_SetItem(pyInjuredPlayerDirections, i, pyInjuredDirection);  // Reference stolen
    }
    // Add the lists to the dictionary
    PyDict_SetItemString(pyResult, "injured_players", pyInjuredPlayers);
    Py_DECREF(pyInjuredPlayers);
    PyDict_SetItemString(pyResult, "injured_player_colors", pyInjuredPlayerColors);
    Py_DECREF(pyInjuredPlayerColors);
    PyDict_SetItemString(pyResult, "injured_player_directions", pyInjuredPlayerDirections);
    Py_DECREF(pyInjuredPlayerDirections);

    PyObject* pySelfIsInjured = PyBool_FromLong(static_cast<long>(self_is_injured));
    PyDict_SetItemString(pyResult, "self_is_injured", pySelfIsInjured);
    Py_DECREF(pySelfIsInjured);

    PyObject* pyPlayerColor = PyUnicode_FromString(player_color.c_str());
    PyDict_SetItemString(pyResult, "player_color", pyPlayerColor);
    Py_DECREF(pyPlayerColor);

    // Return the Python dictionary
    return pyResult;
}

// Define methods for the module
static PyMethodDef FeatureDetectorMethods[] = {
    {"process_observation", process_observation_wrapper, METH_VARARGS, "Process an observation"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Define the module
static struct PyModuleDef featuredetectorterritorymodule = {
    PyModuleDef_HEAD_INIT,
    "feature_detector_territory",
    NULL, // Module documentation
    -1,
    FeatureDetectorMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_feature_detector_territory(void) {
    import_array(); // Initialize numpy array API
    if (PyErr_Occurred()) {
        return NULL; // Return NULL if import_array failed
    }
    return PyModule_Create(&featuredetectorterritorymodule);
}