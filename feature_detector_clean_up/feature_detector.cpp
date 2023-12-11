#include <Python.h>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <set>
#include <map>
#include <tuple>
#include <stdexcept>
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

const Color BLUE_PLAYER_RAW = {45, 110, 220};
const Color PINK_PLAYER_RAW = {205, 5, 165};
const Color PURPLE_PLAYER_RAW = {125, 50, 200};
const Color RED_PLAYER_RAW = {245, 65, 65};
const Color TEAL_PLAYER_RAW = {35, 185, 175};
const Color GREEN_PLAYER_RAW = {125, 185, 65};
const Color LIGHT_PURPLE_PLAYER_RAW = {160, 15, 200};
const Color YELLOW_PLAYER_RAW = {195, 180, 0};
const Color ORANGE_RAW = {245, 130, 0};
const Color LIME_GREEN_PLAYER_RAW = {180, 195, 0};
const Color LIGHT_RED_PLAYER_RAW = {230, 50, 95};
const Color DARK_ORANGE_PLAYER_RAW = {230, 90, 55};
const Color DARK_TEAL_PLAYER_RAW = {25, 170, 200};
const Color BROWN_PLAYER_RAW = {220, 140, 15};
const Color DARK_BLUE_PLAYER_RAW = {85, 80, 210};
const Color MINT_PLAYER_RAW = {25, 210, 140};
std::map<Color, std::string> RAW_PLAYER_COLOR_MAP = {
    {BLUE_PLAYER_RAW, "blue"},
    {PINK_PLAYER_RAW, "pink"},
    {PURPLE_PLAYER_RAW, "purple"},
    {RED_PLAYER_RAW, "red"},
    {TEAL_PLAYER_RAW, "teal"},
    {GREEN_PLAYER_RAW, "green"},
    {LIGHT_PURPLE_PLAYER_RAW, "light_purple"},
    {YELLOW_PLAYER_RAW, "yellow"},
    {ORANGE_RAW, "orange"},
    {LIME_GREEN_PLAYER_RAW, "lime_green"},
    {LIGHT_RED_PLAYER_RAW, "light_red"},
    {DARK_ORANGE_PLAYER_RAW, "dark_orange"},
    {DARK_TEAL_PLAYER_RAW, "dark_teal"},
    {BROWN_PLAYER_RAW, "brown"},
    {DARK_BLUE_PLAYER_RAW, "dark_blue"},
    {MINT_PLAYER_RAW, "mint"}
};

const Color YELLOW_PLAYER_WATER = {100, 146, 99};
const Color PURPLE_PLAYER_WATER = {104, 82, 159};
const Color RED_PLAYER_WATER = {118, 104, 122};
const Color ORANGE_PLAYER_WATER = {118, 128, 99};
const Color GREEN_PLAYER_WATER = {74, 148, 122};
const Color TEAL_PLAYER_WATER = {41, 148, 163};
const Color PINK_PLAYER_WATER = {104, 82, 159};
const Color LIGHT_PURPLE_PLAYER_WATER = {87, 85, 172};
const Color MINT_PLAYER_WATER = {38, 156, 151};
const Color LIME_GREEN_PLAYER_WATER = {89, 164, 92};
const Color DARK_BLUE_PLAYER_WATER = {60, 110, 176};
const Color LIGHT_RED_PLAYER_WATER = {113, 93, 130};
const Color BROWN_PLAYER_WATER = {106, 132, 107};
const Color DARK_ORANGE_PLAYER_WATER = {110, 114, 120};
const Color DARK_TEAL_PLAYER_WATER = {38, 142, 172};

const Color SAND = {220, 219, 187};
const Color SAND_DARK = {206, 205, 175};
const Color SAND_NEAR_WATER = {206, 205, 175};
const Color SAND_NEAR_DIRT = {214, 213, 182};
const std::vector<Color> SAND_COLORS = {SAND, SAND_DARK, SAND_NEAR_WATER, SAND_NEAR_DIRT};
const Color APPLE_COLOR = {170, 153, 69};
const Color APPLE_COLOR2 = {171, 153, 69};
const Color DIRT_COLOR = {28, 152, 147};
const Color WALL = {115, 115, 115};
const Color GRASS = {166, 191, 78};
const Color GRASS_BORDER = {168, 192, 81};
const Color GRASS_DARK = {157, 180, 76};
const Color GRASS_NEAR_WATER = {162, 186, 76};
const Color WATER = {34, 129, 163};
const Color WATER_DARK = {32, 121, 152};
const Color WATER_NEAR_GRASS = {34, 129, 162};
const Color APPLE_COLOR3 = {172, 154, 72};
const Color WATER2 = {34, 129, 162};
const std::vector<std::vector<Color>> APPLE_DIRECTIONS = {
    {SAND, GRASS},
    {SAND, GRASS_BORDER},
    {SAND_DARK, GRASS_DARK},
    {WATER_DARK, GRASS_NEAR_WATER},
    {WATER2, SAND_NEAR_DIRT},
    {WATER_NEAR_GRASS, GRASS_NEAR_WATER},
    {SAND_NEAR_WATER, GRASS_DARK},
    {DIRT_COLOR, SAND_NEAR_DIRT},
    {WATER, SAND_NEAR_DIRT}
};
const std::vector<std::vector<Color>> AWAY_FROM_WALL_DIRECTIONS = {
    {WALL, GRASS},
    {WALL, DIRT_COLOR},
    {WALL, WATER},
    {WALL, APPLE_COLOR},
    {WALL, APPLE_COLOR2},
    {WALL, GRASS_BORDER},
    {WALL, SAND},
    {WALL, APPLE_COLOR3},
    {WALL, WATER_DARK},
    {WALL, GRASS_NEAR_WATER},
    {WALL, SAND_NEAR_WATER},
    {WALL, SAND_NEAR_DIRT}
};

const Color EYE_COLOR = {60, 60, 60};

const int AGENT_VIEW_SIZE = 11;


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

bool is_within_range(const Color& a, const Color& b, int threshold = 5) {
    return std::abs(a.r - b.r) <= threshold &&
           std::abs(a.g - b.g) <= threshold &&
           std::abs(a.b - b.b) <= threshold;
}

bool any_within_range(const Color& color, const std::vector<Color>& color_list) {
    for (const auto& color_i : color_list) {
        if (is_within_range(color, color_i)) {
            return true;
        }
    }
    return false;
}

std::string get_player_direction(const std::array<std::array<Color, 88>, 88>& raw_rgb, int i, int j) {
    if (raw_rgb[i + 3][j + 2] == EYE_COLOR && raw_rgb[i + 3][j + 5] == EYE_COLOR) {
        return "down";
    } else if (raw_rgb[i + 3][j + 2] == EYE_COLOR && raw_rgb[i + 3][j + 4] == EYE_COLOR) {
        return "left";
    } else if (raw_rgb[i + 3][j + 3] == EYE_COLOR && raw_rgb[i + 3][j + 5] == EYE_COLOR) {
        return "right";
    }
    return "up";
}

std::tuple<bool, std::string, std::string> detect_player(const std::array<std::array<Color, 88>, 88>& raw_rgb, int i, int j) {
    if (i == 72 && j == 40) {
        return std::make_tuple(false, "", "");
    }

    Color rgb_val = raw_rgb[i + 5][j + 4];
    auto it = RAW_PLAYER_COLOR_MAP.find(rgb_val);
    if (it != RAW_PLAYER_COLOR_MAP.end()) {
        std::string color = it->second;
        std::string direction = get_player_direction(raw_rgb, i, j);
        return std::make_tuple(true, color, direction);
    }

    return std::make_tuple(false, "", "");
}

std::tuple<bool, std::string, std::string> detect_player_v2(const std::array<std::array<Color, 88>, 88>& raw_rgb, int i, int j) {
    /** 
    * Return is_player, player_color, player_dir by detecting if pixels belonging to the same body part are equal,
    * and verifying that each body part has a different color
    */
    if (i == 72 && j == 40) {
        return {false, "", ""};
    }

    // Define a lambda function to get the color of a pixel at a given offset
    auto get_color = [&raw_rgb, i, j](int oi, int oj) {
        return raw_rgb[i + oi][j + oj];
    };

    // Player color definition
    std::map<std::string, std::map<std::string, std::set<std::pair<int, int>>>> player_color_def = {
        {"up", {
            {"body_light", {{1, 2}, {1, 5},
                            {2, 2}, {2, 3}, {2, 4}, {2, 5},
                            {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5}, {4, 6},
                            {5, 2}, {5, 3}, {5, 4}, {5, 5},
                            {6, 2}, {6, 3}, {6, 4}, {6, 5}}},
            {"body_dark", {{3, 2}, {3, 3}, {3, 4}, {3, 5},
                        {5, 1}, {5, 6},
                        {7, 2}, {7, 5}}}
        }},
        {"down", {
            {"body_light", {{1, 2}, {1, 5},
                            {2, 2}, {2, 3}, {2, 4}, {2, 5},
                            {3, 3}, {3, 4},
                            {4, 2}, {4, 5},
                            {5, 2}, {5, 3}, {5, 4}, {5, 5},
                            {6, 2}, {6, 3}, {6, 4}, {6, 5}}},
            {"body_dark", {{4, 1}, {4, 6},
                        {5, 1}, {5, 6},
                        {7, 2}, {7, 5}}},
            {"eyes", {{3, 2}, {3, 5}}},
            {"mouth", {{4, 3}, {4, 4}}}
        }},
        {"right", {
            {"body_light", {{1, 2}, {1, 4},
                            {2, 2}, {2, 3}, {2, 4}, {2, 5},
                            {3, 2}, {3, 4},
                            {4, 1}, {4, 2}, {4, 5},
                            {5, 2}, {5, 3}, {5, 4}, {5, 5},
                            {6, 2}, {6, 3}, {6, 4}, {6, 5}}},
            {"body_dark", {{4, 6},
                        {5, 1}, {5, 6},
                        {7, 2}, {7, 3}, {7, 5}}},
            {"eyes", {{3, 3}, {3, 5}}},
            {"mouth", {{4, 3}, {4, 4}}}
        }},
        {"left", {
            {"body_light", {{1, 3}, {1, 5},
                            {2, 2}, {2, 3}, {2, 4}, {2, 5},
                            {3, 3}, {3, 5},
                            {4, 2}, {4, 5}, {4, 6},
                            {5, 2}, {5, 3}, {5, 4}, {5, 5},
                            {6, 2}, {6, 3}, {6, 4}, {6, 5}}},
            {"body_dark", {{4, 1},
                        {5, 1}, {5, 6},
                        {7, 2}, {7, 4}, {7, 5}}},
            {"eyes", {{3, 2}, {3, 4}}},
            {"mouth", {{4, 3}, {4, 4}}}
        }}
    };

    // Iterate over each player orientation
    for (const auto& [player_dir, body_parts] : player_color_def) {
        std::map<std::string, Color> colors;
        bool same_color_within_parts = true;

        // Check each body part for the same color
        for (const auto& [part_name, offsets] : body_parts) {
            std::set<Color> part_colors;
            for (const auto& offset : offsets) {
                part_colors.insert(get_color(offset.first, offset.second));
            }

            // If more than one color is found in a body part, it's not a player
            if (part_colors.size() != 1) {
                same_color_within_parts = false;
                break;
            }

            // Store the color for this part
            colors[part_name] = *part_colors.begin();
        }

        // Check for different colors across body parts
        if (same_color_within_parts && colors.size() == body_parts.size()) {
            Color color_array = colors["body_light"];

            // Iterate over RAW_PLAYER_COLOR_MAP to find a matching color
            for (const auto& color_map : RAW_PLAYER_COLOR_MAP) {
                if (color_array == color_map.first) {
                    std::string color_str = color_map.second;
                    return {true, color_str, player_dir};
                }
            }
        }
    }

    // No player detected
    return {false, "", ""};
}

std::string get_direction_from_pixel_transition(const std::array<std::array<Color, AGENT_VIEW_SIZE>, AGENT_VIEW_SIZE>& observation, const Color& color_from, const Color& color_to) {
    for (int i = 1; i < AGENT_VIEW_SIZE - 1; ++i) {
        for (int j = 1; j < AGENT_VIEW_SIZE - 1; ++j) {
            const Color& current_pixel = observation[i][j];
            const Color& left_pixel = observation[i][j - 1];
            const Color& right_pixel = observation[i][j + 1];
            const Color& up_pixel = observation[i - 1][j];
            const Color& down_pixel = observation[i + 1][j];

            if (is_within_range(current_pixel, color_to)) {
                if (is_within_range(right_pixel, color_from)) {
                    return "left";
                } else if (is_within_range(left_pixel, color_from)) {
                    return "right";
                } else if (is_within_range(up_pixel, color_from)) {
                    return "down";
                } else if (is_within_range(down_pixel, color_from)) {
                    return "up";
                }
            } else if (is_within_range(current_pixel, color_from)) {
                if (is_within_range(right_pixel, color_to)) {
                    return "right";
                } else if (is_within_range(left_pixel, color_to)) {
                    return "left";
                } else if (is_within_range(up_pixel, color_to)) {
                    return "up";
                } else if (is_within_range(down_pixel, color_to)) {
                    return "down";
                }
            }
        }
    }
    return "unknown";
}

std::string detect_direction(const std::array<std::array<Color, AGENT_VIEW_SIZE>, AGENT_VIEW_SIZE>& observation, const std::vector<std::vector<Color>>& pixel_transitions) {
    for (const auto& colors : pixel_transitions) {
        std::string direction = get_direction_from_pixel_transition(observation, colors[0], colors[1]);
        if (direction != "unknown") {
            return direction;
        }
    }
    return "unknown";
}

// Wrapper function for Python
static PyObject* process_observation_wrapper(PyObject* self, PyObject* args) {
    PyObject *pyObservation, *pyRawObservation;

    std::array<std::array<Color, AGENT_VIEW_SIZE>, AGENT_VIEW_SIZE> observation;
    std::array<std::array<Color, 88>, 88> raw_observation;

    // Parse the Python arguments to C types
    if (!PyArg_ParseTuple(args, "OO", &pyObservation, &pyRawObservation)) {
        return NULL;
    }

    // Convert Python objects to C++ types (You need to write this part)
    try {
        observation = convert_to_color_array<AGENT_VIEW_SIZE>(pyObservation);
        raw_observation = convert_to_color_array<88>(pyRawObservation);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }

    std::vector<std::pair<int, int>> apples;
    std::vector<std::pair<int, int>> dirt;
    std::vector<std::pair<int, int>> walls;
    std::vector<std::pair<int, int>> sand;
    std::vector<std::pair<int, int>> grass;
    std::vector<std::pair<int, int>> water;
    std::vector<std::pair<int, int>> players;
    std::vector<std::string> water_player_colors;
    std::vector<std::string> player_colors;
    std::vector<std::string> player_directions;
    std::string apple_direction = detect_direction(observation, APPLE_DIRECTIONS);
    std::string away_from_wall_direction = detect_direction(observation, AWAY_FROM_WALL_DIRECTIONS);

    for (int i = 0; i < AGENT_VIEW_SIZE; ++i) {
        for (int j = 0; j < AGENT_VIEW_SIZE; ++j) {
            const Color& current_color = observation[i][j];
            if (is_within_range(current_color, APPLE_COLOR) || is_within_range(current_color, APPLE_COLOR2)) {
                apples.emplace_back(i, j);
            } else if (is_within_range(current_color, DIRT_COLOR)) {
                dirt.emplace_back(i, j);
            } else if (is_within_range(current_color, WALL)) {
                walls.emplace_back(i, j);
            } else if (any_within_range(current_color, SAND_COLORS)) {
                sand.emplace_back(i, j);
            } else if (is_within_range(current_color, GRASS) || 
                    is_within_range(current_color, GRASS_BORDER) ||
                    is_within_range(current_color, GRASS_DARK) ||
                    is_within_range(current_color, GRASS_NEAR_WATER)) {
                grass.emplace_back(i, j);
            } else if (is_within_range(current_color, WATER) ||
                    is_within_range(current_color, WATER_DARK) ||
                    is_within_range(current_color, WATER_NEAR_GRASS)) {
                water.emplace_back(i, j);
            } else if (is_within_range(current_color, YELLOW_PLAYER_WATER, 8)) {
                water_player_colors.push_back("yellow");
            } else if (is_within_range(current_color, PURPLE_PLAYER_WATER, 8)) {
                water_player_colors.push_back("purple");
            } else if (is_within_range(current_color, RED_PLAYER_WATER, 8)) {
                water_player_colors.push_back("red");
            } else if (is_within_range(current_color, ORANGE_PLAYER_WATER, 8)) {
                water_player_colors.push_back("orange");
            } else if (is_within_range(current_color, GREEN_PLAYER_WATER, 8)) {
                water_player_colors.push_back("green");
            } else if (is_within_range(current_color, TEAL_PLAYER_WATER, 8)) {
                water_player_colors.push_back("teal");
            } else if (is_within_range(current_color, PINK_PLAYER_WATER, 8)) {
                water_player_colors.push_back("pink");
            } else if (is_within_range(current_color, LIGHT_PURPLE_PLAYER_WATER, 8)) {
                water_player_colors.push_back("light_purple");
            } else if (is_within_range(current_color, MINT_PLAYER_WATER, 8)) {
                water_player_colors.push_back("mint");
            } else if (is_within_range(current_color, LIME_GREEN_PLAYER_WATER, 8)) {
                water_player_colors.push_back("lime_green");
            } else if (is_within_range(current_color, DARK_BLUE_PLAYER_WATER, 8)) {
                water_player_colors.push_back("dark_blue");
            } else if (is_within_range(current_color, LIGHT_RED_PLAYER_WATER, 8)) {
                water_player_colors.push_back("light_red");
            } else if (is_within_range(current_color, BROWN_PLAYER_WATER, 8)) {
                water_player_colors.push_back("brown");
            } else if (is_within_range(current_color, DARK_ORANGE_PLAYER_WATER, 8)) {
                water_player_colors.push_back("dark_orange");
            } else if (is_within_range(current_color, DARK_TEAL_PLAYER_WATER, 8)) {
                water_player_colors.push_back("dark_teal");
            }
        }
    }

    for (int i = 0; i < 88; i += 8) {
        for (int j = 0; j < 88; j += 8) {
            auto [is_player, player_color, player_dir] = detect_player_v2(raw_observation, i, j);
            if (is_player) {
                players.emplace_back(i / 8, j / 8);
                player_colors.push_back(player_color);
                player_directions.push_back(player_dir);
            }
        }
    }

    // Convert C++ results back to Python objects
    PyObject* pyResult = PyDict_New();

    // Convert each vector to a Python list and add to the dictionary
    // Apples
    PyObject* pyApples = PyList_New(apples.size());
    for (size_t i = 0; i < apples.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", apples[i].first, apples[i].second);
        PyList_SetItem(pyApples, i, pyCoord); // Note: reference to pyCoord is stolen here
    }
    PyDict_SetItemString(pyResult, "apples", pyApples);
    Py_DECREF(pyApples);
    // Dirt
    PyObject* pyDirt = PyList_New(dirt.size());
    for (size_t i = 0; i < dirt.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", dirt[i].first, dirt[i].second);
        PyList_SetItem(pyDirt, i, pyCoord); // Note: reference to pyCoord is stolen here
    }
    PyDict_SetItemString(pyResult, "dirt", pyDirt);
    Py_DECREF(pyDirt);
    // Walls
    PyObject* pyWalls = PyList_New(walls.size());
    for (size_t i = 0; i < walls.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", walls[i].first, walls[i].second);
        PyList_SetItem(pyWalls, i, pyCoord); // Reference to pyCoord is stolen here
    }
    PyDict_SetItemString(pyResult, "walls", pyWalls);
    Py_DECREF(pyWalls);
    // Sand
    PyObject* pySand = PyList_New(sand.size());
    for (size_t i = 0; i < sand.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", sand[i].first, sand[i].second);
        PyList_SetItem(pySand, i, pyCoord); // Reference to pyCoord is stolen here
    }
    PyDict_SetItemString(pyResult, "sand", pySand);
    Py_DECREF(pySand);
    // Grass
    PyObject* pyGrass = PyList_New(grass.size());
    for (size_t i = 0; i < grass.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", grass[i].first, grass[i].second);
        PyList_SetItem(pyGrass, i, pyCoord); // Reference to pyCoord is stolen here
    }
    PyDict_SetItemString(pyResult, "grass", pyGrass);
    Py_DECREF(pyGrass);
    // Water
    PyObject* pyWater = PyList_New(water.size());
    for (size_t i = 0; i < water.size(); ++i) {
        PyObject* pyCoord = Py_BuildValue("(ii)", water[i].first, water[i].second);
        PyList_SetItem(pyWater, i, pyCoord); // Reference to pyCoord is stolen here
    }
    PyDict_SetItemString(pyResult, "water", pyWater);
    Py_DECREF(pyWater);

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

    // water player colors
    PyObject* pyWaterPlayerColors = PyList_New(water_player_colors.size());
    for (size_t i = 0; i < water_player_colors.size(); ++i) {
        PyObject* pyColor = PyUnicode_FromString(water_player_colors[i].c_str());
        PyList_SetItem(pyWaterPlayerColors, i, pyColor); // Note: reference to pyColor is stolen here
    }
    PyDict_SetItemString(pyResult, "water_player_colors", pyWaterPlayerColors);
    Py_DECREF(pyWaterPlayerColors);

    PyObject* pyAppleDirection = PyUnicode_FromString(apple_direction.c_str());
    PyDict_SetItemString(pyResult, "apple_direction", pyAppleDirection);
    Py_DECREF(pyAppleDirection);

    PyObject* pyAwayFromWallDirection = PyUnicode_FromString(away_from_wall_direction.c_str());
    PyDict_SetItemString(pyResult, "away_from_wall_direction", pyAwayFromWallDirection);
    Py_DECREF(pyAwayFromWallDirection);

    // Return the Python dictionary
    return pyResult;
}

// Define methods for the module
static PyMethodDef FeatureDetectorMethods[] = {
    {"process_observation", process_observation_wrapper, METH_VARARGS, "Process an observation"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Define the module
static struct PyModuleDef featuredetectorcleanupmodule  = {
    PyModuleDef_HEAD_INIT,
    "feature_detector_clean_up",
    NULL, // Module documentation
    -1,
    FeatureDetectorMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_feature_detector_clean_up(void) {
    import_array(); // Initialize numpy array API
    if (PyErr_Occurred()) {
        return NULL; // Return NULL if import_array failed
    }
    return PyModule_Create(&featuredetectorcleanupmodule);
}