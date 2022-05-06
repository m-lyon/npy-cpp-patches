// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#include "src/pyparse.hpp"
#include <utility>  // std::pair


namespace pyparse {

/**
 Removes leading and trailing whitespaces
*/
std::string trim(const std::string& str) {
    const std::string whitespace = " \t";
    auto begin = str.find_first_not_of(whitespace);

    if (begin == std::string::npos)
        return "";

    auto end = str.find_last_not_of(whitespace);

    return str.substr(begin, end - begin + 1);
}


/**
 * @brief Gets the string value from a key/value pair string
 * 
 * @param mapstr key/value pair string
 * @return std::string value string
 */
std::string get_value_from_map(const std::string& mapstr) {
    size_t sep_pos = mapstr.find_first_of(":");
    if (sep_pos == std::string::npos)
        return "";

    std::string tmp = mapstr.substr(sep_pos + 1);
    return trim(tmp);
}


/**
 * @brief Parses a python dictionary from a string representation
 * 
 * @param in Input string to parse
 * @param keys Keys of dictionary to use
 * @return std::unordered_map<std::string, std::string> 
 */
std::unordered_map<std::string, std::string> parse_dict(
    std::string in, const std::vector<std::string>& keys
) {
    std::unordered_map<std::string, std::string> dict;

    if (keys.size() == 0)
        return dict;

    in = trim(in);

    // Strip parentheses
    if ((in.front() == '{') && (in.back() == '}'))
        in = in.substr(1, in.length() - 2);
    else
        throw std::runtime_error("Not a valid Python dictionary.");

    std::vector<std::pair<size_t, std::string>> positions;

    for (auto const& value : keys) {
        // Find position of key in string
        size_t pos = in.find("'" + value + "'");

        if (pos == std::string::npos) {
            throw std::runtime_error("Missing '" + value + "' key.");
        }

        std::pair<size_t, std::string> position_pair {pos, value};
        positions.push_back(position_pair);
    }

    // Sort by position in dict
    std::sort(positions.begin(), positions.end());

    for (size_t i = 0; i < positions.size(); ++i) {
        std::string raw_value;
        size_t begin {positions[i].first};
        size_t end {std::string::npos};

        std::string key = positions[i].second;

        // End of this key/value pair string is the beginning of the next
        if (i + 1 < positions.size()) {
            end = positions[i + 1].first;
        }

        raw_value = in.substr(begin, end - begin);
        raw_value = trim(raw_value);

        // Remove trailing comma
        if (raw_value.back() == ',') {
            raw_value.pop_back();
        }

        dict[key] = get_value_from_map(raw_value);
    }

    return dict;
}


/**
 * @brief Parses the string representation of a Python boolean
 * 
 * @param in Boolean string
 * @return Boolean value
 */
bool parse_bool(const std::string& in) {
    if (in == "True") {
        return true;
    }
    if (in == "False") {
        return false;
    }
    throw std::runtime_error("Invalid Python boolean.");
}


/**
 * @brief Parses the string representation of a Python str
 * 
 * @param in Input string
 * @return std::string parsed string
 */
std::string parse_str(const std::string& in) {
    if ((in.front() == '\'') && (in.back() == '\'')) {
        return in.substr(1, in.length() - 2);
    }
    throw std::runtime_error("Invalid Python string.");
}


/**
 * @brief Parses the string represenatation of a Python tuple into a vector of its items
 * 
 * @param in Tuple string
 * @return std::vector<std::string> Vector of strings
 */
std::vector<std::string> parse_tuple(std::string in) {
    std::vector<std::string> v;
    const char seperator = ',';

    in = trim(in);

    if ((in.front() == '(') && (in.back() == ')')) {
        in = in.substr(1, in.length() - 2);
    } else {
        throw std::runtime_error("Invalid Python tuple.");
    }

    std::istringstream iss(in);

    for (std::string token; std::getline(iss, token, seperator);) {
        v.push_back(trim(token));
    }

    return v;
}

}  // namespace pyparse
