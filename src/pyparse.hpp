// Copyright (c) 2022 Matthew Lyon. All rights reserved.
// Use of this source code is governed by an MIT-style license that can be
// found in the LICENSE file.

#ifndef PYPARSE_HPP_
#define PYPARSE_HPP_

#include <string>  // std::string
#include <unordered_map>  // std::unordered_map
#include <vector>  // std::vector
#include <algorithm>  // std::sort
#include <sstream>  // std::istringstream

namespace pyparse {

    std::string trim(const std::string&);
    std::string get_value_from_map(const std::string&);
    std::unordered_map<std::string, std::string> parse_dict(std::string,
                                                            const std::vector<std::string>&);
    bool parse_bool(const std::string&);
    std::string parse_str(const std::string&);
    std::vector <std::string> parse_tuple(std::string);

}  // namespace pyparse

#endif  // PYPARSE_HPP_
