#include "pyparse.hpp"


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


    std::string get_value_from_map(const std::string& mapstr) {
        size_t sep_pos = mapstr.find_first_of(":");
        if (sep_pos == std::string::npos)
            return "";

        std::string tmp = mapstr.substr(sep_pos + 1);
        return trim(tmp);
    }


    /**
     Parses the string representation of a Python dict

    The keys need to be known and may not appear anywhere else in the data.
    */
    std::unordered_map<std::string, std::string> parse_dict(
        std::string in, const std::vector<std::string>& keys
    ){
        std::unordered_map <std::string, std::string> map;

        if (keys.size() == 0)
            return map;

        in = trim(in);

        // unwrap dictionary
        if ((in.front() == '{') && (in.back() == '}'))
            in = in.substr(1, in.length() - 2);
        else
            throw std::runtime_error("Not a Python dictionary.");

        std::vector <std::pair<size_t, std::string>> positions;

        for (auto const &value : keys) {
            size_t pos = in.find("'" + value + "'");

            if (pos == std::string::npos)
            throw std::runtime_error("Missing '" + value + "' key.");

            std::pair <size_t, std::string> position_pair{pos, value};
            positions.push_back(position_pair);
        }

        // sort by position in dict
        std::sort(positions.begin(), positions.end());

        for (size_t i = 0; i < positions.size(); ++i) {
            std::string raw_value;
            size_t begin{positions[i].first};
            size_t end{std::string::npos};

            std::string key = positions[i].second;

            if (i + 1 < positions.size())
            end = positions[i + 1].first;

            raw_value = in.substr(begin, end - begin);

            raw_value = trim(raw_value);

            if (raw_value.back() == ',')
            raw_value.pop_back();

            map[key] = get_value_from_map(raw_value);
        }

        return map;
    }


    /**
        Parses the string representation of a Python boolean
    */
    bool parse_bool(const std::string& in) {
        if (in == "True"){
            return true;
        }
        if (in == "False"){
            return false;
        }
        throw std::runtime_error("Invalid python boolan.");
    }


    /**
        Parses the string representation of a Python str
    */
    std::string parse_str(const std::string& in) {
        if ((in.front() == '\'') && (in.back() == '\''))
            return in.substr(1, in.length() - 2);
        throw std::runtime_error("Invalid python string.");
    }


    /**
        Parses the string represenatation of a Python tuple into a vector of its items
    */
    std::vector <std::string> parse_tuple(std::string in) {
        std::vector <std::string> v;
        const char seperator = ',';

        in = trim(in);

        if ((in.front() == '(') && (in.back() == ')'))
            in = in.substr(1, in.length() - 2);
        else
            throw std::runtime_error("Invalid Python tuple.");

        std::istringstream iss(in);

        for (std::string token; std::getline(iss, token, seperator);) {
            v.push_back(token);
        }

        return v;
    }

}