#ifndef SF_LOGGER_HPP
#define SF_LOGGER_HPP

#include <ctime>
#include <string>
#include <string_view>

std::string get_timestamp();
void debug_log(std::string_view msg);
void debug_err(std::string_view msg);

#endif