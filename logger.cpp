#include "logger.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifndef NDEBUG
#include <cstdlib>
#endif

std::string get_timestamp() {
  auto now = std::chrono::system_clock::now();

  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm = *std::localtime(&now_c);

  std::ostringstream time;
  time << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S");

  return time.str();
}

void debug_log(std::string_view msg) {
#ifndef NDEBUG
  std::cout << get_timestamp() << ": " << msg << '\n';
#endif
}

void debug_err(std::string_view msg) {
#ifndef NDEBUG
  std::cerr << get_timestamp() << ": " << msg << '\n';
  exit(1);
#endif
}