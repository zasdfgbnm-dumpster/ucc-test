#include "utils.hpp"

#include <fstream>
#include <thread>
#include <filesystem>
#include <chrono>

std::ostream &operator<<(std::ostream &os, const std::vector<char> &value) {
  bool first = true;
  os << "[";
  for (char i : value) {
    if (!first) {
      os << ", ";
    }
    os << (int)i;
    first = false;
  }
  os << "]";
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::vector<std::string> &value) {
  bool first = true;
  os << "[";
  for (std::string i : value) {
    if (!first) {
      os << ", ";
    }
    os << i;
    first = false;
  }
  os << "]";
  return os;
}

void Store::set(const std::string &key, const std::vector<char> &value) {
  std::cout << "Store::set(" << key << ", " << value << ");" << std::endl;
  std::ofstream output(key + ".bin", std::ios::binary);
  std::copy(value.begin(), value.end(), std::ostreambuf_iterator<char>(output));
}

std::vector<char> Store::get(const std::string &key) {
  std::string filename = key + ".bin";
  while (!std::filesystem::exists(filename)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::ifstream input(filename, std::ios::binary);
  auto result = std::vector<char>(std::istreambuf_iterator<char>(input), {});
  std::cout << "Store::get(" << key << ") = " << result << std::endl;
  return result;
}

bool Store::check(const std::vector<std::string>& keys) {
  bool result = true;
  for (std::string key : keys) {
    std::string filename = key + ".bin";
    result = result && std::filesystem::exists(filename);
  }
  if (result) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "Store::check(" << keys << ") = " << result << ";" << std::endl;
  return result;
}
