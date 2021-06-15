#include <iostream>
#include <string>
#include <vector>

struct Store {
  void set(const std::string &key, const std::vector<char> &value);
  std::vector<char> get(const std::string &key);
  bool check(const std::vector<std::string>& keys);
};

std::ostream &operator<<(std::ostream &os, const std::vector<char> &value);
std::ostream &operator<<(std::ostream &os, const std::vector<std::string> &value);
