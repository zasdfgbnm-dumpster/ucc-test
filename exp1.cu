#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <chrono>
#include <thread>
#include <string>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

void check(bool condition, std::string msg = "") {
  if (!condition) {
    throw std::runtime_error(msg);
  }
}

#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)

struct Store {
  void set(const std::string &key, const std::vector<char> &value);
  std::vector<char> get(const std::string &key);
  bool check(const std::vector<std::string>& keys);
};

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

int main(int argc, const char *argv[]) {
  check(argc == 3, "Bad argument");
  const int world_size = std::stoi(argv[1]);
  const int rank = std::stoi(argv[2]);

  ucp_context_h context;
  ucp_worker_h worker;
  Store store;

  { // create worker
    ucp_params_t params;
    ucp_config_t *config;
    ucs_status_t st;
    ucp_worker_params_t worker_params;

    st = ucp_config_read("TORCH", nullptr, &config);
    check(st == UCS_OK,
          std::string("failed to read UCP config: ") + ucs_status_string(st));
    memset(&params, 0, sizeof(ucp_params_t));
    params.field_mask =
        UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE |
        UCP_PARAM_FIELD_ESTIMATED_NUM_EPS | UCP_PARAM_FIELD_TAG_SENDER_MASK |
        UCP_PARAM_FIELD_REQUEST_INIT | UCP_PARAM_FIELD_REQUEST_CLEANUP;
    params.request_size = sizeof(ucc_coll_req_t);
    params.features = UCP_FEATURE_TAG;
    params.estimated_num_eps = world_size;
    params.tag_sender_mask = TORCH_UCX_RANK_MASK;
    params.request_init = [](void *request) {
      static_cast<ucc_coll_req_h>(request)->status = UCC_INPROGRESS;
    };
    params.request_cleanup = [](void *) {};
    st = ucp_init(&params, config, &context);
    ucp_config_release(config);
    check(st == UCS_OK,
          std::string("failed to init UCP context: ") + ucs_status_string(st));
    std::cout << "Context initialized successfully." << std::endl;
    memset(&worker_params, 0, sizeof(ucp_worker_params_t));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    st = ucp_worker_create(context, &worker_params, &worker);
    check(st == UCS_OK,
          std::string("failed to create UCP worker: ") + ucs_status_string(st));
    ucp_cleanup(context);
    std::cout << "Worker created successfully." << std::endl;
  }

  { // get self address
    ucs_status_t st;
    ucp_address_t *local_addr;
    size_t local_addr_len;

    st = ucp_worker_get_address(worker, &local_addr, &local_addr_len);
    check(st == UCS_OK, "failed to get worker address");
    std::vector<char> val = std::vector<char>(
        reinterpret_cast<char *>(local_addr),
        reinterpret_cast<char *>(local_addr) + local_addr_len);
    ucp_worker_release_address(worker, local_addr);
    std::cout << "Self address obtained." << std::endl;
    store.set("address:" + std::to_string(rank), val);
  }

  {  // get peer address

  }
}