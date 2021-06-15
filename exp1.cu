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

#include "utils.hpp"

#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)

int main(int argc, const char *argv[]) {
  check(argc == 3, "Bad argument");
  const int world_size = std::stoi(argv[1]);
  const int rank = std::stoi(argv[2]);

  cudaSetDevice(rank);

  ucp_context_h context;
  ucp_worker_h worker;
  Store store;
  std::vector<ucp_ep_h> eps;

  { // create worker
    ucp_params_t params;
    ucp_config_t *config;
    ucs_status_t st;
    ucp_worker_params_t worker_params;

    st = ucp_config_read(nullptr, nullptr, &config);
    // ucp_config_print(config, stdout, "UCP config", UCS_CONFIG_PRINT_CONFIG);
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

  {  // create endpoints
    eps.resize(world_size);
    for (int i = 0; i < world_size; i++) {
      std::vector<char> peer_addr = store.get("address:" + std::to_string(i));
      ucp_ep_params_t ep_params;
      ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
      ep_params.address = reinterpret_cast<ucp_address_t *>(peer_addr.data());
      ucs_status_t st = ucp_ep_create(worker, &ep_params, &(eps[i]));
      check(st == UCS_OK, "failed to create endpoint");
    }
  }
}