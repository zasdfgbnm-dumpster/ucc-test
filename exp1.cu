#include <iostream>
#include <stdexcept>
#include <vector>

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

int main() {
  ucp_context_h context;
  ucp_worker_h worker;

  {
    ucp_params_t params;
    ucp_config_t *config;
    ucs_status_t st;
    ucp_worker_params_t worker_params;

    const int comm_size = 2;

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
    params.estimated_num_eps = comm_size;
    params.tag_sender_mask = TORCH_UCX_RANK_MASK;
    params.request_init = [](void *request) {
      static_cast<ucc_coll_req_h>(request)->status = UCC_INPROGRESS;
    };
    params.request_cleanup = [](void *) {};
    st = ucp_init(&params, config, &context);
    ucp_config_release(config);
    check(st == UCS_OK,
          std::string("failed to init UCP context: ") + ucs_status_string(st));
    memset(&worker_params, 0, sizeof(ucp_worker_params_t));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    st = ucp_worker_create(context, &worker_params, &worker);
    check(st == UCS_OK,
          std::string("failed to create UCP worker: ") + ucs_status_string(st));
    ucp_cleanup(context);
  }

  {
    ucs_status_t st;
    ucp_address_t *local_addr;
    size_t local_addr_len;
    std::vector<uint8_t> peer_addr;

    st = ucp_worker_get_address(worker, &local_addr, &local_addr_len);
    check(st == UCS_OK, "failed to get worker address");
    std::vector<uint8_t> val = std::vector<uint8_t>(
        reinterpret_cast<uint8_t *>(local_addr),
        reinterpret_cast<uint8_t *>(local_addr) + local_addr_len);

    for (uint8_t i : val) {
      std::cout << i << ", ";
    }
    std::cout << std::endl;

    ucp_worker_release_address(worker, local_addr);
  }
}