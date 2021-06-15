#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

#include "utils.hpp"

#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)

int world_size;
int rank;

Store store;

std::string rank_string() {
  return std::string("[") + std::to_string(rank) + "]";
}

namespace ucx {

ucp_context_h context;
ucp_worker_h worker;
std::vector<ucp_ep_h> eps;

void create_worker() { // create worker
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
  std::cout << rank_string() << "[UCX] Context initialized successfully."
            << std::endl;
  memset(&worker_params, 0, sizeof(ucp_worker_params_t));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(context, &worker_params, &worker);
  check(st == UCS_OK,
        std::string("failed to create UCP worker: ") + ucs_status_string(st));
  ucp_cleanup(context);
  std::cout << rank_string() << "[UCX] Worker created successfully."
            << std::endl;
}

void get_self_address() { // get self address
  ucs_status_t st;
  ucp_address_t *local_addr;
  size_t local_addr_len;

  st = ucp_worker_get_address(worker, &local_addr, &local_addr_len);
  check(st == UCS_OK, "failed to get worker address");
  std::vector<char> val =
      std::vector<char>(reinterpret_cast<char *>(local_addr),
                        reinterpret_cast<char *>(local_addr) + local_addr_len);
  ucp_worker_release_address(worker, local_addr);
  std::cout << rank_string() << "[UCX] Self address obtained." << std::endl;
  store.set("address:" + std::to_string(rank), val);
}

void create_endpoints() { // create endpoints
  eps.resize(world_size);
  for (int i = 0; i < world_size; i++) {
    std::vector<char> peer_addr = store.get("address:" + std::to_string(i));
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t *>(peer_addr.data());
    ucs_status_t st = ucp_ep_create(worker, &ep_params, &(eps[i]));
    check(st == UCS_OK, "failed to create endpoint");
  }
  std::cout << rank_string() << "[UCX] End points created." << std::endl;
}

} // namespace ucx

namespace ucc {

ucc_lib_h lib;
ucc_context_h context;
ucc_team_h team;

struct torch_ucc_oob_coll_info_t {
  int rank;
  void *rbuf;
  size_t msglen;
} oob;

ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                           void *coll_info, void **req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t *>(coll_info);
  std::vector<char> val = std::vector<char>(
      reinterpret_cast<char *>(sbuf), reinterpret_cast<char *>(sbuf) + msglen);
  store.set("teamr" + std::to_string(info->rank), val);
  info->rbuf = rbuf;
  info->msglen = msglen;
  *req = coll_info;
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void *req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t *>(req);

  for (int r = 0; r < world_size; r++) {
    if (!store.check({"teamr" + std::to_string(r)})) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < world_size; r++) {
    std::vector<char> data = store.get("teamr" + std::to_string(r));
    memcpy((void *)((ptrdiff_t)info->rbuf + info->msglen * r), data.data(),
           info->msglen);
  }
  return UCC_OK;
}

ucc_status_t oob_allgather_free(void *req) {
  // TODO: do something
  return UCC_OK;
}

void create_context() {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  auto oob_info = &oob;

  st = ucc_lib_config_read("TORCH", nullptr, &lib_config);
  check(st == UCC_OK,
        std::string("failed to read UCC lib config: ") + ucc_status_string(st));
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  st = ucc_init(&lib_params, lib_config, &lib);
  ucc_lib_config_release(lib_config);
  check(st == UCC_OK,
        std::string("failed to init UCC lib: ") + ucc_status_string(st));
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  st = ucc_lib_get_attr(lib, &lib_attr);
  check(st == UCC_OK,
        std::string("failed to query for lib attr: ") + ucc_status_string(st));
  check(lib_attr.thread_mode == UCC_THREAD_MULTIPLE,
        "ucc library wasn't initialized with mt support "
        "check ucc compile options ");
  st = ucc_context_config_read(lib, NULL, &context_config);
  check(st == UCC_OK, std::string("failed to read UCC context config: ") +
                          ucc_status_string(st));
  st = ucc_context_config_modify(context_config, NULL, "ESTIMATED_NUM_EPS",
                                 std::to_string(world_size).c_str());
  check(st == UCC_OK, std::string("failed to modify UCC context config: ") +
                          ucc_status_string(st));
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.participants = world_size;
  context_params.oob.allgather = oob_allgather;
  context_params.oob.req_test = oob_allgather_test;
  context_params.oob.req_free = oob_allgather_free;
  context_params.oob.coll_info = oob_info;
  ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  check(st == UCC_OK,
        std::string("failed to create UCC context: ") + ucc_status_string(st));
  std::cout << rank_string() << "[UCC] Context created." << std::endl;
}

void create_team() {
  auto oob_info = &oob;

  ucc_status_t st;
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
                     UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = oob_info;
  team_params.oob.participants = world_size;
  team_params.ep = rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  st = ucc_team_create_post(&context, 1, &team_params, &team);
  std::cout << rank_string() << "[UCC] Creating team..." << std::endl;
  check(st == UCC_OK,
        std::string("failed to post team create: ") + ucc_status_string(st));
  do {
    st = ucc_team_create_test(team);
  } while (st == UCC_INPROGRESS);
  check(st == UCC_OK,
        std::string("failed to create UCC team: ") + ucc_status_string(st));
  std::cout << rank_string() << "[UCC] Team created." << std::endl;
}

} // namespace ucc

int main(int argc, const char *argv[]) {
  check(argc == 3, "Bad argument");
  world_size = std::stoi(argv[1]);
  rank = std::stoi(argv[2]);

  cudaSetDevice(rank);

  ucx::create_worker();
  ucx::get_self_address();
  ucx::create_endpoints();

  ucc::create_context();
  ucc::create_team();

  std::this_thread::sleep_for(std::chrono::seconds(1));
}