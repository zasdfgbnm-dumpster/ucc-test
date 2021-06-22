#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <memory>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

#include "utils.hpp"

using T = int;
extern int world_size;
extern int rank;
extern T *input;
extern T *output;
extern int N;

ucc_datatype_t dtype = UCC_DT_FLOAT32;

Store store{false};

std::string rank_string() {
  return std::string("[") + std::to_string(rank) + "]";
}

void check_cuda(cudaError_t);

cudaStream_t getStreamFromPool();
cudaStream_t getCurrentCUDAStream();

namespace ucc {

ucc_lib_h lib;
ucc_context_h context;
ucc_team_h team;
ucc_ee_h cuda_ee;
ucc_coll_req_h request;

std::shared_ptr<cudaStream_t> stream;
std::shared_ptr<cudaEvent_t> cuda_ev;

std::vector<ucc_count_t> lengths;
std::vector<ucc_aint_t> offsets;

struct torch_ucc_oob_coll_info_t {
  void *rbuf;
};

ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                           void *coll_info, void **req) {
  static int message_id = 0;
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t *>(coll_info);
  std::vector<char> val = std::vector<char>(
      reinterpret_cast<char *>(sbuf), reinterpret_cast<char *>(sbuf) + msglen);
  store.set("teamr:" + std::to_string(rank) + "," + std::to_string(message_id++), val);
  info->rbuf = rbuf;
  std::cout << rank_string() << "[UCC] OOB all gather, msglen: " << msglen << std::endl;
  *req = coll_info;
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void *req) {
  static int message_id = 0;
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t *>(req);

  for (int r = 0; r < world_size; r++) {
    if (!store.check({"teamr:" + std::to_string(r) + "," + std::to_string(message_id)})) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < world_size; r++) {
    std::vector<char> data = store.get("teamr:" + std::to_string(r) + "," + std::to_string(message_id));
    int msglen = data.size();
    memcpy((void *)((ptrdiff_t)info->rbuf + msglen * r), data.data(), msglen);
  }
  message_id++;
  return UCC_OK;
}

ucc_status_t oob_allgather_free(void *req) {
  return UCC_OK;
}

void create_context() {
  std::shared_ptr<torch_ucc_oob_coll_info_t> coll_info = std::make_shared<torch_ucc_oob_coll_info_t>();
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  st = ucc_lib_config_read(nullptr, nullptr, &lib_config);
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
  context_params.oob.coll_info = coll_info.get();
  ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  check(st == UCC_OK,
        std::string("failed to create UCC context: ") + ucc_status_string(st));
  std::cout << rank_string() << "[UCC] Context created." << std::endl;
}

void create_team() {
  ucc_status_t st;
  std::shared_ptr<torch_ucc_oob_coll_info_t> coll_info = std::make_shared<torch_ucc_oob_coll_info_t>();
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
                     UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = coll_info.get();
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

void create_cuda_ee() {
  ucc_ee_params_t params;
  stream = std::make_shared<cudaStream_t>(getStreamFromPool());
  params.ee_type = UCC_EE_CUDA_STREAM;
  params.ee_context = (void *)(*stream);
  params.ee_context_size = sizeof(cudaStream_t);
  ucc_status_t st = ucc_ee_create(team, &params, &cuda_ee);
  check(st == UCC_OK,
        std::string("failed to create UCC EE: ") + ucc_status_string(st));
  std::cout << rank_string() << "[UCC] CUDA ee created." << std::endl;
}

void compute_lengths_and_offsets() {
  for (int i = 0; i < world_size; i++) {
    lengths.push_back(N);
    offsets.push_back(N * i);
  }
}

void create_request() {
  ucc_coll_args_t coll;

  coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
  coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;

  coll.src.info_v.buffer = input;
  coll.src.info_v.counts = lengths.data();
  coll.src.info_v.displacements = offsets.data();
  coll.src.info_v.datatype = dtype;
  coll.src.info_v.mem_type = UCC_MEMORY_TYPE_CUDA;

  coll.dst.info_v.buffer = output;
  coll.dst.info_v.counts = lengths.data();
  coll.dst.info_v.displacements = offsets.data();
  coll.dst.info_v.datatype = dtype;
  coll.dst.info_v.mem_type = UCC_MEMORY_TYPE_CUDA;
  coll.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
               UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;

  ucc_status_t st = ucc_collective_init(&coll, &request, team);
  check(st == UCC_OK,
        std::string("failed to init collective: ") + ucc_status_string(st));
  std::cout << rank_string() << "[UCC] Request created." << std::endl;
}

void triggered_post() {
  cuda_ev = std::make_shared<cudaEvent_t>();
  check_cuda(cudaEventCreate(cuda_ev.get()));
  auto current_stream = getCurrentCUDAStream();
  check_cuda(cudaEventRecord(*cuda_ev, current_stream));
  check_cuda(cudaStreamWaitEvent(*stream, *cuda_ev));
  std::cout << rank_string() << "[UCC] Sync stream." << std::endl;

  ucc_ev_t comp_ev;
  comp_ev.ev_type = UCC_EVENT_COMPUTE_COMPLETE;
  comp_ev.ev_context = nullptr;
  comp_ev.ev_context_size = 0;
  comp_ev.req = request;
  ucc_status_t st = ucc_collective_triggered_post(cuda_ee, &comp_ev);
  check(st == UCC_OK, std::string("failed to post triggered collective: ") + ucc_status_string(st));

  ucc_ev_t *post_ev;
  st = ucc_ee_get_event(cuda_ee, &post_ev);
  check(st == UCC_OK && post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST, "Bug???");
  ucc_ee_ack_event(cuda_ee, post_ev);
  check(st == UCC_OK, "Bug???");
  std::cout << rank_string() << "[UCC] triggered_post succeed." << std::endl;

  check_cuda(cudaEventRecord(*cuda_ev, *stream));
  while (request->status > 0) {
    ucc_context_progress(context);
  }
  ucc_collective_finalize(request);
  std::cout << rank_string() << "[UCC] request finalized." << std::endl;
}

} // namespace ucc

void alltoall() {
  ucc::create_context();
  ucc::create_team();

  ucc::create_cuda_ee();
  ucc::compute_lengths_and_offsets();
  ucc::create_request();
  ucc::triggered_post();
}