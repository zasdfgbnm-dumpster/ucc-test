#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

#include "utils.hpp"

using T = int;
ucc_datatype_t dtype = UCC_DT_FLOAT32;

extern const int N;
extern int world_size;
extern int rank;
const int size_ = 5; // TODO: what is this?

void check_cuda(cudaError_t);
int get_device();
void set_device(int i);

constexpr auto kUnsetTimeout = std::chrono::milliseconds(-1);

enum OpType { ALLTOALL_BASE };

bool isP2POp(OpType) { return true; }

#define TORCH_UCC_DEVICE_NOT_SET -2

#define TORCH_UCX_COMM_BITS 15
#define TORCH_UCX_RANK_BITS 16
#define TORCH_UCX_TAG_BITS 32
#define TORCH_UCX_OOB_BITS 1

#define TORCH_UCX_COMM_BITS_OFFSET 0
#define TORCH_UCX_RANK_BITS_OFFSET TORCH_UCX_COMM_BITS
#define TORCH_UCX_TAG_BITS_OFFSET (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS)
#define TORCH_UCX_OOB_BITS_OFFSET                                              \
  (TORCH_UCX_COMM_BITS + TORCH_UCX_RANK_BITS + TORCH_UCX_TAG_BITS)

#define TORCH_UCX_MAX_COMM ((((uint64_t)1) << TORCH_UCX_COMM_BITS) - 1)
#define TORCH_UCX_MAX_RANK ((((uint64_t)1) << TORCH_UCX_RANK_BITS) - 1)
#define TORCH_UCX_MAX_TAG ((((uint64_t)1) << TORCH_UCX_TAG_BITS) - 1)
#define TORCH_UCX_MAX_OOB ((((uint64_t)1) << TORCH_UCX_OOB_BITS) - 1)

#define TORCH_UCX_COMM_MASK (TORCH_UCX_MAX_COMM << TORCH_UCX_COMM_BITS_OFFSET)
#define TORCH_UCX_RANK_MASK (TORCH_UCX_MAX_RANK << TORCH_UCX_RANK_BITS_OFFSET)
#define TORCH_UCX_TAG_MASK (TORCH_UCX_MAX_TAG << TORCH_UCX_TAG_BITS_OFFSET)
#define TORCH_UCX_OOB_MASK (TORCH_UCX_MAX_OOB << TORCH_UCX_OOB_BITS_OFFSET)

#define TORCH_UCX_MAKE_P2P_TAG(_tag, _rank, _comm)                             \
  ((((uint64_t)(_tag)) << TORCH_UCX_TAG_BITS_OFFSET) |                         \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) |                       \
   (((uint64_t)(_comm)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_OOB_TAG(_tag, _rank, _comm)                             \
  ((((uint64_t)(_tag)) << TORCH_UCX_OOB_BITS_OFFSET) |                         \
   (((uint64_t)(_rank)) << TORCH_UCX_RANK_BITS_OFFSET) |                       \
   (((uint64_t)(_rank)) << TORCH_UCX_COMM_BITS_OFFSET))

#define TORCH_UCX_MAKE_SEND_TAG(_ucp_tag, _tag, _rank, _comm)                  \
  do {                                                                         \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm));             \
  } while (0)

#define TORCH_UCX_ANY_SOURCE (TORCH_UCX_MAX_RANK - 1)
#define TORCH_UCX_ANY_SOURCE_MASK (~TORCH_UCX_RANK_MASK)
#define TORCH_UCX_SPECIFIC_SOURCE_MASK ((uint64_t)-1)

#define TORCH_UCX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank, _comm)   \
  do {                                                                         \
    (_ucp_tag) = TORCH_UCX_MAKE_P2P_TAG((_tag), (_rank), (_comm));             \
    if ((_rank) == TORCH_UCX_ANY_SOURCE) {                                     \
      (_ucp_tag_mask) = TORCH_UCX_ANY_SOURCE_MASK;                             \
    } else {                                                                   \
      (_ucp_tag_mask) = TORCH_UCX_SPECIFIC_SOURCE_MASK;                        \
    }                                                                          \
  } while (0)

#define TORCH_UCX_MAKE_OOB_SEND_TAG(_ucp_tag, _tag, _rank, _comm)              \
  do {                                                                         \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm));             \
  } while (0)

#define TORCH_UCX_MAKE_OOB_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _rank,      \
                                    _comm)                                     \
  do {                                                                         \
    (_ucp_tag) = TORCH_UCX_MAKE_OOB_TAG((_tag), (_rank), (_comm));             \
    (_ucp_tag_mask) = (uint64_t)-1;                                            \
  } while (0)

class WorkData {
public:
  // TODO enable this
  // std::vector<at::Tensor> src;
  // std::vector<at::Tensor> dst;
  WorkData() {}
  virtual ~WorkData() = default;
};

class AlltoallWorkData : public WorkData {
public:
  AlltoallWorkData(int size)
      : send_lengths(size), send_offsets(size), recv_lengths(size),
        recv_offsets(size) {}
  std::vector<uint32_t> send_lengths;
  std::vector<uint32_t> send_offsets;
  std::vector<uint32_t> recv_lengths;
  std::vector<uint32_t> recv_offsets;
};

cudaStream_t getStreamFromPool(int dev) {
  // TODO
  return 0;
}

cudaStream_t getCurrentCUDAStream(int dev) {
  // TODO
  return 0;
}

struct torch_ucc_oob_coll_info_t {
  std::shared_ptr<Store> store;
  uint32_t comm_id;
  int rank;
  int size;
  void *rbuf;
  size_t msglen;
  std::string getKey(std::string key) { return std::to_string(comm_id) + key; }
};

class CommBase {
public:
  CommBase() {}
  virtual void progress() = 0;
  virtual ~CommBase() {}
};

class CommUCX : public CommBase {
public:
  ucp_context_h context;
  ucp_worker_h worker;

public:
  void progress();
  CommUCX(int comm_size);
  ~CommUCX();
};

CommUCX::CommUCX(int comm_size) {
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

void CommUCX::progress() { ucp_worker_progress(worker); }

CommUCX::~CommUCX() {
  ucp_worker_destroy(worker);
  ucp_cleanup(context);
}

class CommUCC : public CommBase {
public:
  ucc_lib_h lib;
  ucc_context_h context;

public:
  void progress();
  CommUCC(torch_ucc_oob_coll_info_t *oob_info);
  ~CommUCC();
};

ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                           void *coll_info, void **req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t *>(coll_info);
  std::vector<char> val =
      std::vector<char>(reinterpret_cast<char *>(sbuf),
                           reinterpret_cast<char *>(sbuf) + msglen);
  info->store->set(info->getKey("teamr" + std::to_string(info->rank)), val);
  info->rbuf = rbuf;
  info->msglen = msglen;
  *req = coll_info;
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void *req) {
  torch_ucc_oob_coll_info_t *info =
      reinterpret_cast<torch_ucc_oob_coll_info_t *>(req);

  for (int r = 0; r < info->size; r++) {
    if (!info->store->check({info->getKey("teamr" + std::to_string(r))})) {
      return UCC_INPROGRESS;
    }
  }
  for (int r = 0; r < info->size; r++) {
    std::vector<char> data =
        info->store->get(info->getKey("teamr" + std::to_string(r)));
    memcpy((void *)((ptrdiff_t)info->rbuf + info->msglen * r), data.data(),
           info->msglen);
  }
  return UCC_OK;
}

ucc_status_t oob_allgather_free(void *req) {
  // torch_ucc_oob_coll_info_t *info =
  //     reinterpret_cast<torch_ucc_oob_coll_info_t *>(req);
  // int num_done = info->store->add({info->getKey("ag_done")}, 1);
  // if (num_done == info->size) {
  //   info->store->deleteKey(info->getKey("ag_done"));
  //   for (int r = 0; r < info->size; r++) {
  //     info->store->deleteKey(info->getKey("teamr" + std::to_string(r)));
  //   }
  //   for (int r = 0; r < info->size; r++) {
  //     info->store->add({info->getKey("ag_free" + std::to_string(r))}, 1);
  //   }
  // } else {
  //   info->store->wait({info->getKey("ag_free" +
  //   std::to_string(info->rank))});
  // }
  // info->store->deleteKey(info->getKey("ag_free" +
  // std::to_string(info->rank)));
  return UCC_OK;
}

CommUCC::CommUCC(torch_ucc_oob_coll_info_t *oob_info) {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

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
                                 std::to_string(oob_info->size).c_str());
  check(st == UCC_OK, std::string("failed to modify UCC context config: ") +
                          ucc_status_string(st));
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.participants = oob_info->size;
  context_params.oob.allgather = oob_allgather;
  context_params.oob.req_test = oob_allgather_test;
  context_params.oob.req_free = oob_allgather_free;
  context_params.oob.coll_info = oob_info;
  ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  check(st == UCC_OK,
        std::string("failed to create UCC context: ") + ucc_status_string(st));
}

void CommUCC::progress() { ucc_context_progress(context); }

CommUCC::~CommUCC() {
  ucc_context_destroy(context);
  ucc_finalize(lib);
}

enum torch_ucx_tag_type_t { TORCH_UCX_P2P_TAG, TORCH_UCX_OOB_TAG };

struct event_pool_t {
  std::queue<std::unique_ptr<cudaEvent_t>> event_pool;
  std::mutex event_pool_mutex;
};

class WorkUCC {
  OpType opType;
  friend class CommPG;

public:
  WorkUCC(OpType opType, ucc_status_t status, ucc_coll_req_h request,
          ucc_ee_h ee, CommBase *comm)
      : opType(opType), status_(status), request_(request), comm_(comm) {}
  ~WorkUCC();
  bool isCompleted();
  bool isSuccess() const;
  bool wait(std::chrono::milliseconds timeout = kUnsetTimeout);
  void finalize();
  std::unique_ptr<WorkData> data;
  std::unique_ptr<cudaEvent_t> fence = nullptr;
  event_pool_t *ep = nullptr;

protected:
  ucc_status_t status_;
  ucc_coll_req_h request_;
  CommBase *comm_;
};

WorkUCC::~WorkUCC() {
  check(request_ == nullptr, "TorchUCC, request wasn't finalized");
  if (fence && ep) {
    std::lock_guard<std::mutex> lock(ep->event_pool_mutex);
    ep->event_pool.push(std::move(fence));
  }
}

void WorkUCC::finalize() {
  if (request_ != nullptr) {
    if (isP2POp(opType)) {
      request_->status = UCC_INPROGRESS;
      ucp_request_free(request_);
    } else {
      ucc_collective_finalize(request_);
    }
    status_ = UCC_OK;
    request_ = nullptr;
  }
}

class CommPG {
  CommUCX ucx_comm;
  CommUCC ucc_comm;
  int device_index;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<std::shared_ptr<WorkUCC>> progress_queue;
  bool stop_progress_loop;

public:
  int cuda_device_index;
  CommPG(torch_ucc_oob_coll_info_t *oob_info, int dev);

  ~CommPG();

  void ucx_connect_eps(std::vector<ucp_ep_h> &eps,
                       torch_ucc_oob_coll_info_t *oob);

  void ucx_disconnect_eps(std::vector<ucp_ep_h> &eps,
                          torch_ucc_oob_coll_info_t *oob);

  void ucc_create_team(ucc_team_h &team, torch_ucc_oob_coll_info_t *oob_info);

  void ucc_destroy_team(ucc_team_h &team);

  std::shared_ptr<WorkUCC> enqueue_p2p(OpType opType, ucc_coll_req_h request);

  std::shared_ptr<WorkUCC>
  enqueue_cuda_collective(OpType opType, ucc_coll_args_t &coll,
                          std::unique_ptr<WorkData> data, ucc_team_h &team,
                          ucc_ee_h ee, std::unique_ptr<cudaEvent_t> cuda_ev,
                          const cudaStream_t &stream, event_pool_t *ep);

  std::shared_ptr<WorkUCC> enqueue_collective(OpType opType,
                                              ucc_coll_args_t &coll,
                                              std::unique_ptr<WorkData> data,
                                              ucc_team_h &team);

  static std::shared_ptr<CommPG> get_comm(uint32_t &id, int dev,
                                          torch_ucc_oob_coll_info_t *oob);

  void progress_loop();

  ucc_coll_req_h send_nb(ucp_ep_h ep, void *data, ucs_memory_type_t mtype,
                         size_t size, ucp_tag_t ucp_tag);

  ucc_coll_req_h recv_nb(void *data, ucs_memory_type_t mtype, size_t size,
                         ucp_tag_t ucp_tag, ucp_tag_t ucp_tag_mask);
};

CommPG::CommPG(torch_ucc_oob_coll_info_t *oob_info, int dev)
    : ucx_comm(oob_info->size), ucc_comm(oob_info), cuda_device_index(dev) {
  stop_progress_loop = false;
  progress_thread = std::thread(&CommPG::progress_loop, this);
  pthread_setname_np(progress_thread.native_handle(), "ucc-progress");
}

CommPG::~CommPG() {
  std::unique_lock<std::mutex> lock(mutex);
  queue_consume_cv.wait(lock, [&] { return progress_queue.empty(); });
  stop_progress_loop = true;
  lock.unlock();
  queue_produce_cv.notify_all();
  progress_thread.join();
}

std::shared_ptr<CommPG> CommPG::get_comm(uint32_t &id, int dev,
                                         torch_ucc_oob_coll_info_t *oob) {
  static std::mutex m;
  static std::weak_ptr<CommPG> comm;
  static uint32_t comm_id;

  std::lock_guard<std::mutex> lock(m);
  id = (comm_id++ % TORCH_UCX_COMM_BITS);
  oob->comm_id = id;
  std::shared_ptr<CommPG> shared_comm = comm.lock();
  if (!shared_comm) {
    shared_comm = std::make_shared<CommPG>(oob, dev);
    comm = shared_comm;
  } else {
    check((shared_comm->cuda_device_index == TORCH_UCC_DEVICE_NOT_SET) ||
              (shared_comm->cuda_device_index == dev),
          "ucc communicator was initialized with different cuda device,"
          "multi device is not supported");
    shared_comm->cuda_device_index = dev;
  }
  return shared_comm;
}

void CommPG::ucx_connect_eps(std::vector<ucp_ep_h> &eps,
                             torch_ucc_oob_coll_info_t *oob) {
  ucs_status_t st;
  ucp_address_t *local_addr;
  size_t local_addr_len;
  std::vector<char> peer_addr;

  st = ucp_worker_get_address(ucx_comm.worker, &local_addr, &local_addr_len);
  check(st == UCS_OK, "failed to get worker address");
  std::vector<char> val =
      std::vector<char>(reinterpret_cast<char *>(local_addr),
                        reinterpret_cast<char *>(local_addr) + local_addr_len);
  oob->store->set(oob->getKey("wa" + std::to_string(oob->rank)), val);
  ucp_worker_release_address(ucx_comm.worker, local_addr);
  eps.resize(oob->size);
  for (int i = 0; i < oob->size; i++) {
    peer_addr = oob->store->get(oob->getKey("wa" + std::to_string(i)));
    ucp_ep_params_t ep_params;
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address = reinterpret_cast<ucp_address_t *>(peer_addr.data());
    st = ucp_ep_create(ucx_comm.worker, &ep_params, &(eps[i]));
    check(st == UCS_OK, "failed to create endpoint");
  }
}

// TODO: remove this?
// void CommPG::ucx_disconnect_eps(std::vector<ucp_ep_h> &eps,
//                                 torch_ucc_oob_coll_info_t *oob) {
//   ucs_status_t st;

//   for (ucp_ep_h &ep : eps) {
//     ucs_status_ptr_t close_req = ucp_ep_close_nb(ep,
//     UCP_EP_CLOSE_MODE_FLUSH); check(!UCS_PTR_IS_ERR(close_req), "failed to
//     close endpoint"); if (UCS_PTR_IS_PTR(close_req)) {
//       do {
//         ucp_worker_progress(ucx_comm.worker);
//         st = ucp_request_check_status(close_req);
//       } while (st != UCS_OK);
//       ucp_request_free(close_req);
//     }
//   }
//   if ((size_t)oob->store->add(oob->getKey("epclosed"), 1) == eps.size()) {
//     oob->store->add(oob->getKey("epfinished"), 1);
//   } else {
//     oob->store->wait({oob->getKey("epfinished")});
//   }
// }

// TODO: can I delete this?
// ucc_coll_req_h CommPG::send_nb(ucp_ep_h ep, void *data, ucs_memory_type_t
// mtype,
//                                size_t size, ucp_tag_t ucp_tag) {
//   ucs_status_ptr_t st;
//   ucp_request_param_t params;
//   params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
//                         UCP_OP_ATTR_FIELD_DATATYPE |
//                         UCP_OP_ATTR_FIELD_MEMORY_TYPE;
//   params.datatype = ucp_dt_make_contig(size);
//   params.memory_type = mtype;
//   params.cb.send = [](void *request, ucs_status_t status, void *user_data) {
//     static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
//   };
//   st = ucp_tag_send_nbx(ep, data, 1, ucp_tag, &params);
//   check(!UCS_PTR_IS_ERR(st), std::string("failed to send message: ") +
//                                  ucs_status_string(UCS_PTR_STATUS(st)));
//   return reinterpret_cast<ucc_coll_req_h>(st);
// }

// TODO: can I delete this?
// ucc_coll_req_h CommPG::recv_nb(void *data, ucs_memory_type_t mtype, size_t
// size,
//                                ucp_tag_t ucp_tag, ucp_tag_t ucp_tag_mask) {
//   ucs_status_ptr_t st;
//   ucp_request_param_t params;
//   params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
//                         UCP_OP_ATTR_FIELD_DATATYPE |
//                         UCP_OP_ATTR_FIELD_MEMORY_TYPE;
//   params.datatype = ucp_dt_make_contig(size);
//   params.cb.recv = [](void *request, ucs_status_t status,
//                       const ucp_tag_recv_info_t *info, void *user_data) {
//     static_cast<ucc_coll_req_h>(request)->status = UCC_OK;
//   };
//   params.memory_type = mtype;
//   st = ucp_tag_recv_nbx(ucx_comm.worker, data, 1, ucp_tag, ucp_tag_mask,
//                         &params);
//   check(!UCS_PTR_IS_ERR(st), std::string("failed to recv message: ") +
//                                  ucs_status_string(UCS_PTR_STATUS(st)));
//   return reinterpret_cast<ucc_coll_req_h>(st);
// }

void CommPG::ucc_create_team(ucc_team_h &team,
                             torch_ucc_oob_coll_info_t *oob_info) {
  ucc_status_t st;
  ucc_team_params_t team_params;
  team_params.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
                     UCC_TEAM_PARAM_FIELD_OOB;
  team_params.oob.allgather = oob_allgather;
  team_params.oob.req_test = oob_allgather_test;
  team_params.oob.req_free = oob_allgather_free;
  team_params.oob.coll_info = oob_info;
  team_params.oob.participants = oob_info->size;
  team_params.ep = oob_info->rank;
  team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
  st = ucc_team_create_post(&ucc_comm.context, 1, &team_params, &team);
  std::cout << "ucc_team_create_post" << std::endl;
  check(st == UCC_OK,
        std::string("failed to post team create: ") + ucc_status_string(st));
  do {
    st = ucc_team_create_test(team);
  } while (st == UCC_INPROGRESS);
  check(st == UCC_OK,
        std::string("failed to create UCC team: ") + ucc_status_string(st));
  std::cout << "ucc_create_team" << std::endl;
}

void CommPG::ucc_destroy_team(ucc_team_h &team) {
  ucc_status_t status;
  while (UCC_INPROGRESS == (status = ucc_team_destroy(team))) {
    check(status == UCC_OK,
          std::string("ucc team destroy error: ") + ucc_status_string(status));
  }
}

std::shared_ptr<WorkUCC> CommPG::enqueue_p2p(OpType opType,
                                             ucc_coll_req_h request) {
  if (request == nullptr) {
    // p2p2 request completed immediately don't save it to progress queue
    return std::make_shared<WorkUCC>(opType, UCC_OK, request, nullptr,
                                     &ucx_comm);
  }
  auto work = std::make_shared<WorkUCC>(opType, UCC_INPROGRESS, request,
                                        nullptr, &ucx_comm);
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(work);
  lock.unlock();
  queue_produce_cv.notify_one();
  return work;
}

std::shared_ptr<WorkUCC>
CommPG::enqueue_collective(OpType opType, ucc_coll_args_t &coll,
                           std::unique_ptr<WorkData> data, ucc_team_h &team) {
  ucc_coll_req_h request;
  ucc_status_t st;
  st = ucc_collective_init(&coll, &request, team);
  check(st == UCC_OK,
        std::string("failed to init collective: ") + ucc_status_string(st));
  st = ucc_collective_post(request);
  check(st == UCC_OK,
        std::string("failed to post collective: ") + ucc_status_string(st));
  auto work = std::make_shared<WorkUCC>(opType, UCC_INPROGRESS, request,
                                        nullptr, &ucc_comm);
  work->data = std::move(data);
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(work);
  lock.unlock();
  queue_produce_cv.notify_one();
  return work;
}

std::shared_ptr<WorkUCC> CommPG::enqueue_cuda_collective(
    OpType opType, ucc_coll_args_t &coll, std::unique_ptr<WorkData> data,
    ucc_team_h &team, ucc_ee_h ee, std::unique_ptr<cudaEvent_t> cuda_ev,
    const cudaStream_t &stream, event_pool_t *ep) {
  ucc_coll_req_h request;
  ucc_status_t st;
  st = ucc_collective_init(&coll, &request, team);
  check(st == UCC_OK,
        std::string("failed to init collective: ") + ucc_status_string(st));
  ucc_ev_t comp_ev, *post_ev;
  comp_ev.ev_type = UCC_EVENT_COMPUTE_COMPLETE;
  comp_ev.ev_context = nullptr;
  comp_ev.ev_context_size = 0;
  comp_ev.req = request;
  st = ucc_collective_triggered_post(ee, &comp_ev);
  check(st == UCC_OK, std::string("failed to post triggered collective: ") +
                          ucc_status_string(st));
  st = ucc_ee_get_event(ee, &post_ev);
  check(st == UCC_OK && post_ev->ev_type == UCC_EVENT_COLLECTIVE_POST,
        "Bug???");
  ucc_ee_ack_event(ee, post_ev);
  auto work =
      std::make_shared<WorkUCC>(opType, UCC_INPROGRESS, request, ee, &ucc_comm);
  work->data = std::move(data);
  work->ep = ep;
  check_cuda(cudaEventRecord(*cuda_ev, stream));
  work->fence = std::move(cuda_ev);
  std::unique_lock<std::mutex> lock(mutex);
  progress_queue.push_back(work);
  lock.unlock();
  queue_produce_cv.notify_one();
  return work;
}

void CommPG::progress_loop() {
  std::unique_lock<std::mutex> lock(mutex);
  bool device_set = false;
  while (!stop_progress_loop) {
    if (progress_queue.empty()) {
      queue_produce_cv.wait(lock);
      continue;
    }
    auto work = progress_queue.front();
    progress_queue.pop_front();
    lock.unlock();
    queue_consume_cv.notify_one();
    if ((!device_set) && (cuda_device_index != TORCH_UCC_DEVICE_NOT_SET)) {
      set_device(cuda_device_index);
      device_set = true;
    }
    while (work->request_->status > 0) {
      // operation initialized is in progress or
      work->comm_->progress();
    }
    work->finalize();
    work->data.reset();
    lock.lock();
  }
}

torch_ucc_oob_coll_info_t oob;
std::shared_ptr<CommPG> comm;
uint32_t comm_id;
std::vector<ucp_ep_h> eps;
ucc_team_h team;
ucc_ee_h cuda_ee;
std::shared_ptr<cudaStream_t> stream =
    nullptr; // TODO, it was unique_ptr in its original code
event_pool_t ep;

void initProcessGroupUCC(const std::shared_ptr<Store> &store, int rank,
                         int size) {
  // TODO: should size be world size?
  oob.rank = rank;
  oob.size = size;
  oob.store = store;
  comm = nullptr;
  cuda_ee = nullptr;
}

void initComm(int dev) {
  if (!comm) {
    set_device(dev);
    comm = CommPG::get_comm(comm_id, dev, &oob);
    comm->ucx_connect_eps(eps, &oob);
    comm->ucc_create_team(team, &oob);
  } else {
    check((comm->cuda_device_index == TORCH_UCC_DEVICE_NOT_SET) ||
              (comm->cuda_device_index == dev),
          "ucc communicator was initialized with different cuda device, "
          "multi device is not supported");
    comm->cuda_device_index = dev;
  }
  if (!cuda_ee) {
    ucc_status_t st;
    stream = std::make_shared<cudaStream_t>(getStreamFromPool(dev));
    ucc_ee_params_t params;
    params.ee_type = UCC_EE_CUDA_STREAM;
    params.ee_context = (void *)stream.get();
    params.ee_context_size = sizeof(cudaStream_t);
    st = ucc_ee_create(team, &params, &cuda_ee);
    check(st == UCC_OK,
          std::string("failed to create UCC EE: ") + ucc_status_string(st));
  }
}

std::shared_ptr<WorkUCC> collective_post(OpType opType, ucc_coll_args_t &coll,
                                         std::unique_ptr<WorkData> data,
                                         int dev) {
  std::unique_ptr<cudaEvent_t> cuda_ev;
  {
    std::lock_guard<std::mutex> lock(ep.event_pool_mutex);
    if (ep.event_pool.empty()) {
      cuda_ev = std::make_unique<cudaEvent_t>();
    } else {
      cuda_ev = std::move(ep.event_pool.front());
      ep.event_pool.pop();
    }
  }
  auto current_stream = getCurrentCUDAStream(dev);
  check_cuda(cudaEventRecord(*cuda_ev, current_stream));
  check_cuda(cudaStreamWaitEvent(*stream, *cuda_ev));
  auto work =
      comm->enqueue_cuda_collective(opType, coll, std::move(data), team,
                                    cuda_ee, std::move(cuda_ev), *stream, &ep);
  return work;
}

std::shared_ptr<WorkUCC> alltoall() {
  initProcessGroupUCC({}, rank, world_size); // TODO use a real store
  initComm(get_device());

  // TODO initialize them
  std::vector<int64_t> outputSplitSizes;
  std::vector<int64_t> inputSplitSizes;
  ucc_coll_args_t coll;
  AlltoallWorkData *data;

  if ((outputSplitSizes.size() == 0) && (inputSplitSizes.size() == 0)) {
    data = new AlltoallWorkData(0);
    // TODO: migrate this
    // TORCH_CHECK((outputTensor.size(0) % size_ == 0) &&
    //                 (inputTensor.size(0) % size_ == 0),
    //             "Tensor's dim 0 does not divide equally across group size");
    coll.mask = 0;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALL;
    // TODO: enable this
    // coll.src.info.buffer = inputTensor.data_ptr();
    // coll.src.info.count =
    //     inputTensor.element_size() * inputTensor.numel() / size_;
    coll.src.info.datatype = UCC_DT_UINT8;
    coll.src.info.mem_type = UCC_MEMORY_TYPE_CUDA;
    // TODO: enable this
    // coll.dst.info.buffer = outputTensor.data_ptr();
    // coll.dst.info.count =
    //     outputTensor.element_size() * outputTensor.numel() / size_;
    coll.dst.info.datatype = UCC_DT_UINT8;
    coll.dst.info.mem_type = UCC_MEMORY_TYPE_CUDA;
  } else {
    data = new AlltoallWorkData(size_);
    // TODO: migrate this
    // c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    // c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    // computeLengthsAndOffsets(outputSplitSizes, outputTensor,
    //                          &data->recv_lengths, &data->recv_offsets);
    // computeLengthsAndOffsets(inputSplitSizes, inputTensor,
    // &data->send_lengths,
    //                          &data->send_offsets);
    coll.mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll.coll_type = UCC_COLL_TYPE_ALLTOALLV;
    // TODO: enable this
    // coll.src.info_v.buffer = inputTensor.data_ptr();
    coll.src.info_v.counts = (ucc_count_t *)data->send_lengths.data();
    coll.src.info_v.displacements = (ucc_aint_t *)data->send_offsets.data();
    coll.src.info_v.datatype = dtype;
    coll.src.info_v.mem_type = UCC_MEMORY_TYPE_CUDA;
    // TODO: enable this
    // coll.dst.info_v.buffer = outputTensor.data_ptr();
    coll.dst.info_v.counts = (ucc_count_t *)data->recv_lengths.data();
    coll.dst.info_v.displacements = (ucc_aint_t *)data->recv_offsets.data();
    coll.dst.info_v.datatype = dtype;
    coll.dst.info_v.mem_type = UCC_MEMORY_TYPE_CUDA;
    coll.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                 UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
  }
  // TODO: enable this
  // data->src = {inputTensor};
  // data->dst = {outputTensor};
  return collective_post(OpType::ALLTOALL_BASE, coll,
                         std::unique_ptr<WorkData>(data), get_device());
}