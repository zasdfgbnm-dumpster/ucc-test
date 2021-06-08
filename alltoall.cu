#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include <ucc/api/ucc.h>
#include <ucp/api/ucp.h>

using T = int;
ucc_datatype_t dtype = UCC_DT_FLOAT32;

extern const int N;
extern int world_size;
extern int rank;
const int size_ = 5;  // TODO: what is this?

void check(bool, std::string);
int get_device();

constexpr auto kUnsetTimeout = std::chrono::milliseconds(-1);

enum OpType { ALLTOALL_BASE };

class Store {};

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

class CommUCC : public CommBase {
public:
  ucc_lib_h lib;
  ucc_context_h context;

public:
  void progress();
  CommUCC(torch_ucc_oob_coll_info_t *oob_info);
  ~CommUCC();
};

#define TORCH_UCC_DEVICE_NOT_SET -2

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

enum torch_ucx_tag_type_t { TORCH_UCX_P2P_TAG, TORCH_UCX_OOB_TAG };

struct event_pool_t {
  std::queue<std::unique_ptr<cudaEvent_t>> event_pool;
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

torch_ucc_oob_coll_info_t oob;
std::shared_ptr<CommPG> comm;
uint32_t comm_id;
std::vector<ucp_ep_h> eps;
ucc_team_h team;
ucc_ee_h cuda_ee;
std::shared_ptr<cudaStream_t> stream =
    nullptr; // TODO, it was unique_ptr in its original code
event_pool_t ep;

void initComm(int dev) {
  if (!comm) {
    comm = CommPG::get_comm(comm_id, dev, &oob);
    comm->ucx_connect_eps(eps, &oob);
    comm->ucc_create_team(team, &oob);
  } else {
    if ((comm->cuda_device_index != TORCH_UCC_DEVICE_NOT_SET) &&
        (comm->cuda_device_index != dev)) {
      check(false,
            "ucc communicator was initialized with different cuda device, "
            "multi device is not supported");
      throw std::runtime_error(ucc_status_string(UCC_ERR_NOT_SUPPORTED));
    }
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

void alltoall() {
  std::vector<int64_t> outputSplitSizes;
  std::vector<int64_t> inputSplitSizes;

  initComm(get_device());
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
    // computeLengthsAndOffsets(inputSplitSizes, inputTensor, &data->send_lengths,
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