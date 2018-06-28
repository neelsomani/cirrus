#include <cassert>
#include <stdexcept>
#include "PSSparseServerInterface.h"
#include "Constants.h"
#include "MFModel.h"
#include "Checksum.h"
#include "Constants.h"

#ifdef USE_EFS
#include "NFSFile.h"
#include "NFSls.h"
#endif

//#define DEBUG

#define MAX_MSG_SIZE (1024*1024)

namespace cirrus {

PSSparseServerInterface::PSSparseServerInterface(const std::string& ip, int port) :
  ip(ip), port(port) {

  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    throw std::runtime_error("Error when creating socket.");
  }
  int opt = 1;
  if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt))) {
    throw std::runtime_error("Error setting socket options.");
  }

  struct sockaddr_in serv_addr;
  serv_addr.sin_family = AF_INET;
  if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) != 1) {
    throw std::runtime_error("Address family invalid or invalid "
        "IP address passed in");
  }
  // Save the port in the info
  serv_addr.sin_port = htons(port);
  std::memset(serv_addr.sin_zero, 0, sizeof(serv_addr.sin_zero));

  // Connect to the server
  if (::connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    throw std::runtime_error(
        "Client could not connect to server."
        " Address: " + ip + " port: " + std::to_string(port));
  }
}

PSSparseServerInterface::~PSSparseServerInterface() {
  if (sock != -1) {
    close(sock);
  }
}

#ifdef USE_EFS
/**
  * Sending gradient means concatenating the gradient to a file in efs
  */
void PSSparseServerInterface::send_lr_gradient(const LRSparseGradient& gradient) {

  uint32_t worker_id = 1337;
  static int num_grads = 0; // number of gradients already concatenated
  static uint32_t cum_size_grads = 0; // cumulative size of gradients written to file
  std::string filename = "gradient." + std::to_string(worker_id);
  NFSFile file("/efs_experiment/" + filename); //open nfs file

  uint32_t size = gradient.getSerializedSize();
  if (size > (1024 * 1024)) {
    throw std::runtime_error("Too large to store in the stack"); // just be cautious
  }

  // we write <size of serialized gradient, serialized. gradient>
  char data[size + sizeof(uint32_t)];
  char* data_ptr = data;

  // first we write gradient size and then we write the gradient
  store_value<uint32_t>(data_ptr, size);
  gradient.serialize(data_ptr);

  cum_size_grads += size;
  // write this gradient to the file with the right offset
  file.write(cum_size_grads, data, size + sizeof(uint32_t));
}

void PSSparseServerInterface::get_lr_sparse_model_inplace(
    const SparseDataset& ds, SparseLRModel& lr_model,
    const Configuration& config) {
  // here we do an ls of the directory
  // and get the latest model

  NFSls ls("/efs_experiment/");
  std::vector<std::pair<std::string, uint64_t>> result = ls.do_ls();

  uint32_t highest = 0;
  for (const auto& entry : result) {
    std::string path = entry.first;
    std::cout << "file path: " << path << std::endl;
    // if filename starts with "model"
    if (path.compare(0, std::string("model").size(), "model") == 0) {
      std::string num = path.substr(5);
      uint32_t number = string_to<uint32_t>(num);
      highest = std::max(number, highest);
    }
  }

  std::cout << "Highest model seen: " << highest << std::endl;
  // now we get the last model

  std::string filename = "/efs_experiment/model" + std::to_string(highest);
  NFSFile model_file(filename);

  uint32_t model_size = 0;
  if (model_size > 10 * 1024 * 1024) {
    throw std::runtime_error("something is wrong");
  }
  char* model_data = new char[model_size]; // 4MB should be enough
  // first we read the model size
  model_file.read(0, (char*)&model_size, sizeof(model_size));
  model_file.read(sizeof(model_size), model_data, model_size);

  std::cout << "Read model" << std::endl;
  
}

std::unique_ptr<CirrusModel> PSSparseServerInterface::get_full_model(bool isCollaborative) {
  throw "not supported";
  std::unique_ptr<CirrusModel> model = std::make_unique<SparseLRModel>(0);
  return model;
}

/**
 * USE_EFS
 */
#else 
void PSSparseServerInterface::send_lr_gradient(const LRSparseGradient& gradient) {
  uint32_t operation = SEND_LR_GRADIENT;
#ifdef DEBUG
  std::cout << "Sending gradient" << std::endl;
#endif
  int ret = send(sock, &operation, sizeof(uint32_t), 0);
  if (ret == -1) {
    throw std::runtime_error("Error sending operation");
  }

  uint32_t size = gradient.getSerializedSize();
#ifdef DEBUG
  std::cout << "Sending gradient with size: " << size << std::endl;
#endif
  ret = send(sock, &size, sizeof(uint32_t), 0);
  if (ret == -1) {
    throw std::runtime_error("Error sending grad size");
  }
  
  char data[size];
  gradient.serialize(data);
  ret = send_all(sock, data, size);
  if (ret == 0) {
    throw std::runtime_error("Error sending grad");
  }
}

void PSSparseServerInterface::get_lr_sparse_model_inplace(const SparseDataset& ds, SparseLRModel& lr_model,
    const Configuration& config) {
#ifdef DEBUG
  std::cout << "Getting LR sparse model inplace" << std::endl;
#endif
  // we don't know the number of weights to start with
  char* msg = new char[MAX_MSG_SIZE];
  char* msg_begin = msg; // need to keep this pointer to delete later

  uint32_t num_weights = 0;
  store_value<uint32_t>(msg, num_weights); // just make space for the number of weights
  for (const auto& sample : ds.data_) {
    for (const auto& w : sample) {
      store_value<uint32_t>(msg, w.first); // encode the index
      num_weights++;
    }
  }
  msg = msg_begin;
  store_value<uint32_t>(msg, num_weights); // store correct value here
#ifdef DEBUG
  assert(std::distance(msg_begin, msg) < MAX_MSG_SIZE);
  std::cout << std::endl;
#endif

#ifdef DEBUG
  std::cout << "Sending operation and size" << std::endl;
#endif
  // 1. Send operation
  uint32_t operation = GET_LR_SPARSE_MODEL;
  if (send_all(sock, &operation, sizeof(uint32_t)) == -1) {
    throw std::runtime_error("Error getting sparse lr model");
  }
  // 2. Send msg size
  uint32_t msg_size = sizeof(uint32_t) + sizeof(uint32_t) * num_weights;
#ifdef DEBUG
  std::cout << "msg_size: " << msg_size
    << " num_weights: " << num_weights
    << std::endl;
#endif
  send_all(sock, &msg_size, sizeof(uint32_t));
  // 3. Send num_weights + weights
  if (send_all(sock, msg_begin, msg_size) == -1) {
    throw std::runtime_error("Error getting sparse lr model");
  }
  
  //4. receive weights from PS
  uint32_t to_receive_size = sizeof(FEATURE_TYPE) * num_weights;
  //std::cout << "Model sent. Receiving: " << num_weights << " weights" << std::endl;

#ifdef DEBUG
  std::cout << "Receiving " << to_receive_size << " bytes" << std::endl;
#endif
  char* buffer = new char[to_receive_size];
  read_all(sock, buffer, to_receive_size);

#ifdef DEBUG
  std::cout << "Loading model from memory" << std::endl;
#endif
  // build a truly sparse model and return
  // XXX this copy could be avoided
  lr_model.loadSerializedSparse((FEATURE_TYPE*)buffer, (uint32_t*)msg, num_weights, config);
  
  delete[] msg_begin;
  delete[] buffer;
}

SparseLRModel PSSparseServerInterface::get_lr_sparse_model(const SparseDataset& ds, const Configuration& config) {
  SparseLRModel model(0);
  get_lr_sparse_model_inplace(ds, model, config);
  return std::move(model);
}

std::unique_ptr<CirrusModel> PSSparseServerInterface::get_full_model(
    bool isCollaborative //XXX use a better argument here
    ) {
#ifdef DEBUG
  std::cout << "Getting full model isCollaborative: " << isCollaborative << std::endl;
#endif
  if (isCollaborative) {
    // 1. Send operation
    uint32_t operation = GET_MF_FULL_MODEL;
    send_all(sock, &operation, sizeof(uint32_t));

    uint32_t to_receive_size;
    read_all(sock, &to_receive_size, sizeof(uint32_t));
    std::cout << "Request sent. Receiving: " << to_receive_size << " bytes" << std::endl;

    char* buffer = new char[to_receive_size];
    read_all(sock, buffer, to_receive_size);
    
    std::cout
      << " buffer checksum: " << crc32(buffer, to_receive_size)
      << std::endl;

    // build a sparse model and return
    std::unique_ptr<CirrusModel> model = std::make_unique<MFModel>(
        (FEATURE_TYPE*)buffer, 0, 0, 0); //XXX fix this
    delete[] buffer;
    return model;
  } else {
    // 1. Send operation
    uint32_t operation = GET_LR_FULL_MODEL;
    send_all(sock, &operation, sizeof(uint32_t));
    //2. receive size from PS
    int model_size;
    if (read_all(sock, &model_size, sizeof(int)) == 0) {
      throw std::runtime_error("Error talking to PS");
    }
    char* model_data = new char[sizeof(int) + model_size * sizeof(FEATURE_TYPE)];
    char*model_data_ptr = model_data;
    store_value<int>(model_data_ptr, model_size);

    if (read_all(sock, model_data_ptr, model_size * sizeof(FEATURE_TYPE)) == 0) {
      throw std::runtime_error("Error talking to PS");
    }
    std::unique_ptr<CirrusModel> model = std::make_unique<SparseLRModel>(0);
    model->loadSerialized(model_data);

    delete[] model_data;
    return model;
  }
}
#endif

#ifdef USE_EFS
#else

// Collaborative filtering

/**
  * This function needs to send to the PS a list of users and items
  * FORMAT of message to send is:
  * K item ids to send (uint32_t)
  * base user id (uint32_t)
  * minibatch size (uint32_t)
  * magic number (MAGIC_NUMBER) (uint32_t)
  * list of K item ids (K * uint32_t)
  */
SparseMFModel PSSparseServerInterface::get_sparse_mf_model(
    const SparseDataset& ds, uint32_t user_base, uint32_t minibatch_size) {
  char* msg = new char[MAX_MSG_SIZE];
  char* msg_begin = msg; // need to keep this pointer to delete later
 
  uint32_t item_ids_count = 0;
  store_value<uint32_t>(msg, 0); // we will write this value later
  store_value<uint32_t>(msg, user_base);
  store_value<uint32_t>(msg, minibatch_size);
  store_value<uint32_t>(msg, MAGIC_NUMBER); // magic value
  bool seen[17770] = {false};
  for (const auto& sample : ds.data_) {
    for (const auto& w : sample) {
      uint32_t movieId = w.first;
      //std::cout << "movieId: " << movieId << "\n";

      if (seen[movieId])
          continue;
      store_value<uint32_t>(msg, movieId);
      seen[movieId] = true;
      //store_value<uint32_t>(msg, movieId); // encode the index
      item_ids_count++;
    }
  }
  msg = msg_begin;
  store_value<uint32_t>(msg, item_ids_count); // store correct value here

  // 1. Send operation
  uint32_t operation = GET_MF_SPARSE_MODEL;
  send_all(sock, &operation, sizeof(uint32_t));
  // 2. Send msg size
  uint32_t msg_size = sizeof(uint32_t) * 4 + sizeof(uint32_t) * item_ids_count;
  send_all(sock, &msg_size, sizeof(uint32_t));
  // 3. Send request message
  if (send_all(sock, msg_begin, msg_size) == -1) {
    throw std::runtime_error("Error getting sparse mf model");
  }
  
  // 4. receive user vectors and item vectors
  // FORMAT here is
  // minibatch_size * user vectors. Each vector is user_id + user_bias + NUM_FACTORS * FEATURE_TYPE
  // num_item_ids * item vectors. Each vector is item_id + item_bias + NUM_FACTORS * FEATURE_TYPE
  uint32_t to_receive_size;
  read_all(sock, &to_receive_size, sizeof(uint32_t));
  //minibatch_size * (sizeof(uint32_t) + (NUM_FACTORS + 1) * sizeof(FEATURE_TYPE)) +
  //item_ids_count * (sizeof(uint32_t) + (NUM_FACTORS + 1) * sizeof(FEATURE_TYPE));

  std::cout << "Request sent. Receiving: " << to_receive_size << " bytes" << std::endl;

  char* buffer = new char[to_receive_size];
  if (read_all(sock, buffer, to_receive_size) == 0) {
    throw std::runtime_error("");
  }

  // build a sparse model and return
  SparseMFModel model((FEATURE_TYPE*)buffer, minibatch_size, item_ids_count);
  
  delete[] msg_begin;
  delete[] buffer;

  return std::move(model);
}

// 1. send operation (uint32_t)
// 2. send gradient size (uint32_t)
// 3. send gradient data
void PSSparseServerInterface::send_mf_gradient(const MFSparseGradient& gradient) {
  uint32_t operation = SEND_MF_GRADIENT;
  if (send(sock, &operation, sizeof(uint32_t), 0) == -1) {
    throw std::runtime_error("Error sending operation");
  }

  uint32_t size = gradient.getSerializedSize();
  if (send(sock, &size, sizeof(uint32_t), 0) == -1) {
    throw std::runtime_error("Error sending grad size");
  }
  
  char* data = new char[size];
  gradient.serialize(data);
  if (send_all(sock, data, size) == 0) {
    throw std::runtime_error("Error sending grad");
  }
  delete[] data;
}
#endif
  
void PSSparseServerInterface::set_status(uint32_t id, uint32_t status) {
  std::cout << "Setting status id: " << id << " status: " << status << std::endl;
  uint32_t data[3] = {SET_TASK_STATUS, id, status};
  if (send_all(sock, data, sizeof(uint32_t) * 3) == -1) {
    throw std::runtime_error("Error setting task status");
  }
}

uint32_t PSSparseServerInterface::get_status(uint32_t id) {
  uint32_t data[2] = {GET_TASK_STATUS, id};
  if (send_all(sock, data, sizeof(uint32_t) * 2) == -1) {
    throw std::runtime_error("Error getting task status");
  }
  uint32_t status;
  if (read_all(sock, &status, sizeof(uint32_t)) == 0) {
    throw std::runtime_error("Error getting task status");
  }
  return status;
}

} // namespace cirrus

