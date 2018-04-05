#include <cstdio>

#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_(), callbacks_(), requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_(), callbacks_(), requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
//从配置文件中初始化求解器的参数
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  //param是一个配置文件中的一个SolverParameter 类型的对象，已经被类注册工厂实例化并且起了别名传递过来
  param_ = param;

  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());
  }
  // Scaffolding code
  //将训练网络和测试网络都进行初始化
  //这两步的内部会开始各自调用net.cpp函数来进行初始化训练网络和测试网络 ，这里面会对这两个网络进行初始化
  InitTrainNet();
  
  InitTestNets();

  if (Caffe::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

// Load weights from the caffemodel(s) specified in "weights" solver parameter
// into the train and test nets.
template <typename Dtype>
void LoadNetWeights(shared_ptr<Net<Dtype> > net,
    const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(","));
  for (int i = 0; i < model_names.size(); ++i) {
    boost::trim(model_names[i]);
    LOG(INFO) << "Finetuning from " << model_names[i];
    net->CopyTrainedLayersFrom(model_names[i]);
  }
}

template <typename Dtype>
//初始化训练网络
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).

  NetState net_state;
  //创建出来一个net_state的一个NetState来标志他是训练网络还是测试网络，然后将这个state对象
  //作为初始化参数来初始化 net_param 
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());

  net_param.mutable_state()->CopyFrom(net_state);
  //从这里对网络的状态进行了更新
  //之后又利用net_param来对net_这个对象重新进行赋值，来说明现在是TRAIN网络 
  //这个net_类型的网络是用来专门保存训练网络的
  //但这里并没有直接开始调用初始化
  //然后这里就进行了new Net这个操作，调用Net的构造函数，在Net的构造函数中，调用网络的初始化
  net_.reset(new Net<Dtype>(net_param));


  for (int w_idx = 0; w_idx < param_.weights_size(); ++w_idx) {
    LoadNetWeights(net_, param_.weights(w_idx));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
        //和上面的训练网络的初始化一样，只不过这里的测试网络有好几个
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    test_nets_[i]->set_debug_info(param_.debug_info());
    for (int w_idx = 0; w_idx < param_.weights_size(); ++w_idx) {
      LoadNetWeights(test_nets_[i], param_.weights(w_idx));
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  //设置开始的迭代次数(如果是从之前的snapshot恢复的，那iter_等于snapshot时的迭代次数)和结束的迭代次数
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;


  //average_loss 是次数的意思，输出的loss是前average_loss 次Loss的平均值，在solver.prototxt里面设置的，默认为1
  //losses存储之前average_loss个Loss,smoothed_loss为最后要输出的均值
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

  //进行迭代
  while (iter_ < stop_iter) {
    // zero-init the params 清空上一次所有参数的梯度
    net_->ClearParamDiffs();
    //判断这次迭代是否需要进行测试
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      if (Caffe::root_solver()) {
        TestAll();
      }
      //判断是否需要提前结束迭代
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }
      //？？
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    //判断当前迭代次数是否需要显示loss信息
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    
    // accumulate the loss and gradient
    Dtype loss = 0;
    // iter_size也是在solver.prototxt里设置，实际上的batch_size=iter_size*网络定义里的batch_size，
    // 因此每一次迭代的loss是iter_size次迭代的和，再除以iter_size，这个loss是通过调用`Net::ForwardBackward`函数得到的
    // 这个设置我的理解是在GPU的显存不够的时候使用，比如我本来想把batch_size设置为128，但是会out_of_memory，
    // 借助这个方法，可以设置batch_size=32，iter_size=4，那实际上每次迭代还是处理了128个数据
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();


    // average the loss across iterations for smoothed reporting
    // 计算要输出的smoothed_loss，如果losses里还没有存够average_loss个loss则将当前的loss插入，
    //如果已经存够了，则将之前的替换掉
    /*
     * 这个函数主要做Loss的平滑。由于Caffe的训练方式是SGD，我们无法把所有的数据同时
     * 放入模型进行训练，那么部分数据产生的Loss就可能会和全样本的平均Loss不同，在必要
     * 时候将Loss和历史过程中更新的Loss求平均就可以减少Loss的震荡问题。
     */
    UpdateSmoothedLoss(loss, start_iter, average_loss);
   
    //输出当前的迭代信息
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;

      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;

      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }

    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }

    // 执行梯度的更新，这个函数在基类`Solver`中没有实现，会调用每个子类自己的实现。
    ApplyUpdate();

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    //迭代次数加1
    ++iter_;
    
    // 调用GetRequestedAction，实际是通过action_request_function_函数指针调用之前设置好(通过`SetRequestedAction`)的
    // signal_handler的`CheckForSignals`函数，这个函数的作用是
    // 会根据之前是否遇到系统信号以及信号的类型和我们设置(或者默认)的方式返回处理的方式
    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
     // 判断当前迭代是否需要snapshot，如果request等于`SNAPSHOT`则也需要
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
    }

    // 如果request为`STOP`则修改`requested_early_exit_`为true，之后就会提前结束迭代
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}

template <typename Dtype>
//训练的主函数
void Solver<Dtype>::Solve(const char* resume_file) {
  //检查当前是不是root_solver（多GPU模式下，只有root_solver才能运行这一部分的代码）
  CHECK(Caffe::root_solver());
  //输出网络的名字
  LOG(INFO) << "Solving " << net_->name();
  //学习的策略
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.一开始赋值赋值为false,也就是现在没有要求在优化结束前推出
  requested_early_exit_ = false;
//判断 resume_file 这个指针是否为空，如果不为空，则需要从resume_file存储的路径去读之前训练的状态
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  //iter_是当前迭代的次数
  int start_iter = iter_;
  //设置迭代的次数
  //这个函数执行了实际的逐步迭代的训练过程
  Step(param_.max_iter() - iter_); 
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  //如果在`Step`函数的迭代过程中遇到了系统信号，且我们的处理方式设置为`STOP`，
  // 那么`requested_early_exit_`会被修改为true，迭代提前结束，输出相关信息
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  // 判断是否需要输出最后的loss
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
/*按网络循环，每个网络调用Test函数*/
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
/*具体每个网络的测试*/
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();/*获得信号*/
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {/*如果传入信号是保存快照，则调用Snapshot()函数保存快照*/
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;/*如果 是stop则退出*/
        }
        request = GetRequestedAction();/
    }
    if (requested_early_exit_) { /*不停接收信号*/
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss); /*执行前向传播测试图片*/
    if (param_.test_compute_loss()) {
      loss += iter_loss; /*累加损失便于后续统计*/
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j); /*保存结果*/
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  /*一些测试结果打印*/
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
/*保存快照函数*/
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();/*要么保存到二进制文件*/
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();/*要么保存到HDF5文件*/
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
/*检查对应目录下是否有保存快照文件的权限*/
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
/*生成快照的名称*/
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
/*快照以二进制proto文件形式保存*/
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  /*调用网络ToProto函数，再调用层的ToProto函数将每数据保存到proto对象中*/
  net_->ToProto(&net_param, param_.snapshot_diff());
  /*写到具体文件*/
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  /*调用网络的ToHDF5函数，网络的ToHDF5函数再调用ToHDF5的库函数保存参数*/
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}
//存储函数实现如何存储solver到快照模型中
template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    //从之前的训练的状态读取
     /*调用具体的Solver的RestoreSolverStateFromHDF5来实现, 从HDF5文件来保存快照*/
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    /*调用具体的Solver的RestoreSolverStateFromBinaryProto来实现, 从二进制文件来保存快照*/
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
/*更新平滑损失*/
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
