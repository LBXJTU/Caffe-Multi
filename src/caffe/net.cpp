#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hdf5.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/insert_splits.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
  //构造函数  一个网络具体的网络结构
Net<Dtype>::Net(const NetParameter& param) {
  //Net网络的初始化，这个初始化函数会被调用train网络 和 test网络都调用
  Init(param);
}

template <typename Dtype>
//也是构造函数，从配置文件读的
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages) {
  //NetParameter代表的是一个完整的网络结构，本身有一个名字name=ResNet等等，还有一个数组型的
  //LayerParameter 的layer数据类型  repeated LayerParameter layer
  NetParameter param;

  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  //为网络设置字段，是train,还是test
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  //开始进行初始化
  Init(param);
}

template <typename Dtype>
//显示构造函数，内部调用Init函数，并且param代表的是一个完整网络结构图，里面包括了各种各样的layer层
//Train网络和Test网络各自调用这个Init两次，并打印各自网络内部的一些参数
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Set phase from the state.看这个网络是训练的网络还是测试的网络
  phase_ = in_param.state().phase();
  // Filter layers based on their include/exclude rules and
  // the current NetState.
  //因为在配置文件中，有一个训练网络有一个测试的网络，这里是为了分开
  NetParameter filtered_param;

  //想看这个网络是是什么类型的
  FilterNet(in_param, &filtered_param);
  //将网络配置文件里面的信息打印出来
  LOG_IF(INFO, Caffe::root_solver())
      << "Initializing net from parameters: " << std::endl
      << filtered_param.DebugString();
  // Create a copy of filtered_param with splits added where necessary.
  NetParameter param;
  //又定义一个新的网络，将过滤出来的信息，然后用这个过滤的信息来打印
  InsertSplits(filtered_param, &param);
  // Basically, build all the layers and set up their connections.
  //将这个网络里面的所有名字都放入到一个name_ 的这个序列里面
  name_ = param.name();
  //想要根据name_内部的层的顺序 将这些类按照这个顺序连接起来
  //？？？这里是哪里传进去的
  map<string, int> blob_name_to_idx;
  
  set<string> available_blobs;
  
  memory_used_ = 0;
  // For each layer, set up its input and output
  //vector<vector<Blob<Dtype>*> > bottom_vecs_; 所有第一个vector代表的是param这个NetParam中所有的网络层数，第二个vector代表的是每个层里面bottom里面的数量 ，每一个bottom是一个blob
  //下面所有都同理
  bottom_vecs_.resize(param.layer_size()); //存每一层的输入(bottom)blob指针
  top_vecs_.resize(param.layer_size());// 存每一层输出(top)的blob指针 
  bottom_id_vecs_.resize(param.layer_size());// 存每一层输入(bottom)blob的id 
  param_id_vecs_.resize(param.layer_size());// 存每一层参数blob的id
  top_id_vecs_.resize(param.layer_size());// 存每一层输出(top)的blob的id  
  bottom_need_backward_.resize(param.layer_size());//该blob是需要返回的bool值 


  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Inherit phase from net if unset.
    //如果在网络配置文件中没有明确的设置 phase这个字段 
    //如果开头的input data层里面有个phase ，写的是那个是train,那个是test，只需要开头写，下面的没有设置
    //就默认和上面输入的phase一样，因为train的网络和test不一样，比如：train网络需要逆传播，而test不需要逆传播

    //这块的意思就是为了检查这个网络是test的 还是train的
    if (!param.layer(layer_id).has_phase()) {
      //将整个网络的每一层网络都的phase都赋值 这个网络的标志，是test还是train
      param.mutable_layer(layer_id)->set_phase(phase_);
    }

    // Setup layer.
    //外层是一个循环，获取到NetParameter里面的每一个网络层的信息
    const LayerParameter& layer_param = param.layer(layer_id);
    
    //是否需要进行逆传播
    if (layer_param.propagate_down_size() > 0) {

      CHECK_EQ(layer_param.propagate_down_size(),
          layer_param.bottom_size())
          << "propagate_down param must be specified "
          << "either 0 or bottom_size times ";
    }

    //用类工厂创建每一个层，这是在一个循环的内部
    //layer_param 这个包含了本层网络层的信息，进行注册，并且添加到layers_内
    layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
    
    layer_names_.push_back(layer_param.name());


    LOG_IF(INFO, Caffe::root_solver())
        << "Creating Layer " << layer_param.name();
    bool need_backward = false;

    // Figure out this layer's input and output
    //对每一个网络层传递bottom进行添加初始化
    //bottom_size是caffe.protobuff进行映射算出大小，没有具体代码怎么映射
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      //给这个网络的第几层的第几个bottom添加 
      //其实这个一层的bottom的空间地址和上一层的top的空间地址是一样的。
      //这一层的意思是给某一个网络层，根据他的bottom的个数来进行添加bottom
      //param表示当前的训练网络或者是测试网络，layer_id表示当前param网络中的某一层网络层，bottom_id表示这个网络层添加到第几个Bottom了
      //available_blobs 是一个set防止在本层添加bottom的时候名字重复，blob_name_to_idx 代表的是本层第几个bottom
      const int blob_id = AppendBottom(param, layer_id, bottom_id,
                                       &available_blobs, &blob_name_to_idx);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
    }
    //top_size从配置文件获取到个数
    int num_top = layer_param.top_size();


    //此函数为该层创建top blob，该函数真正的new的一个blob的对象。并将topblob 的指针压入到top_vecs_中
    for (int top_id = 0; top_id < num_top; ++top_id) {

      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      //因为数据层的输入层 和 其他的层的bottom是不一样的，所以需要特殊的处理
      //正常的输入层的type=data,不是 input？？？Data层有专门初始化，不再这里初始化
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
    // If the layer specifies that AutoTopBlobs() -> true and the LayerParameter
    // specified fewer than the required number (as specified by
    // ExactNumTopBlobs() or MinTopBlobs()), allocate them here.
    //到这里这一层的bottom和top全部都初始化完了

    //找出本层网络层有多少个top
    Layer<Dtype>* layer = layers_[layer_id].get();

    if (layer->AutoTopBlobs()) {

      const int needed_num_top =
      //如果一个没配，就用另一个
          std::max(layer->MinTopBlobs(), layer->ExactNumTopBlobs());

      for (; num_top < needed_num_top; ++num_top) {
        // Add "anonymous" top blobs -- do not modify available_blobs or
        // blob_name_to_idx as we don't want these blobs to be usable as input
        // to other layers.
        //为什么要要调用两次
        AppendTop(param, layer_id, num_top, NULL, NULL);
      }
    }
    // After this layer is connected, set it up.
    
    //上面是创建了信息，这里就要开始进行set up了，拿到某一层的输入和输出就开进行建立层次了 
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    
    LOG_IF(INFO, Caffe::root_solver())
        << "Setting up " << layer_names_[layer_id];

    //对每一个层的top开始进行循环
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      
      //在blob逆传播的时候，都会向前传播的，通过blob_loss_weigts 来调整这个层是多传一点loss还是i少穿
      //1就是 正常传播
      /// Vector of weight in the loss (or objective) function of each net blob,
      /// indexed by blob_id.
      //vector<Dtype> blob_loss_weights_; 个数和blob 的个数一样
      // vector<vector<int> > top_id_vecs_;  top_id_vecs_[layer_id][top_id] 代表的是当前层的 的某一个top输出 是blob中的哪一个blob_id 
      if (blob_loss_weights_.size() <= top_id_vecs_[layer_id][top_id]) {
        //对blob_loss_weights的个数进行调整，不断的加1
        blob_loss_weights_.resize(top_id_vecs_[layer_id][top_id] + 1, Dtype(0));
      }
      //这一句是在取每一个blob的权重损失的系数
      blob_loss_weights_[top_id_vecs_[layer_id][top_id]] = layer->loss(top_id);
     

      LOG_IF(INFO, Caffe::root_solver())
          << "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();

      if (layer->loss(top_id)) {
        LOG_IF(INFO, Caffe::root_solver())
            << "    with loss weight " << layer->loss(top_id);
      }

      //把所有要用内存都累加起来，将每一层的所有的blob的几个维度乘积全部加起来
      memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    //将需要的内存打印出来
    LOG_IF(INFO, Caffe::root_solver())
        << "Memory required for data: " << memory_used_ * sizeof(Dtype);

    //将这一层的参数的个数全部统计出来，参数的个数，需要配置的个数
    // param_size 是Layermeter 类型对象layer_param 中ParamSpec param 成员的个数,
    // 是层内blob_ 的数量，即该层有几个权重参数（每个blob内有一个参数）
    // 例如；cov层和IP层都有两个参数对应w和b
    // repeated .caffe.ParamSpec param = 6;
    ////指定训练参数（全局学习常数的乘数，以及用于权重共享的名称和其他设置）。

    //此处表示的是 从ResNet.prototxt这种具体的网络配置文件中读取到的需要配置的参数，这一层有需要共享的参数
    const int param_size = layer_param.param_size();

    
    //应该是这个网络层的blob的个数统计出来，包括输入输出  和一些可以作为blob的参数 ，这一层本来的参数的blobs的个数
    //网络层中包括数字参数的blob
    //layers_是从层注册工厂中根据具体层制作出来的的层，仅仅是根据网络配置文件中网络的名称而制作出来的模板
    const int num_param_blobs = layers_[layer_id]->blobs().size();



    //做检测用 看参数有没又配置的过多
    CHECK_LE(param_size, num_param_blobs)
        << "Too many params specified for layer " << layer_param.name();
    
    ParamSpec default_param_spec;

    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      //看用全局的共享参数 还是 用自己的参数 就好比solver里面的学习率 还是 卷积层自己的学习率
      const ParamSpec* param_spec = (param_id < param_size) ?
      //layer_param.param(param_id) 代表到具体的一个网络层，已经有配置好的参数
          &layer_param.param(param_id) : &default_param_spec;
            // 这里说明了如果在prototxt 中将lr置为0，即关掉，该层参数便不再更新
      const bool param_need_backward = param_spec->lr_mult() != 0;
      //按位或
      need_backward |= param_need_backward;
      //归根到底要设置每一层的参数要不要回传
      layers_[layer_id]->set_param_propagate_down(param_id,
                                                  param_need_backward);
    }
    

    //num_param_blobs 所有blobs
    //这个层中param的初始化
    //这个blob   和bottom  top  的blob是没有关系的,只是一些参数的blob
    // 添加parameter blob,如果当前layer没有parameter blob(num_param_blobs==0), 
    // 比如ReLU，那么就不进入循环，不添加parameter blob     
    // AppendParam 只是执行为当前layer 添加parameter blob 的相关工作， 
    // 并不会修改与backward的相关属性  
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      // 将param blob以及blob的id添加到params_,param_id_vecs_等 
      //为层注册工厂注册出来的层添加参数
      AppendParam(param, layer_id, param_id);
    
    }
    

    // Finally, set the backward flag，然后就设置这个网络层是否需要你传播
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
       //需要的话 将这个层的所有top都进行修改
      for (int top_id = 0; top_id < top_id_vecs_[layer_id].size(); ++top_id) {
        blob_need_backward_[top_id_vecs_[layer_id][top_id]] = true;
      }
    }
  }
  //到这里的，他的top bottom和 param都初始化完了 
  // Go through the net backwards to determine which blobs contribute to the
  // loss.  We can skip backward computation for blobs that don't contribute
  // to the loss.
  // Also checks if all bottom blobs don't need backward computation (possible
  // because the skip_propagate_down param) and so we can skip bacward
  // computation for the entire layer
  //后通过网络来确定哪些blob造成了损失。 我们可以跳过对不会造成损失的斑点的后向计算。 
  //同时检查所有底层blob是否不需要反向计算（可能是因为skip_propagate_down参数），所以我们可以跳过整个层的bacwardcomputation
  
  set<string> blobs_under_loss;  //那些blobs产生了损失？
  
  set<string> blobs_skip_backp; //那些块需要你传播

  for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
    bool layer_contributes_loss = false;
    bool layer_skip_propagate_down = true;
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
      if (layers_[layer_id]->loss(top_id) ||
          (blobs_under_loss.find(blob_name) != blobs_under_loss.end())) {
        layer_contributes_loss = true;
      }
      if (blobs_skip_backp.find(blob_name) == blobs_skip_backp.end()) {
        layer_skip_propagate_down = false;
      }
      if (layer_contributes_loss && !layer_skip_propagate_down)
        break;
    }
    // If this layer can skip backward computation, also all his bottom blobs
    // don't need backpropagation
    if (layer_need_backward_[layer_id] && layer_skip_propagate_down) {
      layer_need_backward_[layer_id] = false;
      for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
    }
    if (!layer_contributes_loss) { layer_need_backward_[layer_id] = false; }
    if (Caffe::root_solver()) {
      if (layer_need_backward_[layer_id]) {
        LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      } else {
        LOG(INFO) << layer_names_[layer_id]
            << " does not need backward computation.";
      }
    }
    for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
         ++bottom_id) {
      if (layer_contributes_loss) {
        const string& blob_name =
            blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_under_loss.insert(blob_name);
      } else {
        bottom_need_backward_[layer_id][bottom_id] = false;
      }
      if (!bottom_need_backward_[layer_id][bottom_id]) {
        const string& blob_name =
                   blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
        blobs_skip_backp.insert(blob_name);
      }
    }
  }
  // Handle force_backward if needed. 如果需要的画 就传播
  if (param.force_backward()) {
    for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      layer_need_backward_[layer_id] = true;
      for (int bottom_id = 0;
           bottom_id < bottom_need_backward_[layer_id].size(); ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] =
            bottom_need_backward_[layer_id][bottom_id] ||
            layers_[layer_id]->AllowForceBackward(bottom_id);
        blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] =
            blob_need_backward_[bottom_id_vecs_[layer_id][bottom_id]] ||
            bottom_need_backward_[layer_id][bottom_id];
      }
      for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
           ++param_id) {
        layers_[layer_id]->set_param_propagate_down(param_id, true);
      }
    }
  }
  // In the end, all remaining blobs are considered output blobs. 剩下的就被认为需要输出的blob
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {

    LOG_IF(INFO, Caffe::root_solver())
        << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
    net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
  ShareWeights();
  debug_info_ = param.debug_info();
  LOG_IF(INFO, Caffe::root_solver()) << "Network initialization done.";
}

template <typename Dtype>
//因为在最之前的slover中，已经对这个NetParameter的 phase字段已经指定是 Train还是Test
void Net<Dtype>::FilterNet(const NetParameter& param,
    NetParameter* param_filtered) {

  NetState net_state(param.state());
  //把训练网络的参数复制到测试网络的参数
  param_filtered->CopyFrom(param);
  //因为用的训练层复制过来的测试网络，所以需要将原来的训练层的参数给清除掉，下面是为了给这个测试网络添加参数
  param_filtered->clear_layer();

  //param.layer_size() 是这个网络结构中的网络层数
  for (int i = 0; i < param.layer_size(); ++i) {
    //log    可以看出来 训练网络和测试网络 分别都初始化了出来，获取其中的某一层网络层的信息
    const LayerParameter& layer_param = param.layer(i);
    
    const string& layer_name = layer_param.name();
 

    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
          << "Specify either include rules or exclude rules; not both.";
    // If no include rules are specified, the layer is included by default and
    // only excluded if it meets one of the exclude rules.
    bool layer_included = (layer_param.include_size() == 0);

    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.exclude(j), layer_name)) {
        layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (StateMeetsRule(net_state, layer_param.include(j), layer_name)) {
        layer_included = true;
      }
    }
    if (layer_included) {

      param_filtered->add_layer()->CopyFrom(layer_param);
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::StateMeetsRule(const NetState& state,
    const NetStateRule& rule, const string& layer_name) {
  // Check whether the rule is broken due to phase.
  if (rule.has_phase()) {
      if (rule.phase() != state.phase()) {
        LOG_IF(INFO, Caffe::root_solver())
            << "The NetState phase (" << state.phase()
            << ") differed from the phase (" << rule.phase()
            << ") specified by a rule in layer " << layer_name;
        return false;
      }
  }
  // Check whether the rule is broken due to min level.
  if (rule.has_min_level()) {
    if (state.level() < rule.min_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the min_level (" << rule.min_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to max level.
  if (rule.has_max_level()) {
    if (state.level() > rule.max_level()) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState level (" << state.level()
          << ") is above the max_level (" << rule.max_level()
          << ") specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to stage. The NetState must
  // contain ALL of the rule's stages to meet it.
  for (int i = 0; i < rule.stage_size(); ++i) {
    // Check that the NetState contains the rule's ith stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState did not contain stage '" << rule.stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  // Check whether the rule is broken due to not_stage. The NetState must
  // contain NONE of the rule's not_stages to meet it.
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    // Check that the NetState contains the rule's ith not_stage.
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      LOG_IF(INFO, Caffe::root_solver())
          << "The NetState contained a not_stage '" << rule.not_stage(i)
          << "' specified by a rule in layer " << layer_name;
      return false;
    }
  }
  return true;
}

// Helper for Net::Init: add a new top blob to the net.
template <typename Dtype>
//一个网络层有多少个top就要调用这个AppendTop多少次
//此函数为该层创建top blob，该函数真正的new的一个blob的对象。并将topblob 的指针压入到top_vecs_中
//available_blobs 代表由appendtop所产生的可供后面bottom添加的blob
//blob_name_to_idx 将名字和序号对应起来，是整个网络中的，可以多对1，但不能一对多，也就是多个bottom和 top指向同一个blob
void Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                           const int top_id, set<string>* available_blobs,
                           map<string, int>* blob_name_to_idx) {
 
  //定义了一个新的层，一个网络层 有多少个top就要创建多少个临时的LayerParameter名字叫 layer_param??
  //？智能指针，因为每次里面new 的对象都是一样的，所以都只会返回第一次的new的对象
  //怀疑这里返回的是同一个对象，不是多个，而是获得当前网络层？？
  //这句话的意思，有多少top就要初始化多个layer层，多个top不能放在
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));

  //如果当前创建的top数还没有到配置文件的定义的top数
  //获取到这个top的名字的这个blob的名字
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";


  // Check if we are doing in-place computation检查我们是否正在进行就地计算
  //layer_param->bottom_size() > top_id 也就是说本层的bottom需要的数量大于 现在要产生有的top的数量
  // 意思是，如本层想直接把输入当成输出给输出出去，那么他就要将这个输出的blob的顺序要和输入的bolb的顺序一致，也就是bottom1=a,bottom2=b,如果这个a想直接输出，就必须是top1=a...不能是top2=a
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      //本层内此时要产生的top如果等于输入的名字话，也就是同一个blob
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    LOG_IF(INFO, Caffe::root_solver())
        << layer_param->name() << " -> " << blob_name << " (in-place)";
    //(*blob_name_to_idx)[blob_name] 是一个int值，代表的是整个网络中的第几个blob
    //blobs_[] 是 vector<shared_ptr<Blob<Dtype> > > blob的集合
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    //将这个blob的编号放入 top_id_vecs_中
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } 
  //也就是这层网络的top输出的名字和之前blob的输出的名字重复，就会报错
  //也就是一个blob必须只有一个名字，他只能作为他前面网络层的一个输出，但是他可以作为后面多个网络层的输入
  else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } 
  else {
    //当这个网络层输入输出都正常 不一样的时候
    // Normal output.
    if (Caffe::root_solver()) {
      LOG(INFO) << layer_param->name() << " -> " << blob_name;
    }
    //创建新的blob出来
    shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
    //每创建一个新的，那么此时的blob_id就是blobs_的大小
    const int blob_id = blobs_.size();
    //这里定义blobs_的结构，将新生成的blob压入栈中
    blobs_.push_back(blob_pointer);
    //这些blob的名字是不能重复的，且可同时在bottom和top中出现
    blob_names_.push_back(blob_name);
    //这个blob是否需要逆传播，这个blob里面由data,由diff,既能是Bottom,又能是top
    blob_need_backward_.push_back(false);

    //键值对赋值
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    //将这个blob的id和层的id联系起来
    top_id_vecs_[layer_id].push_back(blob_id);
    //将层和blob的地址联系起来
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }

  //将这个blob的名字插入进去，告诉后面的bottom可以使用了
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
//在上面的
//此函数为该层创建bottom blob，由于网络是堆叠而成，即：当前层的输出 bottom是前一层的输出top blob，因此此函数并没没有真正的创建blob，只是在将前一层的指针压入到了bottom_vecs_中。
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
    const int bottom_id, set<string>* available_blobs,
    map<string, int>* blob_name_to_idx) {
  //层的参数都进来
  const LayerParameter& layer_param = param.layer(layer_id);
  //layer_param此时是某一个网络中某一个具体的网络层
  //bottom的id都进来，就是配置里面的bottom的名字读取进来
  //根据bottom_id，可以都出来，当前层的第几个Bottom，这些东西是网络配置文件中配置的，由google的protobuff来进行自动设置
  //blob_name保存的是本层的那个bottom的名字
  const string& blob_name = layer_param.bottom(bottom_id);
  //从头找到尾，名字不能重复，这些bottom的名字
  
  //available_blobs里面存的是所有层的bottom的名字，第一次在调用appendbottom的时候，这句话应该是没有执行的？直接中断？
  //意思就是 从网络结构配置中，不允许一个layer中有两个一摸一样的bottom存在，这个available_blobs 应该是在appendtop中有添加，来提供给后面的所有layer的bottom使用
  //日志文件这块是没有执行的？？
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  //如果找到这个Bottom名字了，就在map里面根据这个bottom的名字当成索引，取出来bottom id取出来
  const int blob_id = (*blob_name_to_idx)[blob_name];
  
  LOG_IF(INFO, Caffe::root_solver())
      << layer_names_[layer_id] << " <- " << blob_name;

  //其实这个一层的bottom的空间地址和上一层的top的空间地址是一样的。
  //将这个 bottom的这个名字压入到栈，分配空间实在类工厂中进行的申请空间
  //bottom_vecs_ 是所有层的bottom都存在一个数据结构中，bottom_vecs_ 将每个网络层 和 各自的bottom的 指针地址 保存起来，这是一个vector
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  
  //bottom_id_vecs_ 是将 所有层的id 和 各自层的bottom的id 都联系起来
  bottom_id_vecs_[layer_id].push_back(blob_id);
 
  //用完一个Bottom就要把他删掉，意思就是本层的bottom不能重名，但是这里是对一个指针来进行操作的，所以删掉后，后面的layer层将对不能对之前的blob重复使用？？？
  available_blobs->erase(blob_name);
  
  bool need_backward = blob_need_backward_[blob_id];
  // Check if the backpropagation on bottom_id should be skipped
  
  //看这个bottom是否需要backward
  if (layer_param.propagate_down_size() > 0) {
    need_backward = layer_param.propagate_down(bottom_id);
  }
  bottom_need_backward_[layer_id].push_back(need_backward);
  
  return blob_id;
}

template <typename Dtype>
//这里是对每一层的网络层的参数进行保存
void Net<Dtype>::AppendParam(const NetParameter& param, const int layer_id,
                             const int param_id) {

  //layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));
  //获取到层注册工厂注册出来的类，根据具体的网络结构中的层制作出来的层，里面有所有需要的参数
  const &layer_param = layers_[layer_id]->layer_param();
  //获得网络层中的参数
  const int param_size = layer_param.param_size();
  //如果这个配置的param_id不大于层模板的参数的尺寸，那么就根据param_id 取出在层模板中的名字
  string param_name =
      (param_size > param_id) ? layer_param.param(param_id).name() : "";

  //如果存在param_name这个参数
  if (param_name.size()) {
    //就把这个参数放入到param_display_names_
    param_display_names_.push_back(param_name);
  } else {
    //param_name为空的情况下也就说出现了param_id出现了大于情况 就将param_id作为字符串放进去
    ostringstream param_display_name;
    param_display_name << param_id;
    param_display_names_.push_back(param_display_name.str());
  }

  //获取到当前params_参数blob的第几个的序列数
  const int net_param_id = params_.size();

  //将这个由这个网络层制作出来的blob放入到整体网络的params存放的blobs中
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  //并且在 param_id_vecs_ 中放入这个blob的id数 
  param_id_vecs_[layer_id].push_back(net_param_id);
  //将参数块和网络的层联系起来，作为一个键值对，放入param_layer_indices_，因为这个参数块可能会被不同的网络层
  param_layer_indices_.push_back(make_pair(layer_id, param_id));

  ParamSpec default_param_spec;

  const ParamSpec* param_spec = (layer_param.param_size() > param_id) ?

      &layer_param.param(param_id) : &default_param_spec;

  if (!param_size || !param_name.size() || (param_name.size() &&
      param_names_index_.find(param_name) == param_names_index_.end())) {
    // This layer "owns" this parameter blob -- it is either anonymous
    // (i.e., not given a param_name) or explicitly given a name that we
    // haven't already seen.
    //这是一个新的参数块
    //该层“拥有”该参数blob - 它可以是匿名的（即，没有给出param_name），也可以显式给出我们尚未见过的名称。
    param_owners_.push_back(-1);
    //map<string, int> param_names_index_;  将参数块和名字放入键值对
    if (param_name.size()) {
      param_names_index_[param_name] = net_param_id;
    }

    //每个参数的学习率
    const int learnable_param_id = learnable_params_.size();
    learnable_params_.push_back(params_[net_param_id].get());
    learnable_param_ids_.push_back(learnable_param_id);

    //参数的基础学习率和衰减率
    has_params_lr_.push_back(param_spec->has_lr_mult());
    has_params_decay_.push_back(param_spec->has_decay_mult());
    params_lr_.push_back(param_spec->lr_mult());
    params_weight_decay_.push_back(param_spec->decay_mult());

  } else {
    //这不是一个新的参数块，之前的参数快向量里面由
    // Named param blob with name we've seen before: share params

    //获得到这个参数名字在参数向量中的序列数
    const int owner_net_param_id = param_names_index_[param_name];
    //？？？这个参数的含义 param_owners_
    param_owners_.push_back(owner_net_param_id);
    
    //当前的Blob块和那个网络层联系在一起
    //param_layer_indices_.push_back(make_pair(layer_id, param_id));
    const pair<int, int>& owner_index =
        param_layer_indices_[owner_net_param_id];

    const int owner_layer_id = owner_index.first;
    const int owner_param_id = owner_index.second;

    //分享参数，指出当前需要用的这个参数块由之前哪个网络层创建的
    LOG_IF(INFO, Caffe::root_solver()) << "Sharing parameters '" << param_name
        << "' owned by "
        << "layer '" << layer_names_[owner_layer_id] << "', param "
        << "index " << owner_param_id;

    //获取到当前这个网络的 相对应 这个参数块的 之前创建出来的 真实地址
    Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
    //获取到这个块 之前最早的 被创建出来的网络层 的真实地址 
    Blob<Dtype>* owner_blob =
        layers_[owner_layer_id]->blobs()[owner_param_id].get();

    const int param_size = layer_param.param_size();

    if (param_size > param_id && (layer_param.param(param_id).share_mode() ==
                                  ParamSpec_DimCheckMode_PERMISSIVE)) {
 
    //具体共享操作？？？？

      // Permissive dimension checking -- only check counts are the same.
      //检查要被共享的两个块的维度的所有乘机是否一样，不一样就能共享
      CHECK_EQ(this_blob->count(), owner_blob->count())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; count mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "shape is " << this_blob->shape_string();
    } else {
      // Strict dimension checking -- all dims must be the same.
      //在维度相等的情况下每个维度都必须是一样的
      CHECK(this_blob->shape() == owner_blob->shape())
          << "Cannot share param '" << param_name << "' owned by layer '"
          << layer_names_[owner_layer_id] << "' with layer '"
          << layer_names_[layer_id] << "'; shape mismatch.  Owner layer param "
          << "shape is " << owner_blob->shape_string() << "; sharing layer "
          << "expects shape " << this_blob->shape_string();
    }

    //整个网络应该还有个学习率的向量
    const int learnable_param_id = learnable_param_ids_[owner_net_param_id];
    learnable_param_ids_.push_back(learnable_param_id);

    if (param_spec->has_lr_mult()) {

      if (has_params_lr_[learnable_param_id]) {

        CHECK_EQ(param_spec->lr_mult(), params_lr_[learnable_param_id])

            << "Shared param '" << param_name << "' has mismatched lr_mult.";
      } else {

        has_params_lr_[learnable_param_id] = true;
        params_lr_[learnable_param_id] = param_spec->lr_mult();
      }
    }
    if (param_spec->has_decay_mult()) {
      if (has_params_decay_[learnable_param_id]) {
        CHECK_EQ(param_spec->decay_mult(),
                 params_weight_decay_[learnable_param_id])
            << "Shared param '" << param_name << "' has mismatched decay_mult.";
      } else {
        has_params_decay_[learnable_param_id] = true;
        params_weight_decay_[learnable_param_id] = param_spec->decay_mult();
      }
    }
  }
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  LOG_EVERY_N(WARNING, 1000) << "DEPRECATED: Forward(bottom, loss) "
      << "will be removed in a future version. Use Forward(loss).";
  // Copy bottom to net bottoms
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return Forward(loss);
}

template <typename Dtype>
void Net<Dtype>::BackwardFromTo(int start, int end) {
  CHECK_GE(end, 0);
  CHECK_LT(start, layers_.size());
  for (int i = start; i >= end; --i) {
    for (int c = 0; c < before_backward_.size(); ++c) {
      before_backward_[c]->run(i);
    }
    if (layer_need_backward_[i]) {
      layers_[i]->Backward(
          top_vecs_[i], bottom_need_backward_[i], bottom_vecs_[i]);
      if (debug_info_) { BackwardDebugInfo(i); }
    }
    for (int c = 0; c < after_backward_.size(); ++c) {
      after_backward_[c]->run(i);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ForwardDebugInfo(const int layer_id) {
  for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
    const Blob<Dtype>& blob = *top_vecs_[layer_id][top_id];
    const string& blob_name = blob_names_[top_id_vecs_[layer_id][top_id]];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", top blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const int net_param_id = param_id_vecs_[layer_id][param_id];
    const string& blob_name = param_display_names_[net_param_id];
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Forward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << blob_name
        << " data: " << data_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<Blob<Dtype>*>& bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }
    const Blob<Dtype>& blob = *bottom_vec[bottom_id];
    const string& blob_name = blob_names_[bottom_id_vecs_[layer_id][bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", bottom blob " << blob_name
        << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }
    const Blob<Dtype>& blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Backward] "
        << "Layer " << layer_names_[layer_id]
        << ", param blob " << param_id
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::UpdateDebugInfo(const int param_id) {
  const Blob<Dtype>& blob = *params_[param_id];
  const int param_owner = param_owners_[param_id];
  const string& layer_name = layer_names_[param_layer_indices_[param_id].first];
  const string& param_display_name = param_display_names_[param_id];
  const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
  if (param_owner < 0) {
    const Dtype data_abs_val_mean = blob.asum_data() / blob.count();
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param " << param_display_name
        << " data: " << data_abs_val_mean
        << "; diff: " << diff_abs_val_mean;
  } else {
    const string& owner_layer_name =
        layer_names_[param_layer_indices_[param_owner].first];
    LOG_IF(INFO, Caffe::root_solver())
        << "    [Update] Layer " << layer_name
        << ", param blob " << param_display_name
        << " (owned by layer " << owner_layer_name << ", " << "param "
        << param_display_names_[param_owners_[param_id]] << ")"
        << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(const Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK(target_blobs[j]->shape() == source_blob->shape())
          << "Cannot share param " << j << " weights from layer '"
          << source_layer_name << "'; shape mismatch.  Source param shape is "
          << source_blob->shape_string() << "; target param shape is "
          << target_blobs[j]->shape_string();
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::BackwardFrom(int start) {
  BackwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::BackwardTo(int end) {
  BackwardFromTo(layers_.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::Backward() {
  BackwardFromTo(layers_.size() - 1, 0);
  if (debug_info_) {
    Dtype asum_data = 0, asum_diff = 0, sumsq_data = 0, sumsq_diff = 0;
    for (int i = 0; i < learnable_params_.size(); ++i) {
      asum_data += learnable_params_[i]->asum_data();
      asum_diff += learnable_params_[i]->asum_diff();
      sumsq_data += learnable_params_[i]->sumsq_data();
      sumsq_diff += learnable_params_[i]->sumsq_diff();
    }
    const Dtype l2norm_data = std::sqrt(sumsq_data);
    const Dtype l2norm_diff = std::sqrt(sumsq_diff);
    LOG(ERROR) << "    [Backward] All net params (data, diff): "
               << "L1 norm = (" << asum_data << ", " << asum_diff << "); "
               << "L2 norm = (" << l2norm_data << ", " << l2norm_diff << ")";
  }
}

template <typename Dtype>
void Net<Dtype>::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  if (H5Fis_hdf5(trained_filename.c_str())) {
    CopyTrainedLayersFromHDF5(trained_filename);
  } else {
    CopyTrainedLayersFromBinaryProto(trained_filename);
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromBinaryProto(
    const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFromHDF5(const string trained_filename) {
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_layers = hdf5_get_num_links(data_hid);
  for (int i = 0; i < num_layers; ++i) {
    string source_layer_name = hdf5_get_name_by_idx(data_hid, i);
    if (!layer_names_index_.count(source_layer_name)) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    int target_layer_id = layer_names_index_[source_layer_name];
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    hid_t layer_hid = H5Gopen2(data_hid, source_layer_name.c_str(),
        H5P_DEFAULT);
    CHECK_GE(layer_hid, 0)
        << "Error reading weights from " << trained_filename;
    // Check that source layer doesn't have more params than target layer
    int num_source_params = hdf5_get_num_links(layer_hid);
    CHECK_LE(num_source_params, target_blobs.size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      ostringstream oss;
      oss << j;
      string dataset_name = oss.str();
      int target_net_param_id = param_id_vecs_[target_layer_id][j];
      if (!H5Lexists(layer_hid, dataset_name.c_str(), H5P_DEFAULT)) {
        // Target param doesn't exist in source weights...
        if (param_owners_[target_net_param_id] != -1) {
          // ...but it's weight-shared in target, so that's fine.
          continue;
        } else {
          LOG(FATAL) << "Incompatible number of blobs for layer "
              << source_layer_name;
        }
      }
      hdf5_load_nd_dataset(layer_hid, dataset_name.c_str(), 0, kMaxBlobAxes,
          target_blobs[j].get());
    }
    H5Gclose(layer_hid);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::ToHDF5(const string& filename, bool write_diff) const {
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";
  hid_t diff_hid = -1;
  if (write_diff) {
    diff_hid = H5Gcreate2(file_hid, "diff", H5P_DEFAULT, H5P_DEFAULT,
        H5P_DEFAULT);
    CHECK_GE(diff_hid, 0) << "Error saving weights to " << filename << ".";
  }
  for (int layer_id = 0; layer_id < layers_.size(); ++layer_id) {
    const LayerParameter& layer_param = layers_[layer_id]->layer_param();
    string layer_name = layer_param.name();
    hid_t layer_data_hid = H5Gcreate2(data_hid, layer_name.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_GE(layer_data_hid, 0)
        << "Error saving weights to " << filename << ".";
    hid_t layer_diff_hid = -1;
    if (write_diff) {
      layer_diff_hid = H5Gcreate2(diff_hid, layer_name.c_str(),
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      CHECK_GE(layer_diff_hid, 0)
          << "Error saving weights to " << filename << ".";
    }
    int num_params = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_params; ++param_id) {
      ostringstream dataset_name;
      dataset_name << param_id;
      const int net_param_id = param_id_vecs_[layer_id][param_id];
      if (param_owners_[net_param_id] == -1) {
        // Only save params that own themselves
        hdf5_save_nd_dataset<Dtype>(layer_data_hid, dataset_name.str(),
            *params_[net_param_id]);
      }
      if (write_diff) {
        // Write diffs regardless of weight-sharing
        hdf5_save_nd_dataset<Dtype>(layer_diff_hid, dataset_name.str(),
            *params_[net_param_id], true);
      }
    }
    H5Gclose(layer_data_hid);
    if (write_diff) {
      H5Gclose(layer_diff_hid);
    }
  }
  H5Gclose(data_hid);
  if (write_diff) {
    H5Gclose(diff_hid);
  }
  H5Fclose(file_hid);
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    learnable_params_[i]->Update();
  }
}

template <typename Dtype>
void Net<Dtype>::ClearParamDiffs() {
  for (int i = 0; i < learnable_params_.size(); ++i) {
    Blob<Dtype>* blob = learnable_params_[i];
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_set(blob->count(), static_cast<Dtype>(0),
                blob->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_set(blob->count(), static_cast<Dtype>(0),
                    blob->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    }
  }
}

template <typename Dtype>
//不用手动调
void Net<Dtype>::ShareWeights() {
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) { continue; }
    params_[i]->ShareData(*params_[param_owners_[i]]);
    params_[i]->ShareDiff(*params_[param_owners_[i]]);
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) const {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) const {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
