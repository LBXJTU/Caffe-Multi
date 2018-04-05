#include <algorithm>
#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //slice_param 是一个 SliceParameter的对象，一共有axis、slice_point和slice_dim三个属性
  const SliceParameter& slice_param = this->layer_param_.slice_param();

  CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
      << "Either axis or slice_dim should be specified; not both.";
  //slice_point()是一个vector
  slice_point_.clear();
  //vector清空，并从参数将切分点拷贝进来
  //这里的slice_point()是一个数组类型，里面保存的都有要切割的几个轴，然后把要切割轴从头复制
  //到尾到slice_point_这个vector里面
  std::copy(slice_param.slice_point().begin(),
      slice_param.slice_point().end(),
      std::back_inserter(slice_point_));
}

template <typename Dtype>
void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //返回的是这个blob的一共有几个轴也就是几个维度 这里就是2，因为标签最为bottom是两个维度，在我们修改成多标签的时候
  const int num_axes = bottom[0]->num_axes();
  //获取到SliceParameter这个对象，里面就有一共有axis、slice_point和slice_dim三个属性
  const SliceParameter& slice_param = this->layer_param_.slice_param();
   //如果指定切分维度的话，则在指定维度上进行切分，比如在H维
  if (slice_param.has_slice_dim()) {
    //获取到指定的切分维度
    slice_axis_ = static_cast<int>(slice_param.slice_dim());
    // Don't allow negative indexing for slice_dim, a uint32 -- almost
    // certainly unintended.
    CHECK_GE(slice_axis_, 0) << "casting slice_dim from uint32 to int32 "
        << "produced negative result; slice_dim must satisfy "
        << "0 <= slice_dim < " << kMaxBlobAxes;
        //检查切分维度是否超出总维度数
    CHECK_LT(slice_axis_, num_axes) << "slice_dim out of range.";
  } else {
    //否则就是将axis作为slice_axis的切分维度，这一步的操作就是为了防止索引不对，代表的就是0 1 第几个索引
    //所以这里的slice_axis=1
    slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis());
  }
  //获取到bottom的维度，放在top_shape中
  vector<int> top_shape = bottom[0]->shape();
  //获取到要进行切分的维度数上的具体的值，该轴上具体的值，比如 8个标签，所以这里等于8
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  //num_slices_就是第一个维度上的值，也就是样本的个数  
  //计算每次切分在一个batch中需要切分的次数，若在channel维进行切分，等于batchsize
  //几个样本就对label_这个一维数组切几刀，把各自的标签都划分好
  num_slices_ = bottom[0]->count(0, slice_axis_);
  //返回的就是h*w的值，如果没有4个维度，那么就默认为1 
  //计算每次切分最小feature map的大小，若在channel维进行切分，等于HxW
  //label_是存放在bottom[1]中，他的存放的是二维数据
  ////<batch-size,每个样本的标签数量>
  //top_label[item_id*datum.labels_size() + labi] = datum.labels(labi)，所以这里算出来是1
  //以后还有可能对h*w进行切分，这里留出接口
  slice_size_ = bottom[0]->count(slice_axis_ + 1);
  
  int count = 0;
  //我们在切分标签的时候，并没有配置这个slice_point_这个属性，所以是0空的，所以下面这段没有执行
  if (slice_point_.size() != 0) {
    
    CHECK_EQ(slice_point_.size(), top.size() - 1);
    CHECK_LE(top.size(), bottom_slice_axis)
        << "slice axis: " << slice_axis_
        << ", bottom[0] shape: " << bottom[0]->shape_string();
    int prev = 0;

    vector<int> slices;
    for (int i = 0; i < slice_point_.size(); ++i) {
      //判断是否大于
      CHECK_GT(slice_point_[i], prev);
      
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }

    slices.push_back(bottom_slice_axis - prev);
    for (int i = 0; i < top.size(); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
    //上面这块没有执行
  } else {
    //这里是均匀切分！上面提到bottom_slice_axis是8，而我们的top也是8，所以准备均匀切分
    CHECK_EQ(bottom_slice_axis % top.size(), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    //top_shape是一个二维的vector<int>类型的数据 
    //slice_axis_是=1的 所以top_shape[1]=1;
    top_shape[slice_axis_] = bottom_slice_axis / top.size();
    for (int i = 0; i < top.size(); ++i) {
      //将每一个top[i]都重新reshape成 <batch-size,1>，但这里还没有进行赋值，只是将空间的尺寸都分配好了
      top[i]->Reshape(top_shape);
      //这个blob所有维度之和，这里的意思应该是想看到底切了多少次
      count += top[i]->count();
    }
  }
  //判断且的次数，是否曾于2个维度之积，相等就说明对了
  CHECK_EQ(count, bottom[0]->count());
  if (top.size() == 1) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

    //简单总结的来说，就是 bottom 中存的每个样本的标签，是一个实质上是一个一维的连续数组，而top[0]~top[7]每个top内部是一个连续1维数组，而每个top之间不连续
    //在data_layer.cpp中已经标出 label是一个二维的
     ////上面已经将label的shape转换成了<label_size,batchsize>的二维了
    //Dtype* top_label = batch->label_.mutable_cpu_data();
      //对label进行了 获取和 赋值 所以需要对label的函数进行修改
       //因为在配置文件中，label是datum数据结构中的一个repeated的数据类型，其实这里是一条一条数据读取进来，这条数据包含了数据+n个标签值！，然后将这里面的标签取出来保存好，再去下一条数据
       //top_label已经指向了label_这个blob的内存地址了，然后直接对这个数组进行操作修改
      //top_label[item_id*datum.labels_size() + labi] = datum.labels(labi);
template <typename Dtype>
void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  //特殊说明，bottom现在就是label,是两个维度的标签，<batch-size,8>
  //top一共有8个，每个都是<batch-size,1>
  //如果等于1说明直接不用切了
  if (top.size() == 1) { return; }

  int offset_slice_axis = 0;
  //获取到bottom[0]标签的指针
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //bottom_slice_axis=8，slice_axis_是等于1的
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
   //做法就是对每个输出计算输出和输入之间的offset关系，然后拷贝一下数据就ok了
  /*需要注意的是，这个二维数组在存储的时候实际上是一维数组的格式 这是bottom的数据存储格式
  样本0：[0,1，2，3，4，5，6，7]
  样本1：[0,1，2，3，4，5，6，7]
  样本2：[0,1，2，3，4，5，6，7]
  ...
  */
  /*top的存储格式
  top[0]  标签0：[样本0的标签0，样本1的标签0，.....样本n的标签0]
  top[1]  标签1：[样本0的标签1，样本1的标签1，.....样本n的标签1]
  top[2]  标签0：[样本0的标签2，样本1的标签2，.....样本n的标签2]
   ...
  top[7]  标签7：.....
  */
  //一个标签一个标签的复制
  for (int i = 0; i < top.size(); ++i) {
    //获取每一个标签的top指针，来复制所有样本的 同一个标签值，这里的top的存储的时候，不是一个连续的空间应该，所以需要每次重新获取
    Dtype* top_data = top[i]->mutable_cpu_data();
    //top[i]都重新reshape成 <batch-size,1>
    //因为我们之前是均匀切分的，所以这里的top_slice_axis是等于1的，  
    const int top_slice_axis = top[i]->shape(slice_axis_);
    //num_slices_是一个hpp中的变量，已经在setupLayer设置过了，等于样本的个数，也就是对bottom进行切的刀数
    for (int n = 0; n < num_slices_; ++n) {
    //slice_size_也是等于1的，前面已经计算过了，所以这个top_offset指的直接就是top[i]中的第几个位置，他的索引和样本数的索引是一样的
      const int top_offset = n * top_slice_axis * slice_size_;
    //计算bottom的位置
    //offset_slice_axis 会在当前的for循环外面进行+1，也就是和样本的标签数有关
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
    //直接进行地址的复制   
    //top_slice_axis * slice_size_=1 
    //一个标签要复制 num_slices_次，也就是下面这个caffe_copy要复制的次数
    //top_data 是每个top的也就是每个标签的首地址，top_offset是就是1~n
    //bottom_data是 标签的地址，bottom_offset是每次定位的索引，
    //比如：offset_slice_axis每次增长的步长为1，增加完了之后，bottom_offset在遍历所有样本的过程中，会重复使用n次，并且每次都不一样
      caffe_copy(top_slice_axis * slice_size_,
          bottom_data + bottom_offset, top_data + top_offset);
    }

    offset_slice_axis += top_slice_axis;
  }  //至此就将标签已经全部分号
     
}    
     
template <typename Dtype>
void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0] || top.size() == 1) { return; }
  int offset_slice_axis = 0;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const int top_slice_axis = top[i]->shape(slice_axis_);
    for (int n = 0; n < num_slices_; ++n) {
      const int top_offset = n * top_slice_axis * slice_size_;
      const int bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy(top_slice_axis * slice_size_,
          top_diff + top_offset, bottom_diff + bottom_offset);
    }
    offset_slice_axis += top_slice_axis;
  }
}

#ifdef CPU_ONLY
STUB_GPU(SliceLayer);
#endif

INSTANTIATE_CLASS(SliceLayer);
REGISTER_LAYER_CLASS(Slice);

}  // namespace caffe
