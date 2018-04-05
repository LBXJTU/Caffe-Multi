#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  //这句话从配置文件layer_param读取到
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //这句话通过读取配置文件的参数来设置本层的相应需要的参数，比如：num_output,用来下面reshape中设置
  //top的尺寸，LayerParameter 是一个共有层参数类，通过caffe.proto 获取到参数，创造出来的
  LayerParameter softmax_param(this->layer_param_);
  //设置名字
  softmax_param.set_type("Softmax");
  //通过LayerParameter 生成 softmax_layer_ 层类
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  

  softmax_bottom_vec_.clear();
  //bottom[0] 和 softmax_bottom_vec_ 联系起来，方便下面的函数使用
  softmax_bottom_vec_.push_back(bottom[0]);

  softmax_top_vec_.clear();
  //将&prob_ 和 softmax_top_vec_ 联系起来 新开辟了一个名字为&prob的空间
  //softmax_top_vec_ 是softmax_layer_里面的成员变量
  softmax_top_vec_.push_back(&prob_);
  


  //每个网络层都有都有layersetup,定义网络的结构，中间有一个reshape()函数，
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  /*// /*NormalizationMode {
    // Divide by the number of examples in the batch times spatial dimensions.
    // Outputs that receive the ignore label will NOT be ignored in computing
    // the normalization factor.
    FULL = 0;
    // Divide by the total number of output locations that do not take the
    // ignore_label.  If ignore_label is not set, this behaves like FULL.
    VALID = 1;
    // Divide by the batch size.
    BATCH_SIZE = 2;
    // Do not normalize the loss.
    NONE = 3*/
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

//主要是为了设置 outer_num 和inner_num
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);
 
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

  //这一步看源码，如果在配置文件中不进行设置会有一个默认值1，其次是为了防止使用python出现索引是-1的情况，时候会返回(-1+维度数)，如果比-1
  //更小，那么直接回在CanonicalAxisIndex 中报错，就是为了区分那几个维度是和样本有关的，那几个维度是和特征数有关的，一般情况下不用修改 返回
  //1就可以了
  //在调用SoftmaxWithLossLayer的时候，前一层一般都是全连接层，全连接层传入进来的维度是2，而layer_param_.softmax_param().axis()在配置文件中默认是1
  //意思就是第一维是样本是，也就是 softmax_axis_=1，一般正好是通道所在为位置，返回（通常）用户指定 轴 的'规范'版本，，也就是通道（可能）

  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  //softmax_axis_ 是1，count函数，就会返回[0,1)的乘积，也就是样本数
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  //softmax_axis_是1，传进去是2，看里面这个函数的具体步骤就会发现是 最后等于是1，所以inner_num_是等于1的
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output 调整维度
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}
//bttom传进来就是一个z值，因为前一层是一个全连接层，所以是一个二维矩阵，N行（n个特征），m列
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  //对列向量的每一个数求e的幂指数

  //softmax_bottom_vec_ 进行计算 保存在 softmax_top_vec_ 变量里，计算出每个样本在每个分类上的概率
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();

  //prob_.count() =样本数*特征数  outer_num是样本数，所以dim是等于一列的特征数,也就是对应全连接层的 第二个维度
  int dim = prob_.count() / outer_num_;
  int count = 0;
 
  Dtype loss = 0;
  
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      //caffe在对一个样本多分类的时候，只有一个标记值 可是0~100任何一个数，来当成一个标签值
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      //label标签大于等于0 检查 >=0
      DCHECK_GE(label_value, 0);
      //这一步是检查标签值 要小于 类别数，起一个检查的作用
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      //因为看softmax的损失函数最后的结果，是一个-logy{hat}的累加，把label_value当成一个下标在用，来做为对全连接层某个样本第几个分类取出来的
      // y{hat}值，行数代表是样本数，一行的列数代表的是特征数 也就是分类数
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  //top[0]里面存储的是损失值
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    //这个top[1]将本村的输出保存起来 因为&prob_ 和 softmax_top_vec_ 联系起来
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //propagate_down[1] 表示本层有两个输入，标签输入那层不需要 求导数
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }//propagate_down[0]表示本层第一个输入，也就是y{hat}需要进行求导 
  if (propagate_down[0]) {


    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //也就top的输出值，也就是每个样本的在每个分类上的概率值
    const Dtype* prob_data = prob_.cpu_data();
    //将prob_data, 拷贝到bottom_diff 进行 count次 
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    //将标签值取出来
    const Dtype* label = bottom[1]->cpu_data();
    //dim也就是每个样本的分类数
    int dim = prob_.count() / outer_num_;

    int count = 0;

    //outer_num_ 样本数， inner_num_是1
    for (int i = 0; i < outer_num_; ++i) {

      for (int j = 0; j < inner_num_; ++j) {
        //从一个lable中取出一个样本的标签值
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        //如果有忽略标签的标志，并且 label_value ==忽略标签的标记
        if (has_ignore_label_ && label_value == ignore_label_) 
        {

          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          //label_value 所在位置的得到的概率值 减去1，其他位置的值因为y的正确标签都是0，所以不需要去用y{hat}减0
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    //用loss_weight 乘以 bottom_diff 权重衰减
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
