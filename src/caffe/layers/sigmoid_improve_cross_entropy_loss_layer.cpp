#include <algorithm>
#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
//#SigmoidCrossEntropyLossLayer的输入bottom[0]，bottom[1]，
//#其中bottom[0]是输入的预测的结果，bottom[1]是标签值
template <typename Dtype>
void SigmoidCrossImproveEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  //将sigmoid_top_vec_ 和 sigmoid_output_ 两个类成员变量联系起来
  sigmoid_top_vec_.push_back(sigmoid_output_.get());

  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  /*NormalizationMode {
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
  sample_rate_value_=
    this->layer_param_.loss_param().sample_rate();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    //normalization_ 代表是除法 除以n
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void SigmoidCrossImproveEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

// TODO(shelhamer) loss normalization should be pulled up into LossLayer,
// instead of duplicated here and in SoftMaxWithLossLayer
//表示了各种除法，根据model的类型，来选择相应的除法
template <typename Dtype>
Dtype SigmoidCrossImproveEntropyLossLayer<Dtype>::get_normalizer(
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

template <typename Dtype>
//sigmoid 的输入 bottom 是二维的矩阵，行数是否为1？ 传进来是一个z值了，是不是只有一行
void SigmoidCrossImproveEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  //保存了Z值 sigmoid_bottom_vec_[0] 是一个blob, sigmoid_bottom_vec_ 是包含blob的vector，在GPU CPU版中，都可以是使用
  sigmoid_bottom_vec_[0] = bottom[0];
  //sigmoid_top_vec_ 和之前的 sigmoid_output_ 已经联系起来了
  //那么在计算逆传播的时候，sigmoid的导数等于a(1-a) a=sigmoid_output_,提前保存起来了方便逆传播
  //将sigmoid_bottom_vec_ 经过 sigmoid_layer_->Forward() 层的计算出的结果 
  //保存在sigmoid_top_vec_ 
  //sigmoid_layer_->Forward 操作的对象是一个Vector类型的数据,sigmoid_bottom_vec_ 和 sigmoid_bottom_vec_[0] 是不一样的数据类型
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  //要通过变换以后的sigmoid的算值稳定 bottom[0]->cpu_data()是上一层的输出值Z
  //bottom[0] 是一个blob,有cpu的数据和Gpu数据，,通过bottom[0]->cpu_data() 找到 前一层传进来的Z的在内存中的首个地址
  const Dtype* input_data = bottom[0]->cpu_data();
  //bottom[1]->cpu_data(); 是label标签值
  const Dtype* target = bottom[1]->cpu_data();
  
  int valid_count = 0;
  Dtype loss = 0;
  //sigmoid 上一层（全连接层）输出是一个两维的矩阵，（样本数，特征数）
  for (int i = 0; i < bottom[0]->count(); ++i) {
    //在c语言，不管target的维度，自动进行行优先/列优先的 一个一个取
    const int target_value = static_cast<int>(target[i]);
    //人为标签出现失误，有的样本没有标注数据，那么这样本就不再计算损失 进行下一个
    if (has_ignore_label_ && target_value == ignore_label_) {
      continue;
    }
    //运用稳定版本的sigmoid的运算技巧，将z>=0和z<0的情况都合并一起了
    //input_data[i] >= 0 就是1，否则为0
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    //实际进行计算Loss的次数，有的样本没有算入Loss
    ++valid_count;
  }

  normalizer_ = get_normalizer(normalization_, valid_count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossImproveEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //propagate_down[1] 表示本层有两个输入，标签输入那层不需要 求导数
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  //propagate_down[0]表示本层第一个输入，也就是y{hat}需要进行求导 
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();

    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    //标签值
    //const Dtype* target = bottom[1]->cpu_data();
    const Dtype* target[] = bottom[1]->cpu_data();
    //y     [0  ,   1    ,  0   ,  0    ,  1    ,  0   ]
    //yhat  [0.3,   0.7  ,  0.2 ,  0.2  ,  0.8  ,  0.3 ]
    //计算出1标签的数量的个数
    //仅仅将1造成的误差扩大三倍
    


    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    /*caffe_mul(count,target,sigmoid_output_data,bottom_diff);
    //标签1的减去相对应概率
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    //扩大三倍
    caffe_scal(count,sample_rate_value_,bottom_diff);
    Dtype* bottom_diff1=(Dtype*)malloc(sizeof(bottom_diff));
    caffe_sub(count,target,Dtype(1),bottom_diff1);
    caffe_mul(count,bottom_diff1,Dtype(-1),bottom_diff1);
    caffe_mul(count,bottom_diff1,target,bottom_diff1);
    caffe_add(count,bottom_diff1,bottom_diff,bottom_diff);*/
    //将标签值修改成一个数组

    //对每个标签进行计算
    
          for(int i=0 ; i<count ; i++) {
 
        bottom_diff[i] = target== 0 ? sigmoid_output_data[i]  : sample_rate_value_ * ( sigmoid_output_data[i]-1 ) ;

    }
    





    //mutable_cpu_diff 取出他的内存地址，然后算出梯度值保存进去 mutable表示可写
    //Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    //主要修改这里
    //caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      for (int i = 0; i < count; ++i) {
        const int target_value = static_cast<int>(target[i]);
        if (target_value == ignore_label_) {
          bottom_diff[i] = 0;
        }
      }
    }
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;

    caffe_scal(count, loss_weight, bottom_diff);
  }
}

/*#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossImproveEntropyLossLayer);*/
//#endif

INSTANTIATE_CLASS(SigmoidCrossImproveEntropyLossLayer);

REGISTER_LAYER_CLASS(SigmoidCrossImproveEntropyLoss);

}  // namespace caffe
