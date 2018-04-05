#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);

  
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);

    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    //类别数，方便下面计算每个类别的准确率
    top[1]->Reshape(top_shape_per_class);
    //
    nums_buffer_.Reshape(top_shape_per_class);
  
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  //返回的是分类数也就是样本的特征数，每个类别的应该有多少个样本
  const int num_labels = bottom[0]->shape(label_axis_);
  

  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    //对每个类别在做初始化
    //计算出的每个样本的个数
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }


  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {

      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);

      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }

      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);

      //看这个样本本身属于哪个类别，进行++
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];

      const Dtype prob_of_true_class = bottom_data[i * dim
                                                   + label_value * inner_num_
                                                   + j];
      //y1hat[0.1 0.1 0.1 0 0.7]  y=5
      //y2hat[0.1 0.2 0.6 0.1 0]  y=3

      int num_better_predictions = -1;  // true_class also counts as "better"
      // Top-k accuracy
      for (int k = 0; k < num_labels && num_better_predictions < top_k_; ++k) {
        //找出比原本类别计算出来的概率更大的值
        num_better_predictions +=
          (bottom_data[i * dim + k * inner_num_ + j] >= prob_of_true_class);
      }
      // check if there are less than top_k_ predictions 将正确的值作为一个阈值，
      //大于这个阈值的有几个数 也就是 可以忍犯错的个数 
      if (num_better_predictions < top_k_) {
        ++accuracy;
        //每个类计算正确有多少个
        if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
      }
      ++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  //这个值算出估计出来正确的 正确率
  top[0]->mutable_cpu_data()[0] = (count == 0) ? 0 : (accuracy / count);
  //从别类的角度，看每个别类的估计的正确率
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

#ifdef CPU_ONLY
STUB_GPU(AccuracyLayer);
#endif

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
