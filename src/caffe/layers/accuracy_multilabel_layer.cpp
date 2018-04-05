#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_multilabel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyMultilabelLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  threshold_value_ = this->layer_param_.accuracy_multilabel_param().threshold();
  CHECK_GT(threshold_value_, -1)
     << "The threshold vale should be great than -1.";
  CHECK_LT(threshold_value_, 1)
     << "The threshold vale should be great than one.";
}

template <typename Dtype>
void AccuracyMultilabelLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions. ";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);

}

template <typename Dtype>
void AccuracyMultilabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int dim = bottom[0]->shape(0);
  const int num_labels = bottom[0]->shape(1);
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  //Blob<Dtype> predictions(bottom[0]->shape());
  //LOG(INFO) << "predictions shape: " ;
  // predictions.CopyFrom(*bottom[0], false, true);
  //Dtype* predictions_data = predictions.mutable_cpu_data();
  //int predictions_num = predictions.count();
  // caffe_add_scalar(predictions_num, Dtype(-threshold_value_), predictions_data);
  int count = 0;
  for (int i = 0; i < dim; ++i) {
    Dtype not_correct_cnt = 0;
    for (int j = 0; j < num_labels; ++j) {
      not_correct_cnt += abs((bottom_data[i * num_labels + j] >= threshold_value_ ? 1 : 0) 
          - bottom_label[i * num_labels + j]);
    }
    if (not_correct_cnt == Dtype(0)) ++accuracy;
    ++count;
  }

  /*
  LOG(INFO) << "predictions shape: " << predictions.shape_string() 
            << ". " << predictions.data_at(0,0,0,0)
            << ", " << predictions.data_at(0,1,0,0)
            << ", " << predictions.data_at(0,2,0,0);
  */



  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;

  // Accuracy layer should not be used as a loss function.
}

//#ifdef CPU_ONLY
//STUB_GPU(AccuracyMultilabelLayer);
//#endif

INSTANTIATE_CLASS(AccuracyMultilabelLayer);
REGISTER_LAYER_CLASS(AccuracyMultilabel);

}  // namespace caffe
