#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_multilabel_all_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyMultilabelAllLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  threshold_value_ = this->layer_param_.accuracy_multilabel_all_param().threshold();
  CHECK_GT(threshold_value_, -1)
     << "The threshold vale should be great than -1.";
  CHECK_LT(threshold_value_, 1)
     << "The threshold vale should be great than one.";
}

template <typename Dtype>
void AccuracyMultilabelAllLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  /*CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions. ";*/
  //top_shape此时应该是多个标签各自的准确率，所以需要改变
  //这里top的数量是由写的网络结构中top的数量来决定，所以不需要人为的再申请空间来输出top的准确率
  

  vector<int> top_shape(0);  // 每一个的top值都是一个标量
  //对每一个top都进行一次reshape,都转换为标量
   for (int i = 0; i < top.size(); ++i) {
      top[i]->Reshape(top_shape);
    }
     

}

template <typename Dtype>
void AccuracyMultilabelAllLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //样本的个数
  const int dim = bottom[0]->shape(0);
  //样本标签的个数
  const int num_labels = bottom[0]->shape(1);
  float accuracy = 0;
  //y{hat}值，也就是预测值 指针地址
  const Dtype* bottom_data = bottom[0]->cpu_data();
  //y值，也就是真实值
  const Dtype* bottom_label = bottom[1]->cpu_data();
  //Blob<Dtype> predictions(bottom[0]->shape());
  //LOG(INFO) << "predictions shape: " ;
  // predictions.CopyFrom(*bottom[0], false, true);
  //Dtype* predictions_data = predictions.mutable_cpu_data();
  //int predictions_num = predictions.count();
  // caffe_add_scalar(predictions_num, Dtype(-threshold_value_), predictions_data);
  //计算出每个top的值并保存起来
  //应该是一个标签对一个准确率
  Dtype not_correct_cnt = 0;
  Dtype count=0;
  LOG(INFO) <<  "labels shape is"  << bottom_label->shape_string();
  LOG(INFO) <<  "data shape is "   << bottom_data->shape_string();
  for(int i=0 ;i<num_labels;i++){
    //一个标签中不正确的个数
    //这里的输入样本 和 输入标签的维度都是二维的，并且同时都是 行数代表样本数，列数代表是标签书，实质都是
    //一维数组
    not_correct_cnt = 0;
     count=0;
     accuracy=0;
    for(int j=0;j<dim;j++){
       not_correct_cnt += abs((bottom_data[j * num_labels + i] >= threshold_value_ ? 1 : 0) 
          - bottom_label[j * num_labels + i]);
       if (not_correct_cnt == Dtype(0)) ++accuracy;
       ++count;
    }
    //计算出每个标签的准确率
     top[i]->mutable_cpu_data()[0] = accuracy / count;  
  }


/*  int count = 0;
  for (int i = 0; i < dim; ++i) {
    Dtype not_correct_cnt = 0;
    for (int j = 0; j < num_labels; ++j) {
      not_correct_cnt += abs((bottom_data[i * num_labels + j] >= threshold_value_ ? 1 : 0) 
          - bottom_label[i * num_labels + j]);
    }
    if (not_correct_cnt == Dtype(0)) ++accuracy;
    ++count;
  }*/
  
  // LOG(INFO) << "Accuracy: " << accuracy;
 

  // Accuracy layer should not be used as a loss function.
}

//#ifdef CPU_ONLY
//STUB_GPU(AccuracyMultilabelLayer);
//#endif

INSTANTIATE_CLASS(AccuracyMultilabelAllLayer);
REGISTER_LAYER_CLASS(AccuracyMultilabelAll);

}  // namespace caffe
