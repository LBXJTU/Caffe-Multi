#include <vector>

#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
#向前传播 的过程
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
#损失层的前一层一定是全连接层，那么此时的原理就和DNN的原理是一样的，将一个样本平铺成一列，然后有m个列，得到全连接层计算出来的的结果就是两维

    const vector<Blob<Dtype>*>& top) {
#bottom 是这一层的输入，他可能有多个输入  bottom[0]就是 y值，bottom[1]就是y {hat}值
#bottom[0]->count(); y这个向量里面有多少个元素
  int count = bottom[0]->count(); 
# caffe_sub 实在将y 和 y{hat}进行相减，一共进行count = 全连接层两个维度的乘积，也就是（m * 特征数 ）次，然后保存在 diff_.mutable_cpu_data() 这是一个向量
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
#caffe_cpu_dot 在做对diff_.mutable_cpu_data() 这个向量进行对应元素的相乘再加起来
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
#dot 然后除以 样本的个数，再除以2
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
#将损失值保存起来
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
#propagate_down是为了方便以后的扩充，而设置出来的一个接口，因为再第一层到L层中，如果某一层需要反馈也就是需要计算出损失函数，propagate_down就直接代表的就是那个具体层的top的L层的值

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
#在计算损失层的时候，如果这个层是 他的top[0]->cpu_diff()[0] =1,就相当于自己对自己求导 就是1，就外层的倒数  top[0]->mutable_cpu_data()[0]=loss
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
#这块原理就相当与分别对y1 y2 y3求偏导，然后将公因数提取出来就是 1/n * (  dy1 , dy2 , dy3    )   dy1=y1-y1{hat}   dy2=y2-y2{hat}  dy3=y3-y3{hat}
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}
