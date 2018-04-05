#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num_instances = bottom[0]->shape(1);
  for (int i = 0; i < bottom.size(); ++i) {
    if (i == 2) {
      //bottom 2 是2维度的 标志位
      CHECK_EQ(2, bottom[i]->num_axes());
    } else {
      //bottom 0,1都是3维度的 上一层输入 和 本层的输入
      CHECK_EQ(3, bottom[i]->num_axes());
    }
    //不管什么输入 记忆单元 还是输入 第一维都是1, 因为 RNN 是可以处理不同长度的序列的神经网络，视频输入进来了，每个视频的帧数不一样
    //16个视频同时每一帧每一帧的进行训练
    CHECK_EQ(1, bottom[i]->shape(0));
    //所有的输入的 第二个维度都是和第一个输入的第二个维度 是相等的
    CHECK_EQ(num_instances, bottom[i]->shape(1));
  }
  //Lstm unit单元里面的 每个i f o g 网络的 W矩阵 神经元数 有多少行
  hidden_dim_ = bottom[0]->shape(2);
  //输入的 第三个维度是 神经元行数的4倍 ，因为里面包括了4个门计算好的数据，所以要是4倍 
  CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2));
  
  top[0]->ReshapeLike(*bottom[0]);
  top[1]->ReshapeLike(*bottom[0]);
  X_acts_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //c_prev词的个数
  const int num = bottom[0]->shape(1);
  //将4个公式的w矩阵合并成一列，同时准备乘以一个输入x
  const int x_dim = hidden_dim_ * 4;
  //bottom[0]是c_prev 上一层的记忆的输出
  const Dtype* C_prev = bottom[0]->cpu_data();
  //每一帧的输入或者每一个单词的输入
  const Dtype* X = bottom[1]->cpu_data();
  //标志位 bottom[2]
  const Dtype* cont = bottom[2]->cpu_data();
  //两个输出的 一个c
  Dtype* C = top[0]->mutable_cpu_data();
  //一个h
  Dtype* H = top[1]->mutable_cpu_data();
  //16个视频 同一个时间的帧 要放在一个矩阵中同时并行处理，所以就要求所有的这16个视频的帧数是一样的，才可以继续运行
  for (int n = 0; n < num; ++n) {
    //d是最多有多少个神经单元的数量，比如有4个门，就是4行，每行的列数就是前面w矩阵的行数，联想DNN
    for (int d = 0; d < hidden_dim_; ++d) {
       //X[1 * hidden_dim_ + d]是在取值
      const Dtype i = sigmoid(X[d]);
       //实质存储是一个连续的一个数组，在里面进行取址运算，hidden_dim就是 矩阵的行数，在计算的时候 要一个一个的取出来，对应的实际的数组存放就是列
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));

      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      
      const Dtype c_prev = C_prev[d];
      //c 是 W矩阵对应的每一行计算出来的值
      const Dtype c = f * c_prev + i * g;
      //d的个数 就是w的行数
      C[d] = c;
      const Dtype tanh_c = tanh(c);
      H[d] = o * tanh_c;
    }
    C_prev += hidden_dim_;
    //X+=4*256 意思就是 加到下一个字的 四个门的 中间计算值 
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    ++cont;
  }
}

template <typename Dtype>
void LSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }
  //输入的字数
  const int num = bottom[0]->shape(1);
  const int x_dim = hidden_dim_ * 4;
  
  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* X = bottom[1]->cpu_data();

  const Dtype* cont = bottom[2]->cpu_data();
  const Dtype* C = top[0]->cpu_data();
  const Dtype* H = top[1]->cpu_data();
  const Dtype* C_diff = top[0]->cpu_diff();
  const Dtype* H_diff = top[1]->cpu_diff();
  //对c的偏导
  Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
  //对x的偏导
  Dtype* X_diff = bottom[1]->mutable_cpu_diff();
  
  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
      //拿数字
      const Dtype i = sigmoid(X[d]);
      const Dtype f = (*cont == 0) ? 0 :
          (*cont * sigmoid(X[1 * hidden_dim_ + d]));
      const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
      const Dtype g = tanh(X[3 * hidden_dim_ + d]);
      const Dtype c_prev = C_prev[d];
      const Dtype c = C[d];
      const Dtype tanh_c = tanh(c);
     
      //指针往前走，取地址
      Dtype* c_prev_diff = C_prev_diff + d;
      Dtype* i_diff = X_diff + d;
      Dtype* f_diff = X_diff + 1 * hidden_dim_ + d;
      Dtype* o_diff = X_diff + 2 * hidden_dim_ + d;
      Dtype* g_diff = X_diff + 3 * hidden_dim_ + d;
     
      //求导
      //因为这里要看Ct对整体输出的影响，那么他包括了两部分，一步是上面的ct，一部分是下面的ht，上面一部分的的ct的导数是： C_diff[d]
      // 下面的ht对ct的倒数 就是H_diff[d] * o * (1 - tanh_c * tanh_c) 
       /*i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ]
 *     f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ]
 *     o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ]
 *     g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ]
 *     c_t := (f_t .* c_{t-1}) + (i_t .* g_t)
 *     h_t := o_t .* \tanh[c_t]*/
      const Dtype c_term_diff =
          C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
      
      *c_prev_diff = c_term_diff * f;
      
      *i_diff = c_term_diff * g * i * (1 - i);

      *f_diff = c_term_diff * c_prev * f * (1 - f);
      //x=w [h,x]+b
      //损失函数 dl/dx=dl/dh * dh/do * do/dx
      *o_diff = H_diff[d] * tanh_c * o * (1 - o);
      *g_diff = c_term_diff * i * (1 - g * g);
    }
    C_prev += hidden_dim_;
    X += x_dim;
    C += hidden_dim_;
    H += hidden_dim_;
    C_diff += hidden_dim_;
    H_diff += hidden_dim_;
    X_diff += x_dim;
    C_prev_diff += hidden_dim_;
    ++cont;
  }
}

#ifdef CPU_ONLY
STUB_GPU(LSTMUnitLayer);
#endif

INSTANTIATE_CLASS(LSTMUnitLayer);
REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe
