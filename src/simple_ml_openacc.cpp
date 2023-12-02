#include "simple_ml_openacc.hpp"

#include <openacc.h>

void matrix_dot_openacc(const float *A, const float *B,  
                        float *C, size_t m, size_t n, size_t k) 
{
  // BEGIN YOUR CODE

  #pragma acc data copyin(A[0:m*n], B[0:n*k]) copyout(C[0:m*k])
  {
    #pragma acc kernels
    {
      #pragma acc loop independent
      for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < n; ++l) {
          for (size_t j = 0; j < k; ++j) {
            C[i*k + j] += A[i*n + l] * B[l*k + j];
          }
        }  
      }
    } 
  }
  
  // END YOUR CODE
}



void matrix_dot_trans_mine_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{   
    // BEGIN YOUR CODE
      #pragma acc data copyin(A[0:m*n], B[0:n*k]) copyout(C[0:m*k])
  {
    #pragma acc kernels
    {
      #pragma acc loop independent
      for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < n; ++l) {
          for (size_t j = 0; j < k; ++j) {
            C[i*k + j] += A[l*n + i] * B[l*k + j];
          }
        }  
      }
    } 
  }
    // END YOUR CODE
}
// void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
// {
//     // BEGIN YOUR CODE

//     // END YOUR CODE
// }

// void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
// {
//     // BEGIN YOUR CODE
//     #pragma acc parallel loop

//     for (size_t i = 0; i < m; ++i) {
//         for (size_t j = 0; j < k; ++j) {
//             C[i * k + j] = 0.0;
//             for (size_t l = 0; l < n; ++l) {
//                 C[i * k + j] += A[i * n + l] * B[j * n + l];
//             }
//         }
//     }
//     // END YOUR CODE
// }

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
        #pragma acc data copy(A[0:m*n], B[0:m*n])
    {
        #pragma acc kernels
        {
            #pragma acc loop independent
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    A[i*n + j] -= B[i*n + j];
                }
            }
        }
    }
    // END YOUR CODE
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
        #pragma acc data copy(C[0:m*n])
    {
        #pragma acc kernels
        {
            #pragma acc loop independent 
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    C[i*n + j] *= scalar; 
                }
            }
        }
    }


    // END YOUR CODE
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE

      #pragma acc data copy(C[0:m*n]) 
  {
    #pragma acc kernels
    {
      #pragma acc loop independent
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            C[i*n + j] /= scalar;
        }
      }
    }
  }

    // END YOUR CODE
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    // BEGIN YOUR CODE

 #pragma acc data copy(C[0:m*n])
  {
    #pragma acc kernels
    {  
      #pragma acc loop independent
      for (size_t i = 0; i < m; ++i) {

        float max_val = -INFINITY;
        
        // 找到最大值
        #pragma acc loop reduction(max:max_val)
        for (size_t j = 0; j < n; ++j) {
            max_val = fmax(max_val, C[i*n + j]); 
        }

        float denominator = 0.0;
        
        // 计算softmax
        #pragma acc loop reduction(+:denominator)  
        for (size_t j = 0; j < n; ++j) {
            C[i*n + j] = exp(C[i*n + j] - max_val);
            denominator += C[i*n + j]; 
        }

        // 归一化 
        #pragma acc loop independent
        for (size_t j = 0; j < n; ++j) {
            C[i*n + j] /= denominator; 
        }
      }
    }
  }
    // END YOUR CODE
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // BEGIN YOUR CODE

  #pragma acc data copyin(y[0:m]) copyout(Y[0:m*k]) 
  {
    #pragma acc kernels
    {
      #pragma acc loop independent  
      for (size_t i = 0; i < m * k; ++i) {
          Y[i] = 0.0f;
      }

      #pragma acc loop independent   
      for(size_t i = 0; i < m; ++i)
      {
        size_t index = i * k + static_cast<size_t>(y[i]);
        Y[index] = 1.0f;  
      }
    }
  }

    // END YOUR CODE
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
{
// BEGIN YOUR CODE

    //update the theta every epoch
        //1 Sample a mini batch of data
        //似乎可以用之前的cuts

    float coefficient= lr/batch;
    int total_row_num = m;//似乎没有clangd能帮我，将就这用把，似乎能被整除，直接用batch也可以，就不用cut了


    float *Final = new float[n*k];
    float *Y = new float[batch*k];//每个batch都是不同的？
    float *Z = new float[batch*k];



    int Final_size=n*k;
    int Z_size=batch*k;
  

    for(int Cur_batch_index=0;Cur_batch_index<total_row_num;Cur_batch_index+=batch){//每次只用这一百行进行训练

        const float*Cur_X= X + n * Cur_batch_index;//指针使用要注意，这里使用一个常量指针，指针指向的值不能被修改。

        const unsigned char*Cur_y= y + Cur_batch_index;
        //把Iy变出来
        vector_to_one_hot_matrix_openacc(Cur_y, Y, batch,k);

        memset(Z, 0, Z_size*sizeof(float));
        matrix_dot_openacc(Cur_X,theta,Z,batch,n,k);

        

        matrix_softmax_normalize_openacc(Z,batch,k);//没问题全是零点一

        


        //Z-Iy  Z这里就不对了

        matrix_minus_openacc(Z,Y,batch,k);//也没问题


        //XT*(Z-Iy)
        memset(Final, 0, Final_size*sizeof(float));

        matrix_dot_trans_mine_openacc(Cur_X,Z,Final,batch,n,k);//结果感觉不应该这么大？


        //-alpha/B*XT*(Z-Iy)

        matrix_mul_scalar_openacc(Final,coefficient,n,k);//这边输入是不是错了，应该是n*k,batch远远小于

        


        //得到最后的theta

        matrix_minus_openacc(theta,Final,n,k);

        



        
    }

    // cout<<"achieved here"<<endl;
    delete[] Final;//这里等下优化一下看看能不能循环利用
    // cout<<"achieved here"<<endl;

    delete[] Y;
    delete[] Z;






    // END YOUR CODE
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    /*
    Example function to fully train a softmax regression classifier
    */
    size_t size = train_data->input_dim * num_classes;//图像的大小乘以总的可能性
    float *theta = new float[size];//并不是很懂这个是个啥，哦theta，但是theta为啥要乘以num classes，哦没错就是这样，横着看就对了，n=imputdim,k=num_classes
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];//应该是最后的预测结果，m=images_num
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    // cout<<"What is in X:"<<endl;
    // for(int i=0;i<50000;i++){
    //     if(i%image==0){
    //         cout<<endl;
    //     }
    //     cout<<train_data->images_matrix[i]<<",";
        
    // }
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE

        // cout<<"Change?:"<<theta[0]<<endl;//下面这个函数还是有问题，直接变成nan了。

        //train完了更新theta，然后结算result(X点乘theta)
        softmax_regression_epoch_openacc(train_data->images_matrix,train_data->labels_array,theta,train_data->images_num,train_data->input_dim,num_classes,lr,batch);
        
        memset(train_result, 0, train_data->images_num * num_classes * sizeof(float));
        memset(test_result, 0, test_data->images_num * num_classes * sizeof(float));
        matrix_dot_openacc(train_data->images_matrix,theta,train_result,train_data->images_num,train_data->input_dim,num_classes);
        matrix_dot_openacc(test_data->images_matrix,theta,test_result,test_data->images_num,test_data->input_dim,num_classes);
        // END YOUR CODE
        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    // float *Y = new float[images_num*k];
    // vector_to_one_hot_matrix(labels_array, Y, images_num,k);
    //Y already one hot encoded
    float loss = 0;
    for(size_t i = 0; i < images_num; ++i) {
        float divisor=0;
        for(int j=0;j<num_classes;j++){//j小于写成了i小于
            divisor+=exp(result[i*num_classes+j]);
        }
        float logit = result[i*num_classes + labels_array[i]];
        loss -= log(exp(logit) / divisor); 
    }
    loss /= images_num;
    
    // delete[] Y;用不到
    return loss;
}

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE
    size_t error_count = 0;

    // 遍历每个样本

    for (size_t i = 0; i < images_num; ++i) {
        // 获取当前样本的真实标签
        unsigned char label = labels_array[i];

        // 找到预测的最大概率对应的类别
        size_t predicted_class = 0;
        float max_prob = result[i * num_classes];
        for (size_t j = 1; j < num_classes; ++j) {
            if (result[i * num_classes + j] > max_prob) {
                max_prob = result[i * num_classes + j];
                predicted_class = j;
            }
        }

        // 判断预测是否正确
        if (predicted_class != label) {
            error_count++;
        }
    }

    // 计算平均错误
    float mean_error = static_cast<float>(error_count) / static_cast<float>(images_num);

    return mean_error;
    // END YOUR CODE
}

// void matrix_mul_openacc(float *A, const float *B, size_t size)
// {
//     // BEGIN YOUR CODE

//     // END YOUR CODE
// }

// void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
// {
//     // BEGIN YOUR CODE

//     // END YOUR CODE
// }

// void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
// {
//     size_t size_w1 = train_data->input_dim * hidden_dim;
//     size_t size_w2 = hidden_dim * num_classes;
//     float *W1 = new float[size_w1];
//     float *W2 = new float[size_w2];
//     std::mt19937 rng;
//     rng.seed(0);
//     std::normal_distribution<float> dist(0.0, 1.0);
//     for (size_t i = 0; i < size_w1; i++)
//     {
//         W1[i] = dist(rng);
//     }
//     for (size_t i = 0; i < size_w2; i++)
//     {
//         W2[i] = dist(rng);
//     }
//     matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
//     matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
//     size_t size_tr = train_data->images_num * num_classes;
//     size_t size_te = test_data->images_num * num_classes;
//     float *train_result = new float[size_tr];
//     float *test_result = new float[size_te];
//     float train_loss, train_err, test_loss, test_err;
//     std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
//     std::chrono::milliseconds elapsed_time;
//     auto start_time = std::chrono::high_resolution_clock::now();
//     for (size_t epoch = 0; epoch < epochs; epoch++)
//     {
//         // BEGIN YOUR CODE

//         // END YOUR CODE
//         train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
//         test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
//         train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
//         test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
//         std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
//                   << std::fixed << std::setprecision(5) << train_loss << " |   "
//                   << std::fixed << std::setprecision(5) << train_err << " |   "
//                   << std::fixed << std::setprecision(5) << test_loss << " |  "
//                   << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
//     }
//     auto end_time = std::chrono::high_resolution_clock::now();
//     elapsed_time =
//         std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
//                                                               start_time);
//     std::cout << "Execution Time: " << elapsed_time.count()
//               << " milliseconds\n";
//     delete[] W1;
//     delete[] W2;
//     delete[] train_result;
//     delete[] test_result;
// }
