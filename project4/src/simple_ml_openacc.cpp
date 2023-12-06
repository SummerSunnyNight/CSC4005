#include "simple_ml_ext.hpp"

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);//多少张照片，照片的尺寸大小

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);//每一张照片都有一个label array
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;//这里为啥要除以255
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{   

    // for (size_t i = 0; i < m; ++i) {
    //     for (size_t j = 0; j < k; ++j) {
    //         C[i * k + j] = 0.0;
    //         for (size_t l = 0; l < n; ++l) {
    //             C[i * k + j] += A[i * n + l] * B[l * k + j];
    //         }
    //     }
    // }
    // BEGIN YOUR CODE
        #pragma acc data present(A[0:m*n], B[0:n*k],C[0:m*k])

    {
        #pragma acc parallel loop 

        for (size_t i = 0; i < m; ++i) {
            #pragma acc loop independent
            for (size_t j = 0; j < k; ++j) {
                float sum=0;
                #pragma acc loop independent reduction (+:sum)
                // #pragma acc loop vector

                for (size_t l = 0; l < n; ++l) {
                    // #pragma acc atomic怪了这个不能加入
                    sum += A[i * n + l] * B[l * k + j];
                }
            C[i * k + j] = sum; // 初始化 C[i * k + j] 为 0

            }
        }
    }





    // END YOUR CODE
}
//嘿嘿

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size m*n
 *     B (const float*): Matrix of size m*k
 *     C (float*): Matrix of size n * k
 **/


void matrix_dot_trans_mine(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // BEGIN YOUR CODE
    #pragma acc data present(A[0:m*n], B[0:m*k],C[0:n*k])
    {
        #pragma acc parallel loop
            for (size_t i = 0; i < n; i++) {
                #pragma acc loop

                for (size_t j = 0; j < k; j++) {
                    float sum=0;
                    #pragma acc loop independent reduction (+:sum)

                    for (size_t l = 0; l < m; l++) {
                        sum +=A[l*n + i] * B[l*k + j]; //好像这里加temp能稍微变快，但是matrix dot不行。
                    }
                    C[i*k + j]=sum;
                }
        }
    }

    // // // END YOUR CODE
    // for (size_t i = 0; i < n; i++) {
    //     for (size_t j = 0; j < k; j++) {
    //         C[i*k + j] = 0;
    //         for (size_t l = 0; l < m; l++) {
    //             C[i*k + j] += A[l*n + i] * B[l*k + j]; 
    //         }
    //     }
    // }
}


/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    // BEGIN YOUR CODE
    //这里似乎和想象的不一样，但是头文件也是这样就不改了，输入的时候注意一下
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            C[i * k + j] = 0.0;
            for (size_t l = 0; l < n; ++l) {
                C[i * k + j] += A[l * m + i] * B[l * k + j];
            }
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
// void matrix_trans_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
// {
    // BEGIN YOUR CODE
void matrix_trans_dot(const float *A, const float *B, float *C, 
                      size_t m, size_t n, size_t k) 
{
  #pragma acc data present(A[0:m*n], B[0:k*n],C[0:m*k])
  {
    #pragma acc parallel loop 
    for (size_t i = 0; i < m; ++i) {
        #pragma acc loop
      for (size_t j = 0; j < k; ++j) {
        float sum=0;
        #pragma acc loop independent reduction (+:sum)
        // C[i*k + j] = 0.0;
        for (size_t l = 0; l < n; ++l) {
          sum += A[i*n + l] * B[j*n + l];
        } 
         C[i*k + j]=sum;
      }
    }
  }
}
    // END YOUR CODE
// }

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    // for (size_t i = 0; i < m; ++i) {
    //     for (size_t j = 0; j < n; ++j) {
    //         A[i * n + j] -= B[i * n + j];
    //     }
    // }
    // END YOUR CODE
        #pragma acc data present(A[0:m*n], B[0:m*n])
    {


            #pragma acc parallel loop independent
            for (size_t i = 0; i < m; ++i) {
                #pragma acc loop independent

                for (size_t j = 0; j < n; ++j) {
                    A[i*n + j] -= B[i*n + j];
                }
            }

    }
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc data present(C[0:m*n])
    {
        {
            #pragma acc parallel loop
            for (size_t i = 0; i < m; ++i) {
                #pragma acc loop

                for (size_t j = 0; j < n; ++j) {
                    C[i*n + j] *= scalar;
                }
            }
        }
    }
    // for (size_t i = 0; i < m; i++) {
    //     for (size_t j = 0; j < n; j++) {
    //         C[i * n + j] *= scalar;
    //     }
    // }
    // END YOUR CODE
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    // BEGIN YOUR CODE
    #pragma acc data present(C[0:m*n])
    {
        {
            #pragma acc parallel loop
            for (size_t i = 0; i < m; ++i) {
                #pragma acc loop

                for (size_t j = 0; j < n; ++j) {
                    C[i*n + j] /= scalar;
                }
            }
        }
    }
    // END YOUR CODE
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    // // BEGIN YOUR CODE
    // //这里按照题意不考虑溢出情况、
    //  for (size_t i = 0; i < m; ++i) {
    //     // // 找到每一行的最大值
    //     // float max_val = *std::max_element(C + i * n, C + (i + 1) * n);

    //     // 计算 softmax 归一化的分母
    //     float denominator = 0.0;
    //     for (size_t j = 0; j < n; ++j) {
    //         // C[i * n + j] = std::exp(C[i * n + j] - max_val);
    //         C[i * n + j] = std::exp(C[i * n + j]);
    //         denominator += C[i * n + j];
    //     }

    //     // 进行 softmax 归一化
    //     for (size_t j = 0; j < n; ++j) {
    //         C[i * n + j] /= denominator;
    //     }
    // }
    // // END YOUR CODE


#pragma acc data present(C[0:m*n]) 
  {

    // #pragma acc kernels
    // {
    
      #pragma acc parallel loop independent
      for(size_t i = 0; i < m; ++i) {

        float denominator = 0.0;
        
        #pragma acc loop reduction(+:denominator)
        for(size_t j = 0; j < n; ++j) {
          C[i*n + j] = exp(C[i*n + j]);
          denominator += C[i*n + j];
        }

        #pragma acc loop independent
        for(size_t j = 0; j < n; ++j) {
          C[i*n + j] /= denominator; 
        }
      }

    // }
  
  }

}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // // BEGIN YOUR CODE
    // //这里虽然有typo但是实际没问题
    //     // 清零矩阵 Y
    // for (size_t i = 0; i < m * k; ++i) {
    //     Y[i] = 0.0;
    // }

    // // 将 y 转换为独热编码矩阵 Y
    // for (size_t i = 0; i < m; ++i) {
    //     size_t index = i * k + static_cast<size_t>(y[i]);
    //     Y[index] = 1.0;
    // }
    // END YOUR CODE
      #pragma acc data present(y[0:m],Y[0:m*k])
  {

    // #pragma acc kernels
    // {
    
      #pragma acc parallel loop independent
      for(size_t i = 0; i < m * k; ++i) {
        Y[i] = 0.0f;
      }

      #pragma acc parallel loop independent    
      for(size_t i = 0; i < m; ++i) {
      
        size_t index = i * k + static_cast<size_t>(y[i]);
        Y[index] = 1.0f;
      
      }

    // }

  }
}

/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): SGD minibatch size
 *
 * Returns:
 *     (None)
 */

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta,float*Final,float*Y,float*Z, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE

    //update the theta every epoch
        //1 Sample a mini batch of data
        //似乎可以用之前的cuts

    float coefficient= lr/batch;
    int total_row_num = m;//似乎没有clangd能帮我，将就这用把，似乎能被整除，直接用batch也可以，就不用cut了


//     float *Final = new float[n*k];
// float *Y = new float[batch*k];
// float *Z = new float[batch*k];
    //每个batch都是不同的？
    

    // #pragma acc data copyin(A[0:m*n], B[0:n*k]) copyout(C[0:m*k])


    int Final_size=n*k;
    int Z_size=batch*k;

    // cout<<"trial"<<endl;
    // int* ptr=200000; // 指针声明但未初始化

    // // 尝试使用 delete 释放未分配的内存
    // delete ptr; // 这是一个未定义行为
    // // delete ptr; // 这是一个未定义行为
    // cout<<"trial1"<<endl;
    // #pragma acc kernels
    // #pragma acc data copyin(X[0:m*n],y[0:m]) copy(theta[0:n*k])
    // {
    // #pragma acc data copyin(X[0:batch*n], theta[0:n*k],y[0:batch],Y[0:batch*k],Z[0:batch*k],Final[0:n*k]) copyout(theta[0:n*k])
    #pragma acc data present(Y[0:batch*k],Z[0:batch*k],Final[0:n*k]) 
    {
    // #pragma acc parallel loop seq 

    

    for(int Cur_batch_index=0;Cur_batch_index<total_row_num;Cur_batch_index+=batch){//每次只用这一百行进行训练
        //Cur_X是那个公式里的X
        // cout<<"achieved here1!"<<endl;

        const float*Cur_X= X + n * Cur_batch_index;//指针使用要注意，这里使用一个常量指针，指针指向的值不能被修改。
        // cout<<"X_test:";
        // for(int i=0;i<20;i++){
        //     cout<<Cur_X[i]<<",";   //这边怎么全是零
        // }
        // cout<<endl;
        const unsigned char*Cur_y= y + Cur_batch_index;
        //把Iy变出来
        // #pragma acc parallel
        // {

        vector_to_one_hot_matrix(Cur_y, Y, batch,k);

//         #pragma acc data copyin(Cur_X[0:batch*n], theta[0:n*k]) copyout(Z[0:batch*k])
// {

        // #pragma acc parallel
        // {
        matrix_dot(Cur_X ,theta,Z,batch,n,k);

        // }

// }
       


        matrix_softmax_normalize(Z,batch,k);//没问题全是零点一
        



        //Z-Iy  Z这里就不对了
        matrix_minus(Z,Y,batch,k);//也没问题

        //XT*(Z-Iy)


        matrix_dot_trans_mine(Cur_X,Z,Final,batch,n,k);//结果感觉不应该这么大？
     


        //-alpha/B*XT*(Z-Iy)

        matrix_mul_scalar(Final,coefficient,n,k);//这边输入是不是错了，应该是n*k,batch远远小于


        // cout<<"FinalTest"<<Final[0]<<endl;

        //得到最后的theta

        matrix_minus(theta,Final,n,k);
        }


    }
    // }
    // }
    


    // // cout<<"achieved here"<<endl;
    // delete[] Final;//这里等下优化一下看看能不能循环利用
    // // cout<<"achieved here"<<endl;

    // delete[] Y;
    // delete[] Z;






    // END YOUR CODE
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    // #pragma acc data copyin(A[0:m*n], B[0:n*k]) copyout(C[0:m*k])

    size_t size = train_data->input_dim * num_classes;//图像的大小乘以总的可能性
    float *theta = new float[size];//并不是很懂这个是个啥，哦theta，但是theta为啥要乘以num classes，哦没错就是这样，横着看就对了，n=imputdim,k=num_classes
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];//应该是最后的预测结果，m=images_num
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    
    float *X_test=test_data->images_matrix;
    float * X=train_data->images_matrix;
    unsigned char*y_train=train_data->labels_array;
    unsigned char*y_test=test_data->labels_array;
    size_t images_num_train=train_data->images_num;
    size_t images_num_test=test_data->images_num;
    size_t input_dim=train_data->input_dim; 
    float *Final = new float[input_dim*num_classes];
    float *Y = new float[batch*num_classes];
    float *Z = new float[batch*num_classes];


    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    // std::cout << "嘿嘿" << std::endl;
    std::chrono::milliseconds elapsed_time;

        #pragma acc data copyin(X_test[0:images_num_test*input_dim],Y[0:batch*num_classes],Z[0:batch*num_classes],Final[0:input_dim*num_classes],X[0:images_num_train * input_dim],theta[0:size],train_result[0:images_num_train*num_classes],test_result[0:images_num_test*num_classes],y_train[0:images_num_train],y_test[0:images_num_test])
        {


    static auto start_time = std::chrono::high_resolution_clock::now();
    // cout<<"What is in X:"<<endl;
    // for(int i=0;i<50000;i++){
    //     if(i%image==0){
    //         cout<<endl;
    //     }
    //     cout<<train_data->images_matrix[i]<<",";

    // }images_num * input_dim





        


    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE

        // cout<<"Change?:"<<theta[0]<<endl;//下面这个函数还是有问题，直接变成nan了。

        //train完了更新theta，然后结算result(X点乘theta)

        // #pragma acc data copy()
        {
            softmax_regression_epoch_cpp(X,y_train,theta,Final,Y,Z,images_num_train,input_dim,num_classes,lr,batch);


            matrix_dot(X,theta,train_result,images_num_train,input_dim,num_classes);
            matrix_dot(X_test,theta,test_result,images_num_test,input_dim,num_classes);
            // END YOUR CODE
            train_loss = mean_softmax_loss(train_result, y_train, images_num_train, num_classes);
            test_loss = mean_softmax_loss(test_result, y_test, images_num_test, num_classes);
            train_err = mean_err(train_result, y_train, images_num_train, num_classes);
            test_err = mean_err(test_result, y_test, images_num_test, num_classes);
        }
        
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }

        auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
        }


    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // BEGIN YOUR CODE

    // float loss = 0;
    // for(size_t i = 0; i < images_num; ++i) {
    //     float divisor=0;
    //     for(int j=0;j<num_classes;j++){//j小于写成了i小于
    //         divisor+=exp(result[i*num_classes+j]);
    //     }
    //     float logit = result[i*num_classes + labels_array[i]];
    //     loss -= log(exp(logit) / divisor);
    // }
    // loss /= images_num;


    // return loss;



    float loss = 0;

     #pragma acc data present(result[0:images_num*num_classes],labels_array[0:images_num])
  {

    // #pragma acc kernels
    // {

      #pragma acc parallel loop reduction(+:loss)
      for(size_t i = 0; i < images_num; ++i)
      {
        float divisor = 0;

        #pragma acc loop reduction(+:divisor)
        for(int j = 0; j < num_classes; ++j) {
          divisor += exp(result[i*num_classes + j]);
        }

        float logit = result[i*num_classes + labels_array[i]];
        loss -= log(exp(logit) / divisor);
      }

      loss /= images_num;
    // }

  }
    return loss;
    // END YOUR CODE
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    // // BEGIN YOUR CODE
    // size_t error_count = 0;

    // // 遍历每个样本
    // for (size_t i = 0; i < images_num; ++i) {
    //     // 获取当前样本的真实标签
    //     unsigned char label = labels_array[i];

    //     // 找到预测的最大概率对应的类别
    //     size_t predicted_class = 0;
    //     float max_prob = result[i * num_classes];
    //     for (size_t j = 1; j < num_classes; ++j) {
    //         if (result[i * num_classes + j] > max_prob) {
    //             max_prob = result[i * num_classes + j];
    //             predicted_class = j;
    //         }
    //     }

    //     // 判断预测是否正确
    //     if (predicted_class != label) {
    //         error_count++;
    //     }
    // }

    // // 计算平均错误
    // float mean_error = static_cast<float>(error_count) / static_cast<float>(images_num);

    // return mean_error;
    // // END YOUR CODE


    size_t error_count = 0;

  #pragma acc data present(result[0:images_num*num_classes],labels_array[0:images_num])
  {

    // #pragma acc kernels
    // {

      #pragma acc parallel loop reduction(+:error_count)
      for(size_t i = 0; i < images_num; ++i)
      {
        unsigned char label = labels_array[i];

        size_t predicted_class = 0;
        float max_prob = result[i*num_classes];

        #pragma acc loop reduction(max:max_prob,predicted_class)
        for(size_t j = 1; j < num_classes; ++j) {

          if(result[i*num_classes + j] > max_prob){
            max_prob = result[i * num_classes + j];
            predicted_class = j;
          }

        }

        if(predicted_class != label) {
          ++error_count;
        }
      }

    // }

  }
    float mean_error = (float)error_count / (float)images_num;

  return mean_error;
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    // // BEGIN YOUR CODE
    // for (size_t i = 0; i < size; i++)
    // {
    //     A[i] *= B[i];
    // }
    // // END YOUR CODE
    #pragma acc data present(A[0:size],B[0:size])
{    
    #pragma acc parallel loop 
    for (size_t i = 0; i < size; i++) {
        A[i] *= B[i]; 
    }}
}

void relu(float *A, size_t size)
{
    // BEGIN YOUR CODE
    #pragma acc data present(A[0:size])
    {
        #pragma acc parallel loop 
    for (size_t i = 0; i < size; i++) {
        A[i] = std::max(0.0f,A[i]);
    }}
    // END YOUR CODE
}
void zero_or_one(float *input, size_t size)
{
    // // BEGIN YOUR CODE
    // for (size_t i = 0; i < size; i++)
    // {
    //     if(input[i]>0){
    //         input[i]=1;
    //     }
    //     else{
    //         input[i]=0;
    //     }
    // }
    // // END YOUR CODE
        #pragma acc data present(input[0:size]) 
    {
        #pragma acc parallel loop 
        for (size_t i = 0; i < size; i++) 
        {
            if(input[i]>0){
                input[i]=1;
            }
            else{
                input[i]=0;
            }
        }
    }
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD minibatch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // BEGIN YOUR CODE
    // BEGIN YOUR CODE
    float *Grad_W1=new float [n*l];
    float *SW2=new float [batch*l];
    float *Grad_W2=new float [l*k];
    float *Y = new float[batch*k];
    float *S= new float [batch*k];
    float *XW1=new float[batch*l];

    #pragma acc data copyin(Grad_W1[0:n*l],SW2[0:batch*l],Grad_W2[0:l*k],Y[0:batch*k],S[0:batch*k],XW1[0:batch*l])
    {

    for (int i=0;i<m;i+=batch){
        const float*Cur_X= X + n * i;
        const unsigned char*Cur_y= y + i;

        //Calsulate S-Iy
            //Y
        vector_to_one_hot_matrix(Cur_y,Y,batch,k);
            //XW1
        memset(XW1,0,batch*l*sizeof(float));
        matrix_dot(Cur_X,W1,XW1,batch,n,l);//每次使用matrix dot的时候都要把结果矩阵清零
            //σ(XW1)
        relu(XW1,batch*l);
            //σ(XW1)*W2
        memset(S,0,batch*k*sizeof(float));
        matrix_dot(XW1,W2,S,batch,l,k);
            //softmax(σ(XW1)*W2)=S
        matrix_softmax_normalize(S, batch, k);
        matrix_minus(S,Y,batch,k);
        //done, S-Iy in S, σ(XW1) in XW1
        

        
        //Calculate gradient for W2
        memset(Grad_W2,0,l*k*sizeof(float));
        matrix_dot_trans_mine(XW1,S,Grad_W2,batch,l,k);

        //Calculate gradient for W1

            //SW2=(S-Iy)*W2T
            //这个函数比较特殊，自己就会覆盖，就不用memset了
        matrix_trans_dot(S,W2,SW2,batch,k, l);

        zero_or_one(XW1, batch*l);
        matrix_mul(SW2,XW1,batch*l);

        memset(Grad_W1,0,n*l*sizeof(float));
        matrix_dot_trans_mine(Cur_X,SW2,Grad_W1,batch,n,l);

        //Done Grad_W1 stores the XT *(S-Iy)*W2T ○ σ'(XW1)


        //Update W1 and W2

        matrix_div_scalar(Grad_W1,batch,n,l);
        matrix_div_scalar(Grad_W2,batch,l,k);


        matrix_mul_scalar(Grad_W1,lr,n,l);
        matrix_mul_scalar(Grad_W2,lr,l,k);


        matrix_minus(W1,Grad_W1,n,l);
        matrix_minus(W2,Grad_W2,l,k);



    }
    }

    delete[] Grad_W1;
    delete[] Grad_W2;
    delete[] SW2;
    delete[] XW1;
    delete[] Y;
    delete[] S;

    // END YOUR CODE
    // END YOUR CODE
}

/**
 *Example function to fully train a nn classifier
 **/
void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;//hidden dim=400,input dim=长度乘以宽度，意思应该是每个input num都有四百个触角触达到下一层
    size_t size_w2 = hidden_dim * num_classes;//hidden dim每一个都有
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    #pragma acc data copyin(W1[0:size_w1],W2[0:size_w2])
    {
        matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);//每一个元素都要除以自身的sqrt（不懂
        matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);

    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];

    float *Temp_Train=new float [hidden_dim*train_data->images_num];
    float *Temp_Test=new float [hidden_dim*test_data->images_num];



    float train_loss, train_err, test_loss, test_err;

//
    float *X_test=test_data->images_matrix;
    float * X=train_data->images_matrix;
    unsigned char*y_train=train_data->labels_array;
    unsigned char*y_test=test_data->labels_array;
    size_t images_num_train=train_data->images_num;
    size_t images_num_test=test_data->images_num;
    size_t input_dim=train_data->input_dim; 



    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    // std::cout << "alskjdklasjd给matrixdot加了个temp" << std::endl;
    std::chrono::milliseconds elapsed_time;
    #pragma acc data present(W1[0:size_w1],W2[0:size_w2])
    #pragma acc data copyin(X_test[0:images_num_test*input_dim],                                                         \
                            Temp_Train[0:hidden_dim*images_num_train],\
                            Temp_Test[0:hidden_dim*images_num_test],                \
                            X[0:images_num_train * input_dim],train_result[0:images_num_train*num_classes],\
                            test_result[0:images_num_test*num_classes],y_train[0:images_num_train],y_test[0:images_num_test])

  
    {
    auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // BEGIN YOUR CODE


        //image_matrix就是images_num*input_dim,一个大的所有数据的集合
        nn_epoch_cpp(X,y_train,W1,W2,images_num_train,input_dim,hidden_dim,num_classes,lr,batch);
        
        memset(train_result, 0, images_num_train * num_classes * sizeof(float));
        memset(test_result, 0, images_num_test * num_classes * sizeof(float));
        memset(Temp_Train, 0, hidden_dim*images_num_train * sizeof(float));
        memset(Temp_Test, 0, hidden_dim*images_num_test * sizeof(float));

        matrix_dot(X, W1, Temp_Train, images_num_train, input_dim, hidden_dim);
        relu(Temp_Train, images_num_train * hidden_dim);
        matrix_dot(Temp_Train, W2, train_result, images_num_train, hidden_dim, num_classes);

        matrix_dot(X_test, W1, Temp_Test, images_num_test, input_dim, hidden_dim);
        relu(Temp_Test, images_num_test * hidden_dim);
        matrix_dot(Temp_Test, W2, test_result, images_num_test, hidden_dim, num_classes);


        // END YOUR CODE
        train_loss = mean_softmax_loss(train_result, y_train, images_num_train, num_classes);
        test_loss = mean_softmax_loss(test_result, y_test, images_num_test, num_classes);
        train_err = mean_err(train_result, y_train, images_num_train, num_classes);
        test_err = mean_err(test_result, y_test, images_num_test, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
     elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    }
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
    delete[] Temp_Test;
    delete[] Temp_Train;
                                                              }

}
