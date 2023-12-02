// #include<iostream>
// using namespace std;

// #pragma acc routine
// void example_func(int *i){
//     *i+=10;
// }
// int main(){
//     int i=1;
//     int *shuaige=i;

// #pragma copy(shuaige)
// #pragma acc parallel num_gangs(10)

// {

//   example_func(shuaige);
// }
// cout<<shuaige;
// }

#include<iostream>
using namespace std;

void process_data_on_gpu(int size, float *data) {
    // 在 GPU 上处理数据
    #pragma acc loop
    for (int i = 0; i < size; ++i) {
        data[i] = data[i] * 2.0;
    }

}

int main() {
    const int size = 100;
    float data[size];

// 初始化为特定值，例如0.0
for (int i = 0; i < 100; ++i) {
    if(i==1){
        data[i]=2;
    }
    else{
        data[i] = 1.0;

    }
}
    cout<<"trial:"<<*(data+1);


    // 初始化数据

    // 在进入并行区域之前，使用 data copy 将数据传输到 GPU
    #pragma acc data copy(data[0:size])
    // #pragma acc enter data copyin(data[0:size])
    {
        // 在 GPU 上处理数据
        #pragma acc parallel//似乎同时要在
        process_data_on_gpu(size, data);
        //  #pragma acc update device(data[0:size])

    } // 在退出并行区域之后，使用 data copy 将数据从 GPU 同步回主机
    // #pragma acc exit data copyout(data[0:size])

    // 数据已经在 GPU 上处理，可以在主机上使用了
    // 此时 data 数组包含经过处理的结果
    for(int i=0;i<size;i++){
        if(i%10==0){
            cout<<endl;
        }
        cout<<data[i]<<",";
    }

    return 0;
}
