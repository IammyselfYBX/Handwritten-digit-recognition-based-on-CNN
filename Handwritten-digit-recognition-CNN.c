/*****************************************************************************
File name: Handwritten-digit-recognition
Description: CNN手写数字识别
Author: 贾继伟
Version: 1.1
Date: 1999年13月32日 
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>  
#define max(a,b)(((a)>(b))?(a):(b))
#define SAMPLE_NUM 30//宏定义的样本数量
double lr;
double result[11];//最后通过softmax输出的结果

/* 定义卷积核和全连接层的参数
 * 3x3的卷积核，两个通道，每个通道有三个卷积层
 * 三个全连接层，
 *  输入维度分别为1152、180、45，
 *  输出维度分别为180、45、10
 */ 
struct parameter{
    double conv_kernel11[3][3]; // 第一个通道的第一个卷积核
    double conv_kernel12[3][3]; // 第一个通道的第二个卷积核
    double conv_kernel21[3][3]; // 第二个通道的第一个卷积核
    double conv_kernel22[3][3]; // 第二个通道的第二个卷积核
    double conv_kernel31[3][3]; // 第三个通道的第一个卷积核
    double conv_kernel32[3][3]; // 第三个通道的第二个卷积核
    double fc_hidden_layer1[1152][180]; // 第一个全连接层，输入维度为1152，输出维度为180
    double fc_hidden_layer2[180][45];   // 第二个全连接层,输入维度为180，输出维度为45
    double fc_hidden_layer3[45][10];    // 第三个全连接层，输入维度为45，输出维度为10
};

/* 存储卷积神经网络（CNN）在前向传播过程中的中间结果和最终结果
 * 存储从输入到输出的所有中间结果和最终结果
 * 网络中每一层的尺寸
 */ 
struct result{
    double mnist_data[30][30]; // 输入数据
    //通道一
    double first_conv1[28][28];   // 第一层卷积层的输出
    double sencond_conv1[26][26]; // 第二层卷积层的输出
    double third_conv1[24][24];   // 第三层卷积层的输出
    //通道二
    double first_conv2[28][28];   // 第一层卷积层的输出
    double sencond_conv2[26][26]; // 第二层卷积层的输出
    double third_conv2[24][24];   // 第三层卷积层的输出
    //全连接
    double flatten_conv[1][1152]; // 扁平化后的特征图
    double first_fc[1][180];      // 第一个全连接层的输出
    double first_relu[1][180];    // 第一个全连接层的激活函数输出
    double second_fc[1][45];      // 第二个全连接层的输出
    double second_relu[1][45];    // 第二个全连接层的激活函数输出
    double outmlp[1][10];         // 全连接的输出
    double result[10];            // Softmax的输出
};

/* 用于存储训练集的结构体
 * 训练集结构体，训练样本30*30
 */
struct input{
    double a[10][SAMPLE_NUM][30][30];//[标签][样本数量][w][h]
};

/* 保存每张图片的数据和标签
 * 保存每一张图片的结构体
 */
struct sample{
    double a[30][30]; //data
    int number;       //label
}Sample[SAMPLE_NUM*10];

//以下函数的实现保持不变，因为它们是独立于平台的：
void Conv2d(int w,int h,int k,double *input_matrix,double *kernel,double *out_matrix){
    for(int i=0;i<w-k+1;i++)
        for(int j=0;j<h-k+1;j++){
            out_matrix[i*(w-k+1)+j]=0;
            for(int row=i;row<i+3;row++)
                for(int col=j;col<j+3;col++)
                    out_matrix[i*(w-k+1)+j]+=input_matrix[row*w+col]*kernel[(row-i)*k+(col-j)];
        }
}
//最大池化操作，池化核大小为k*k
void MaxPool2d(int w,int h,int k,double *input_matrix,double *output_matrix){
    for(int i=0;i<w/k;i++)
        for(int j=0;j<h/k;j++){
            int max_num=-999;
            for(int row=k*i;row<k*i+k;row++)
                for(int col=k*j;col<k*j+k;col++)
                    if(input_matrix[row*w+col]>max_num)
                        max_num=input_matrix[row*w+col];
            output_matrix[i*(w/2)+j]=max_num;
        }
}

//用LeakyRelu代替Relu，避免梯度弥散
void Relu(int w,int h,double *input_matrix,double *output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[i*w+j]=max(input_matrix[i*w+j],input_matrix[i*w+j]*0.05);
}

//特征图扁平化后concat
void MatrixExtensionImproved(int w,int h,double *input_matrix1,double *input_matrix2,double *output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[i*w+j]=input_matrix1[i*w+j];//将通道一的特征图输出加入到output_matrix

    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[w*h+i*w+j]=input_matrix2[i*w+j];//将通道二的特征图输出加入到output_matrix
}

//全连接的矩阵乘法
void MatrixMultiply(int w,int h,int out_deminsion,double *input_matrix,double *para_layer,double*output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<out_deminsion;j++){
            output_matrix[i*w+j]=0;
            for(int k=0;k<h;k++)
                output_matrix[i*w+j]+=input_matrix[i*w+k]*para_layer[k*out_deminsion+j];
        }
}

//将全连接反向传播过来的梯度拆成两部分输入到两个channel中
void MatrixSplit(double *input_matrix,double *splited_matrix1,double *splited_matrix2){
    for(int idx=0;idx<1152;idx++)
        if(idx<576)
            splited_matrix1[idx]=input_matrix[idx];
        else
            splited_matrix2[idx-576]=input_matrix[idx];
}

//更新网络参数
void MatrixBackPropagation(int w,int h,double *input_matrix,double *output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[i*h+j]-=lr*input_matrix[i*h+j];
}

//反向传播时的矩阵乘法
void MatrixBackPropagationMultiply(int w,int h,double *para,double *grad,double *rgrad){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            rgrad[i*h+j]=para[i]*grad[j];

}

/* 计算当前层的参数矩阵的梯度,
 * 利用前一层神经元梯度行矩阵乘本层神经元梯度列矩阵,
 * 得到本层参数梯度
 */ 
void CalculateMatrixGrad(int w,int h,double *input_matrix,double *grad,double *output_matrix){
    for(int i=0;i<w;i++){
        output_matrix[i]=0;//梯度清空，方便累加
        for(int j=0;j<h;j++){
            output_matrix[i]+=input_matrix[i*h+j]*grad[j];
        }
    }
}

/* 
 * 激活函数的反向传播
 */
void ReluBackPropagation(int w,double *input_matrix,double *grad,double *output_matrix){
    for(int i=0;i<w;i++)
        if(input_matrix[i]>0) output_matrix[i]=1*grad[i];
        else output_matrix[i]=0.05*grad[i];
}

/* 反向传播时对梯度进行填充，
 * 由w*h变为(w+2*stride)*(h+2*stride)
 */
void Padding(int w,int stride,double *input_matrix,double *output_matrix){
    for(int i=0;i<w+2*stride;i++)
        for(int j=0;j<w+2*stride;j++)
            output_matrix[i*(w+2*stride)+j]=0;//输出矩阵初始化
//    for(int i=0;i<w;i++)
//        for(int j=0;j<w;j++)
//            output_matrix[(i+stride)*(w+2*stride)+(j+stride)]=input_matrix[i*w+j];
}

/* 
 * 由于卷积核翻转180°后恰好是导数形式，故进行翻转后与后向传播过来的梯度相乘
 */
void OverturnKernel(int k,double *input_matrix,double *output_matrix){
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            output_matrix[(k-1-i)*k+(k-1-j)]=input_matrix[i*k+j];
}

//释放内存
void MemoryFree(double *x){
    free(x);
    x=NULL;
}

//使用随机数初始化网络参数
void init(struct parameter *para){
    srand(time(NULL));
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            para->conv_kernel11[i][j]=(rand()/(RAND_MAX+1.0));
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            para->conv_kernel12[i][j]=(rand()/(RAND_MAX+1.0));
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            para->conv_kernel21[i][j]=(rand()/(RAND_MAX+1.0))/5;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            para->conv_kernel22[i][j]=(rand()/(RAND_MAX+1.0))/5;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            para->conv_kernel31[i][j]=(rand()/(RAND_MAX+1.0))/5;
    for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
            para->conv_kernel32[i][j]=(rand()/(RAND_MAX+1.0))/5;
    for(int i=0;i<1152;i++)
        for(int j=0;j<180;j++)
            para->fc_hidden_layer1[i][j]=(rand()/(RAND_MAX+1.0))/1000;
    for(int i=0;i<180;i++)
        for(int j=0;j<45;j++)
            para->fc_hidden_layer2[i][j]=(rand()/(RAND_MAX+1.0))/100;
    for(int i=0;i<45;i++)
        for(int j=0;j<10;j++)
            para->fc_hidden_layer3[i][j]=(rand()/(RAND_MAX+1.0))/10;
}

//前向传播，包括三层卷积，三层全连接
void forward(double *input_matrix,struct parameter* para,struct result* data){
    Conv2d(30,30,3,input_matrix,&para->conv_kernel11[0][0],&data->first_conv1[0][0]);
    Conv2d(28,28,3,&data->first_conv1[0][0],&para->conv_kernel21[0][0],&data->sencond_conv1[0][0]);
    Conv2d(26,26,3,&data->sencond_conv1[0][0],&para->conv_kernel31[0][0],&data->third_conv1[0][0]);//第一个通道得到24*24的特征图

    Conv2d(30,30,3,input_matrix,&para->conv_kernel12[0][0],&data->first_conv2[0][0]);
    Conv2d(28,28,3,&data->first_conv2[0][0],&para->conv_kernel22[0][0],&data->sencond_conv2[0][0]);
    Conv2d(26,26,3,&data->sencond_conv2[0][0],&para->conv_kernel32[0][0],&data->third_conv2[0][0]);//第二个通道得到24*24的特征图

    MatrixExtensionImproved(24,24,&data->third_conv1[0][0],&data->third_conv2[0][0],&data->flatten_conv[0][0]);
    MatrixMultiply(1,1152,180,&data->flatten_conv[0][0],&para->fc_hidden_layer1[0][0],&data->first_fc[0][0]);
    Relu(1,180,&data->first_fc[0][0],&data->first_relu[0][0]);
    MatrixMultiply(1,180,45,&data->first_relu[0][0],&para->fc_hidden_layer2[0][0],&data->second_fc[0][0]);
    Relu(1,45,&data->second_fc[0][0],&data->second_relu[0][0]);
    MatrixMultiply(1,45,10,&data->second_relu[0][0],&para->fc_hidden_layer3[0][0],&data->outmlp[0][0]);

    double probability;
    for(int i=0;i<10;i++)
        probability+=exp(data->outmlp[0][i]);
    for(int i=0;i<10;i++){
        data->result[i]=exp(data->outmlp[0][i])/probability;
        result[i]=data->result[i];
    }
    return;
}

//反向传播，更新梯度
void backward(int label,struct parameter* para,struct result* data){
    /****************************************************************************************
     * grad结尾的变量代表每一层的梯度
     * wgrad结尾的变量代表每一层的参数的梯度
     * rgrad结尾的代表激活函数的梯度
     * 本网络结构是两个通道的卷积加三层全连接，每个通道有三层卷积层，无池化层，层数使用序数词标明
    ****************************************************************************************/
    int double_len=sizeof(double);
    double *out_grad;
    out_grad=(double*)malloc(10*double_len);//网络的输出是10个double类型
    //交叉熵损失函数求导结果为y_hat_i-y_i
    for(int i=0;i<10;i++)
        if(i==label) out_grad[i]=data->result[i]-1;
        else out_grad[i]=data->result[i]-0;
    //三层全连接层的反向传播
    double *out_wgrad;
    out_wgrad=(double*)malloc(450*double_len);
    MatrixBackPropagationMultiply(45,10,&data->second_relu[0][0],out_grad,out_wgrad);
    double *second_rgrad;
    second_rgrad=(double*)malloc(45*double_len);
    CalculateMatrixGrad(45,10,&para->fc_hidden_layer3[0][0],out_grad,second_rgrad);
    MemoryFree(out_grad);
    double *second_grad;
    second_grad=(double*)malloc(180*double_len);
    ReluBackPropagation(45,&data->second_fc[0][0],second_rgrad,second_grad);
    MemoryFree(second_rgrad);
    double *second_wgrad;
    second_wgrad=(double*)malloc(8100*double_len);
    MatrixBackPropagationMultiply(180,45,&data->first_relu[0][0],second_grad,second_wgrad);
    double *first_rgrad;
    first_rgrad=(double*)malloc(180*double_len);
    CalculateMatrixGrad(180,45,&para->fc_hidden_layer2[0][0],second_grad,first_rgrad);
    MemoryFree(second_grad);
    double *first_grad;
    first_grad=(double*)malloc(180*double_len);
    ReluBackPropagation(180,&data->first_fc[0][0],first_rgrad,first_grad);
    MemoryFree(first_rgrad);
    double *first_wgrad;
    first_wgrad=(double*)malloc(207360*double_len);
    MatrixBackPropagationMultiply(1152,180,&data->flatten_conv[0][0],first_grad,first_wgrad);
    double *all_conv_grad;
    all_conv_grad=(double*)malloc(1152*double_len);
    CalculateMatrixGrad(1152,180,&para->fc_hidden_layer1[0][0],first_grad,all_conv_grad);
    MemoryFree(first_grad);
    //通道一
    double *third_conv_grad1;
    third_conv_grad1=(double*)malloc(576*double_len);
    double *third_conv_grad2;
    third_conv_grad2=(double*)malloc(576*double_len);
    MatrixSplit(all_conv_grad,third_conv_grad1,third_conv_grad2);
    MemoryFree(all_conv_grad);
    double *third_kernel_grad;
    third_kernel_grad=(double*)malloc(9*double_len);
    Conv2d(26,26,24,&data->sencond_conv1[0][0],third_conv_grad1,third_kernel_grad);
    double *second_conv_grad1;
    second_conv_grad1=(double*)malloc(676*double_len);
    double *third_kernel_overturn;
    third_kernel_overturn=(double*)malloc(9*double_len);
    OverturnKernel(3,&para->conv_kernel31[0][0],third_kernel_overturn);
    double *third_conv_grad_padding1;
    third_conv_grad_padding1=(double*)malloc(784*double_len);
    Padding(26,1,third_conv_grad1,third_conv_grad_padding1);
    MemoryFree(third_conv_grad1);
    Conv2d(28,28,3,third_conv_grad_padding1,third_kernel_overturn,second_conv_grad1);
    MemoryFree(third_kernel_overturn);
    MemoryFree(third_conv_grad_padding1);
    double *second_kernel_grad;
    second_kernel_grad=(double*)malloc(9*double_len);
    Conv2d(28,28,26,&data->first_conv1[0][0],second_conv_grad1,second_kernel_grad);
    double *first_conv_grad;
    first_conv_grad=(double*)malloc(784*double_len);
    double *second_kernel_overturn;
    second_kernel_overturn=(double*)malloc(9*double_len);
    OverturnKernel(3,&para->conv_kernel21[0][0],second_kernel_overturn);
    double *second_conv_grad_padding1;
    second_conv_grad_padding1=(double*)malloc(900*double_len);
    Padding(28,1,second_conv_grad1,second_conv_grad_padding1);
    MemoryFree(second_conv_grad1);
    Conv2d(30,30,3,second_conv_grad_padding1,second_kernel_overturn,first_conv_grad);
    MemoryFree(second_kernel_overturn);
    MemoryFree(second_conv_grad_padding1);
    double *first_kernel_grad;
    first_kernel_grad=(double*)malloc(9*double_len);
    Conv2d(30,30,28,&data->mnist_data[0][0],first_conv_grad,first_kernel_grad);
    MemoryFree(first_conv_grad);
    //通道二
    double *third_kernel_grad2;
    third_kernel_grad2=(double*)malloc(9*double_len);
    Conv2d(26,26,24,&data->sencond_conv2[0][0],third_conv_grad2,third_kernel_grad2);
    double *second_conv_grad2;
    second_conv_grad2=(double*)malloc(676*double_len);
    double *third_kernel_overturn2;
    third_kernel_overturn2=(double*)malloc(9*double_len);
    OverturnKernel(3,&para->conv_kernel32[0][0],third_kernel_overturn2);
    double *third_conv_grad_padding2;
    third_conv_grad_padding2=(double*)malloc(784*double_len);
    Padding(26,1,third_conv_grad2,third_conv_grad_padding2);
    MemoryFree(third_conv_grad2);
    Conv2d(28,28,3,third_conv_grad_padding2,third_kernel_overturn2,second_conv_grad2);
    MemoryFree(third_conv_grad_padding2);
    double *second_kernel_grad2;
    second_kernel_grad2=(double*)malloc(9*double_len);
    Conv2d(28,28,26,&data->first_conv2[0][0],second_conv_grad2,second_kernel_grad2);
    double *first_conv_grad2;
    first_conv_grad2=(double*)malloc(784*double_len);
    double *second_kernel_overturn2;
    second_kernel_overturn2=(double*)malloc(9*double_len);
    OverturnKernel(3,&para->conv_kernel22[0][0],second_kernel_overturn2);
    double *second_conv_grad_padding2;
    second_conv_grad_padding2=(double*)malloc(900*double_len);
    Padding(28,1,second_conv_grad2,second_conv_grad_padding2);
    MemoryFree(second_conv_grad2);
    Conv2d(30,30,3,second_conv_grad_padding2,second_kernel_overturn2,first_conv_grad2);
    MemoryFree(second_kernel_overturn2);
    MemoryFree(second_conv_grad_padding2);
    double *first_kernel_grad2;
    first_kernel_grad2=(double*)malloc(9*double_len);
    Conv2d(30,30,28,&data->mnist_data[0][0],first_conv_grad2,first_kernel_grad2);


    //通道一更新参数
    MatrixBackPropagation(3,3,first_kernel_grad,&para->conv_kernel11[0][0]);
    MatrixBackPropagation(3,3,second_kernel_grad,&para->conv_kernel21[0][0]);
    MatrixBackPropagation(3,3,third_kernel_grad,&para->conv_kernel31[0][0]);
    //通道二更新参数
    MatrixBackPropagation(3,3,first_kernel_grad2,&para->conv_kernel12[0][0]);
    MatrixBackPropagation(3,3,second_kernel_grad2,&para->conv_kernel22[0][0]);
    MatrixBackPropagation(3,3,third_kernel_grad2,&para->conv_kernel32[0][0]);
    //全连接层更新参数
    MatrixBackPropagation(1152,180,first_wgrad,&para->fc_hidden_layer1[0][0]);
    MatrixBackPropagation(180,45,second_wgrad,&para->fc_hidden_layer2[0][0]);
    MatrixBackPropagation(45,10,out_wgrad,&para->fc_hidden_layer3[0][0]);
    //清空内存
    MemoryFree(first_kernel_grad);
    MemoryFree(second_kernel_grad);
    MemoryFree(third_kernel_grad);
    MemoryFree(first_kernel_grad2);
    MemoryFree(second_kernel_grad2);
    MemoryFree(third_kernel_grad2);
    MemoryFree(first_wgrad);
    MemoryFree(second_wgrad);
    MemoryFree(out_wgrad);
    return;
}

//从图片中提取数据
int DataLoader() {
    for (int num = 0; num < 10; num++) {
        for (int i = 0; i < SAMPLE_NUM; i++) {
            char *e = (char *)malloc(sizeof(char) * 120);
            int *l = (int *)malloc(sizeof(int) * 960);
            if (e == NULL || l == NULL) {
                printf("内存分配失败！\n");
                free(e); // 释放已分配的内存
                free(l);
                return 1; // 提前退出
            }

            char route_name[50] = "Training_set/";
            char file_name[15];
            sprintf(file_name, "%d/%d.bmp", num, i + 1);
            strcat(route_name, file_name);

            FILE *fp = fopen(route_name, "rb");
            if (fp == NULL) {
                printf("未能打开训练集数据. 检查'Training_set'文件夹是否存在，并且文件夹中有图片!\n");
                free(e);
                free(l);
                return 1;
            }
            fseek(fp, 62, SEEK_SET);
            fread(e, sizeof(char), 120, fp);
            fclose(fp);

            int y = 0;
            for (int r = 0; r < 120; r++) {
                for (int u = 1; u < 9; u++) {
                    l[y] = (int)((e[r] >> (8 - u)) & 0x01);
                    y++;
                    if (y > 960)
                        break;
                }
            }

            int g = 0;
            for (int u = 0; u < 30; u++) {
                y = 0;
                for (int j = 0; j < 32; j++) {
                    if (j != 30 && j != 31) {
                        Sample[num * SAMPLE_NUM + i].a[u][y] = l[g];
                        y++;
                    }
                    g++;
                }
            }

            int q = Sample[num * SAMPLE_NUM + i].a[0][0];
            if (q == 1) {
                for (int b = 0; b < 30; b++) {
                    for (int n = 0; n < 30; n++) {
                        Sample[num * SAMPLE_NUM + i].a[b][n] = 1 - Sample[num * SAMPLE_NUM + i].a[b][n];
                    }
                }
            }

            Sample[num * SAMPLE_NUM + i].number = num;
            free(e);
            free(l);
        }
    }
    return 0;
}

//训练前读取网络参数
int read_file(struct parameter* parameter4){
    FILE*fp;
    fp=fopen("Training_set/Network_parameter.bin","rb");
    if(fp==NULL)
    {
        printf("文件打开失败，请检查网络参数文件是否在训练集文件夹内！\n");
        return 1;
    }
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    fread(parameter1,sizeof(struct parameter),1,fp);
    (*parameter4)=(*parameter1);
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return 0;
}

//训练结束后保存网络参数
void printf_file(struct parameter* parameter4){
    FILE*fp;
    fp=fopen("Training_set//Network_parameter.bin","wb");//采用二进制格式保存参数，便于读取
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    (*parameter1)=(*parameter4);
    fwrite(parameter1,sizeof(struct parameter),1,fp);//打印网络结构体
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return;
}

//训练过程中的最优参数打印函数
void printf_file2(struct parameter* parameter4){
    FILE*fp;
    fp=fopen("NetworkParameters.bin","wb");
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    (*parameter1)=(*parameter4);
    fwrite(parameter1,sizeof(struct parameter),1,fp);//结果指针、大小、数量、文件指针
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return;
};

//交叉熵损失函数
double Cross_entropy(double *a,int m){
    double u=0;
    u=(-log10(a[m]));
    return u;
}

/* 函数名：train
 * 功能:网络训练部分，读取到图像数据进行前向传播的训练
 * 参数：epochs-训练次数，para-网络参数，data-中间结果
 * 返回值：无
 */
void train(int epochs,struct parameter *para,struct result *data){
    printf("\t进入train函数\n");
    double corss_loss=2;//保存每次训练的最大交叉熵
    for(int epoch=0;epoch<epochs;epoch++){
        lr=pow((corss_loss/10),1.7);
        if(lr>0.01) lr=0.01;
        if((epoch+1)%10==0){
            printf("训练进度: %lf",100*((double)(epoch+1)/(double)epochs));
            printf("%%  交叉熵损失: %lf  学习率:%.10lf\n",corss_loss,lr);
            if(lr<0.0000000001)printf_file2(para);//如果找到局部最优则打印网络参数
        }

        int a,b;
        srand(time(NULL));
        for(int q=0;q<300;q++){
            a=(int)((rand()/(RAND_MAX+1.0))*300);//确定本轮随机交换的变量下标
            b=(int)((rand()/(RAND_MAX+1.0))*300);
            if(a>=0&&a<300&&(a!=b)&&b>=0&&b<300){
                struct sample* sample5;
                sample5=(struct sample *)malloc(sizeof(struct sample));
                (*sample5)=Sample[a];
                Sample[a]=Sample[b];
                Sample[b]=(*sample5);
                free(sample5);
                sample5=NULL;
            }
            else continue;
        }
        for(int i=0;i<SAMPLE_NUM*10;i++) {//训练已经打乱的所有样本
            corss_loss=0;
            struct sample* sample3;
            sample3=(struct sample *)malloc(sizeof(struct sample));
            (*sample3)=Sample[i];
            int y=sample3->number;
            forward(&sample3->a[0][0],para,data);//正向传播
            backward(y,para,data);
            free(sample3);
            sample3=NULL;
            double g=Cross_entropy(&data->result[0],y);//计算本轮最大交叉熵损失，用于指导调整学习率
            if(g>corss_loss)corss_loss=g;
        }
    }
    printf("\n");
    return;
}

//用测试集中的样本测试网络，一共有十个测试样本
void test_network(struct parameter* parameter2,struct result *data2){
    char e[120];
    int l[960];
    double data[30][30];
    for(int i=0;i<10;i++){
        FILE *fp;
        char s[30];
        sprintf(s,"%s%d%s","Training_set/Test_set/",i,".bmp");
        printf("\n打开的文件名:%s\n",s);
        fp = fopen(s, "rb");
        if(fp == NULL){
            printf("不能打开文件!\n");
            system("pause");
            return;
        }
        fseek(fp, 62, SEEK_SET);
        fread(e,sizeof(char),120,fp);
        fclose(fp);
        int y=0;
        for(int r=0;r<120;r++){
          for (int u=1;u<9;u++){
            l[y]=(int)((e[r]) >> (8-u) & 0x01);
            y++;
            if(y>960)break;
          };
        };
        y=0;
        int g=0;
        for(int u=0;u<30;u++)
        {
            y=0;
            for(int j=0;j<32;j++)
            {
                if((j!=30)&&(j!=31)){data[u][y]=l[g];y++;};
                g++;
            }
        }
        int q=data[0][0];
        if(q==1){
            int n=0;
            int z=0;
            for(int b=0;b<30;b++)
            {
                n=0;
                for(;;)
                {
                    if(n>=30)break;
                    if(data[z][n]==0)data[z][n]=1;
                    else if(data[z][n]==1)data[z][n]=0;
                    n++;
                }
                z++;
            }
        }
        forward(&data[0][0],parameter2,data2);//把获取的样本数据正向传播一次
        double sum=0;
        int k=0;
        for(int j=0;j<10;j++)
            {
                if(result[j]>sum)
                {
                    sum=result[j];
                    k=j;//获取分类结果
                }
                else continue;
            }
        printf("\n");
        for(int i=0;i<10;i++)//打印分类结果
        {
            printf("预测值是%d的概率：%lf\n",i,result[i]);
        }
        printf("最终预测值:%d\n",k);
}
return ;
}

int main(){
    // 读取训练集
    int h=DataLoader();
    if(h==0)printf("训练数据读取成功\n");
    else if(h==1){
        printf("训练集读取失败！程序自动退出\n");
        return 0;
    }

    // 开始训练网络
    printf("开始训练网络\n");
    struct parameter *storage;//定义存放网络参数的结构体
    (storage) = (struct parameter*)malloc(sizeof(struct parameter));//动态分配空间
    struct result *data;
    (data) = (struct result*)malloc(sizeof(struct result));

    // 读取网络参数
    char g;
    do {
        printf("请问您是否希望从已训练的网络参数文件中读取网络参数？(是请按y，否请按n): ");
        setbuf(stdin,NULL);//清空键盘缓冲区
        g=getchar();
        while(getchar() != '\n'); // 清空输入缓冲区
        if(g=='y'){
            int h=read_file(storage);
            if(h==1){
                printf("参数包不存在！开始自动随机初始化网络参数\n");
                init(storage);
                printf_file(storage);
                printf("网络参数初始化完毕！\n");
                printf("网络参数已保存到 Network_parameter.bin 文件中\n");
            }
            if(h==0)
                printf("参数读取成功!\n");
        }
        else if (g=='n'){
            init(storage);
            printf_file(storage);
            printf("参数初始化完毕！\n");
        }
    } while (g != 'y' && g != 'n');

    int epoch;
    char v;
    do {
        printf("请输入预训练的次数：");
        scanf("%d",&epoch);
        printf("开始训练\n");
        train(epoch,storage,data);
        test_network(storage,data);
        printf_file(storage);

        do {
            printf("继续训练请按回车，退出请按q: ");
            setbuf(stdin,NULL);
            v=getchar();
            while(getchar() != '\n'); // 清空输入缓冲区
            if(v=='q'){
                printf_file(storage);
                return 0;//退出则在退出之前保存网络参数
            }else if (v=='\n') {
                break;
            }else {
                printf("输入错误，请重新输入！\n");
            }
        } while (1);
    } while (v != 'q');

    return 0;
}

