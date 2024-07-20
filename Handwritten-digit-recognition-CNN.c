/*****************************************************************************
File name: Handwritten-digit-recognition
Description: CNN手写数字识别
Author: XXX
Version: 3.1
Date: 1999年13月32日 
*****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h>  
#define max(a,b)(((a)>(b))?(a):(b))
#define SAMPLE_NUM 30
double lr;
double result[11];
#define BOOL int
#define TRUE 1
#define FALSE 0

struct parameter{
    double conv_kernel11[3][3]; 
    double conv_kernel21[3][3]; 
    double conv_kernel31[3][3]; 
    double conv_kernel12[3][3]; 
    double conv_kernel22[3][3]; 
    double conv_kernel32[3][3]; 
    double fc_hidden_layer1[1152][180]; 
    double fc_hidden_layer2[180][45];   
    double fc_hidden_layer3[45][10];    
};

struct result{
    double mnist_data[30][30]; 
    double first_conv1[28][28];   
    double sencond_conv1[26][26]; 
    double third_conv1[24][24];   
    double first_conv2[28][28];   
    double sencond_conv2[26][26]; 
    double third_conv2[24][24];   
    double flatten_conv[1][1152]; 
    double first_fc[1][180];      
    double first_relu[1][180];    
    double second_fc[1][45];      
    double second_relu[1][45];    
    double outmlp[1][10];         
    double result[10];            
};

struct sample{
    double a[30][30]; 
    int number;       
}Sample[SAMPLE_NUM*10];

void Conv2d(int w,int h,int k,double *input_matrix,double *kernel,double *out_matrix){
    for(int i=0;i<w-k+1;i++)        
        for(int j=0;j<h-k+1;j++){   
            out_matrix[i*(w-k+1)+j]=0; 
            for(int row=i;row<i+3;row++)
                for(int col=j;col<j+3;col++)
                    out_matrix[i*(w-k+1)+j]+=input_matrix[row*w+col]*kernel[(row-i)*k+(col-j)];
        }
}

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

void Relu(int w,int h,double *input_matrix,double *output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[i*w+j]=max(input_matrix[i*w+j],input_matrix[i*w+j]*0.05);
}

void MatrixExtensionImproved(int w,int h,double *input_matrix1,double *input_matrix2,double *output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[i*w+j]=input_matrix1[i*w+j];

    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[w*h+i*w+j]=input_matrix2[i*w+j];
}

void MatrixMultiply(int w,int h,int out_deminsion,double *input_matrix,double *para_layer,double*output_matrix){
    for(int i=0;i<w;i++) 
        for(int j=0;j<out_deminsion;j++){ 
            output_matrix[i*w+j]=0; 
            for(int k=0;k<h;k++)
                output_matrix[i*w+j]+=input_matrix[i*w+k]*para_layer[k*out_deminsion+j];
        }
}

void MatrixSplit(double *input_matrix,double *splited_matrix1,double *splited_matrix2){
    for(int idx=0;idx<1152;idx++)
        if(idx<576)
            splited_matrix1[idx]=input_matrix[idx];
        else
            splited_matrix2[idx-576]=input_matrix[idx];
}

void MatrixBackPropagation(int w,int h,double *input_matrix,double *output_matrix){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            output_matrix[i*h+j]-=lr*input_matrix[i*h+j];
}

void MatrixBackPropagationMultiply(int w,int h,double *para,double *grad,double *rgrad){
    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            rgrad[i*h+j]=para[i]*grad[j];

}

void CalculateMatrixGrad(int w,int h,double *input_matrix,double *grad,double *output_matrix){
    for(int i=0;i<w;i++){
        output_matrix[i]=0;//梯度清空，方便累加
        for(int j=0;j<h;j++){
            output_matrix[i]+=input_matrix[i*h+j]*grad[j];
        }
    }
}

void ReluBackPropagation(int w,double *input_matrix,double *grad,double *output_matrix){
    for(int i=0;i<w;i++)
        if(input_matrix[i]>0) output_matrix[i]=1*grad[i];
        else output_matrix[i]=0.05*grad[i];
}

void Padding(int w,int stride,double *input_matrix,double *output_matrix){
    for(int i=0;i<w+2*stride;i++)
        for(int j=0;j<w+2*stride;j++)
            output_matrix[i*(w+2*stride)+j]=0;//输出矩阵初始化
}

void OverturnKernel(int k,double *input_matrix,double *output_matrix){
    for(int i=0;i<k;i++)
        for(int j=0;j<k;j++)
            output_matrix[(k-1-i)*k+(k-1-j)]=input_matrix[i*k+j];
}

void MemoryFree(double *x){
    free(x);
    x=NULL;
}
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
void forward(double *input_matrix,struct parameter* para,struct result* data){
    Conv2d(30,30,3,input_matrix,&para->conv_kernel11[0][0],&data->first_conv1[0][0]);
    Conv2d(28,28,3,&data->first_conv1[0][0],&para->conv_kernel21[0][0],&data->sencond_conv1[0][0]);
    Conv2d(26,26,3,&data->sencond_conv1[0][0],&para->conv_kernel31[0][0],&data->third_conv1[0][0]);//第一个通道得到24*24的特征图
    Conv2d(30,30,3,input_matrix,&para->conv_kernel12[0][0],&data->first_conv2[0][0]);
    Conv2d(28,28,3,&data->first_conv2[0][0],&para->conv_kernel22[0][0],&data->sencond_conv2[0][0]);
    Conv2d(26,26,3,&data->sencond_conv2[0][0],&para->conv_kernel32[0][0],&data->third_conv2[0][0]);//第二个通道得到24*24的特征图
    MatrixExtensionImproved(24,24,&data->third_conv1[0][0],&data->third_conv2[0][0],&data->flatten_conv[0][0]); // 扁平化
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
}

void backward(int label,struct parameter* para,struct result* data){
    int double_len=sizeof(double);
    double *out_grad;
    out_grad=(double*)malloc(10*double_len);
    for(int i=0;i<10;i++)
        if(i==label) out_grad[i]=data->result[i]-1;
        else out_grad[i]=data->result[i]-0;
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
    MatrixBackPropagation(3,3,first_kernel_grad,&para->conv_kernel11[0][0]);
    MatrixBackPropagation(3,3,second_kernel_grad,&para->conv_kernel21[0][0]);
    MatrixBackPropagation(3,3,third_kernel_grad,&para->conv_kernel31[0][0]);
    MatrixBackPropagation(3,3,first_kernel_grad2,&para->conv_kernel12[0][0]);
    MatrixBackPropagation(3,3,second_kernel_grad2,&para->conv_kernel22[0][0]);
    MatrixBackPropagation(3,3,third_kernel_grad2,&para->conv_kernel32[0][0]);
    MatrixBackPropagation(1152,180,first_wgrad,&para->fc_hidden_layer1[0][0]);
    MatrixBackPropagation(180,45,second_wgrad,&para->fc_hidden_layer2[0][0]);
    MatrixBackPropagation(45,10,out_wgrad,&para->fc_hidden_layer3[0][0]);
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

BOOL DataLoader() {
    for (int num = 0; num < 10; num++) { 
        for (int i = 0; i < SAMPLE_NUM; i++) { 
            char *e = (char *)malloc(sizeof(char) * 120);    

            int *l = (int *)malloc(sizeof(int) * 960);      
            if (e == NULL || l == NULL) {
                perror("内存分配失败！\n");
                free(e);        
                free(l);
                return FALSE;
            }

            char route_name[50] = "Training_set/";
            char file_name[15];
            sprintf(file_name, "%d/%d.bmp", num, i + 1);
            strcat(route_name, file_name);

            FILE *fp = fopen(route_name, "rb");
            if (!fp) {
                perror("未能打开训练集数据. 检查'Training_set'文件夹是否存在，并且文件夹中有图片!\n");
                free(e);
                free(l);
                return FALSE;
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
    return TRUE;
}

BOOL read_file(struct parameter* parameter_dest){
    FILE*fp;
    fp=fopen("network_parameter.txt","rb");
    if(fp==NULL)
    {
        printf("文件打开失败，请检查网络参数文件是否在训练集文件夹内！\n");
        return FALSE;
    }
    struct parameter* parameter_tmp=NULL;
    parameter_tmp=(struct parameter*)malloc(sizeof(struct parameter));
    fread(parameter_tmp,sizeof(struct parameter),1,fp);
    (*parameter_dest)=(*parameter_tmp);
    fclose(fp);
    free(parameter_tmp);
    parameter_tmp=NULL;

    return TRUE;
}

BOOL write_para_to_file(struct parameter* parameter_file){
    FILE*fp;
    fp=fopen("network_parameter.txt","wb");
    struct parameter* parameter_tmp;
    parameter_tmp=(struct parameter*)malloc(sizeof(struct parameter));

    (*parameter_tmp)=(*parameter_file);
    fwrite(parameter_tmp,sizeof(struct parameter),1,fp);

    fclose(fp);
    free(parameter_tmp);
    parameter_tmp=NULL;
    
    return TRUE;
}

void printf_file2(struct parameter* parameter4){
    FILE*fp;
    fp=fopen("NetworkParameters.bin","wb");
    struct parameter* parameter1;
    parameter1=(struct parameter*)malloc(sizeof(struct parameter));
    (*parameter1)=(*parameter4);
    fwrite(parameter1,sizeof(struct parameter),1,fp);
    fclose(fp);
    free(parameter1);
    parameter1=NULL;
    return;
};

double Cross_entropy(double *a,int m){
    double u=0;
    u=(-log10(a[m]));
    return u;
}

void show_progress_bar(int progress, int total) {
    int bar_width = 50;
    float percent_complete = (float)progress / total;
    int position = bar_width * percent_complete;

    printf("[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < position) {
            printf("=");
        } else if (i == position) {
            printf(">");
        } else {
            printf(" ");
        }
    }
    printf("] %d%%\r", (int)(percent_complete * 100));
    fflush(stdout);
}

void train(int epochs,struct parameter *para,struct result *data){
    double corss_loss=2; 
    for(int epoch=0;epoch<epochs;epoch++){
        lr=pow((corss_loss/10),1.7);
        if(lr>0.01) {
            lr=0.01;
        }

        show_progress_bar(epoch+1, epochs);
        if((epoch+1)%10==0){
            fflush(stdout);
            printf("\t交叉熵损失: %lf  学习率:%.10lf\n",corss_loss,lr);
            if(lr<0.0000000001)printf_file2(para);
        }

        int a,b;
        srand(time(NULL));
        struct sample* sample_tmp = NULL; 
        sample_tmp=(struct sample *)malloc(sizeof(struct sample));
        for(int q=0;q<SAMPLE_NUM*10;q++){
            a=(int)((rand()/(RAND_MAX+1.0))*300);
            b=(int)((rand()/(RAND_MAX+1.0))*300);
            if(a>=0&&a<300&&(a!=b)&&b>=0&&b<300){
                (*sample_tmp)=Sample[a];
                Sample[a]=Sample[b];
                Sample[b]=(*sample_tmp);
            }
            else continue;
        }
        for(int i=0;i<SAMPLE_NUM*10;i++) {
            corss_loss=0;
            (*sample_tmp)=Sample[i];
            int y=sample_tmp->number;
            forward(&sample_tmp->a[0][0],para,data);
            backward(y,para,data);
            double g=Cross_entropy(&data->result[0],y);
            if(g>corss_loss)corss_loss=g;
        }
        free(sample_tmp);
        sample_tmp=NULL;
    }
    printf("\n");
    return;
}

void test_network(struct parameter* parameter2,struct result *data2){
    char e[120];
    int l[960];
    double data[30][30];
    for(int i=0;i<10;i++){
        FILE *fp;
        char s[30];
        sprintf(s,"%s%d%s","Test_set/",i,".bmp");
        printf("\n打开的文件名:%s",s);
        fp = fopen(s, "rb");
        if(!fp){
            perror("不能打开文件!\n");
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
        forward(&data[0][0],parameter2,data2);
        double sum=0;
        int k=0;
        for(int j=0;j<10;j++)
            {
                if(result[j]>sum)
                {
                    sum=result[j];
                    k=j;
                }
                else continue;
            }
        printf("\n");
        for(int i=0;i<10;i++)
        {
            printf("\t预测值是%d的概率：%lf\n",i,result[i]);
        }

        printf("\033[1;31;43m最终预测值: %d \033[0m\n", k);
}
return ;
}


int main(int argc, char const *argv[]){
    int h=DataLoader();
    if(h==TRUE){
        printf("训练数据读取成功！\n");
    }
    else if(h==FALSE){
        printf("训练集读取失败！程序自动退出\n");
        return 0;
    }

    printf("=============================================\n");
    printf("开始训练网络\n");
    struct parameter *storage;
    (storage) = (struct parameter*)malloc(sizeof(struct parameter));
    struct result *data;
    (data) = (struct result*)malloc(sizeof(struct result));

    char g;
    do {
        printf("请问您是否希望从已训练的网络参数文件中读取网络参数？(是请按 y，否请按 n): ");
        setbuf(stdin,NULL);
        g=getchar();
        while(getchar() != '\n'); 
        if(g=='y'){
            int h=read_file(storage);
            if(h==FALSE){
                printf("参数包不存在！开始自动随机初始化网络参数\n");
                init(storage);
                write_para_to_file(storage);
                printf("网络参数初始化完毕！\n");
                printf("网络参数已保存到 network_parameter.txt 文件中\n");
            }else if(h==TRUE){
                printf("参数读取成功!\n");
            }
        }else if (g=='n'){
            init(storage);
            write_para_to_file(storage);
            printf("参数初始化完毕！\n");
        }
    } while (g != 'y' && g != 'n');
    printf("=============================================\n");

    int epoch;
    char v;
    do {
        printf("请输入预训练的次数：");
        scanf("%d",&epoch);
        printf("开始训练\n");
        train(epoch,storage,data);
        printf("开始测试\n");
        test_network(storage,data);
        write_para_to_file(storage);

        do {
            printf("继续训练请按回车，退出请按q: ");
            setbuf(stdin,NULL);
            v=getchar();
            while(getchar() != '\n'); 
            if(v=='q'){
                write_para_to_file(storage);
                return 0;
            }else if (v=='\n') {
                break;
            }else {
                printf("输入错误，请重新输入！\n");
            }
        } while (1);
    } while (v != 'q');

    free(storage);
    free(data);

    return 0;
}