#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h> 

#define MAXN 1000000

//todo: read clusters from file
//todo: the choise for init clusters
//todo: the ending criteria

const int ThreadsPerBlock = 1024; // max value since CC2.0
int BlocksPerGridN = 0;
int BlocksPerGridK = 0;

int N; // number of points
int K; // number of clusters
int T; // number of iterations
char INPUT_FILE[256]; // input file name

float *POINTS; // POINTS[i*2+0]:x POINTS[i*2+1]:y
int *CLASSES; // class for each point
int *NUM_CLASSES; // number of points in each class
float *CLUSTERS; // position for each cluster
float *OLD_CLUSTERS; // position for each cluster

// size for each array
size_t S_POINTS;
size_t S_CLASSES;
size_t S_NUM_CLASSES;
size_t S_CLUSTERS;

// values on CUDA device
int USEGPU; // use gpu or cpu
int SYNC; // synchronize data between cpu and gpu after each iter

float *D_POINTS; // POINTS[i*2+0]:x POINTS[i*2+1]:y
int *D_CLASSES; // class for each point
int *D_NUM_CLASSES; // number of points in each class
float *D_CLUSTERS; // position for each cluster
float *D_OLD_CLUSTERS; // position for each cluster


void write_results(int n, int k){
    FILE *outputFile;
    int i;
    
    outputFile = fopen("Classes.txt", "w");
    for(i=0;i<n;i++){
        fprintf(outputFile, "%d\n", CLASSES[i]);
    }
    fclose(outputFile);
    
    outputFile = fopen("Clusters.txt", "w");
    for(i=0;i<k;i++){
        fprintf(outputFile, "%f,%f\n", CLUSTERS[i*2], CLUSTERS[i*2+1]);
    }
    fclose(outputFile);    
}


void update_classes(int n, int k){ //based on CLUSTERS
    int i,j,minK;
    float minDis, dis, disX, disY;
    for(i=0;i<n;i++){
        disX = POINTS[i*2]-CLUSTERS[0];
        disY = POINTS[i*2+1]-CLUSTERS[1];
        minK = 0;
        minDis = disX*disX + disY*disY;
        for(j=1;j<k;j++){
            disX = POINTS[i*2]-CLUSTERS[j*2];
            disY = POINTS[i*2+1]-CLUSTERS[j*2+1];
            dis = disX*disX + disY*disY;
            if(dis<minDis){
                minK = j;
                minDis = dis;
            }
        }
        CLASSES[i] = minK;
    }
}


__global__ void cuda_update_classes_kernel(const float *d_points,
                                           const float *d_clusters,
                                           int *d_classes, 
                                           int n, int k){
    int i,j,minK;
    float minDis, dis, disX, disY;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n){
		disX = d_points[i*2]-d_clusters[0];
		disY = d_points[i*2+1]-d_clusters[1];
		minK = 0;
		minDis = disX*disX + disY*disY;
		for(j=1;j<k;j++){
			disX = d_points[i*2]-d_clusters[j*2];
			disY = d_points[i*2+1]-d_clusters[j*2+1];
			dis = disX*disX + disY*disY;
			if(dis<minDis){
				minK = j;
				minDis = dis;
			}
		}
		d_classes[i] = minK;	
	}
}

void cuda_update_classes(int n, int k, int sync=1){ // based on CLUSTERS, sync: synchronize between host and device
	cudaError_t cuerr = cudaSuccess; // use with cudaGetErrorString(cuerr);
	int err;
    // copy data to device
    if(sync){
        err = 1;
        //err &= cudaMemcpy(D_POINTS, POINTS, S_POINTS, cudaMemcpyHostToDevice) == cudaSuccess;
        err &= cudaMemcpy(D_CLUSTERS, CLUSTERS, S_CLUSTERS, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from host to device\n");
            exit(EXIT_FAILURE);
        }
    }
    
    cuda_update_classes_kernel<<<BlocksPerGridN, ThreadsPerBlock>>>(D_POINTS, D_CLUSTERS, D_CLASSES, n, k);
	
	// copy result to host
    if(sync){
        err = 1;
        err &= (cuerr = cudaMemcpy(CLASSES, D_CLASSES, S_CLASSES, cudaMemcpyDeviceToHost)) == cudaSuccess;
        //printf("err code %s %d\n", cudaGetErrorString(cuerr), err);
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from device to host\n");
            exit(EXIT_FAILURE);
        }
    }
}

void count_classes(int n, int k){
    int i;
    for(i=0;i<k;i++){
        NUM_CLASSES[i]=0;
    }
    for(i=0;i<n;i++){
        NUM_CLASSES[CLASSES[i]]++;
    }   
}

__global__ void cuda_count_classes_kernel_clean(int *d_num_classes, 
                                                int k){
    int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < k){
        d_num_classes[i]=0;
	}
}


__global__ void cuda_count_classes_kernel_sum(const int *d_classes,
                                              int *d_num_classes,
                                              int n){
    int i;
    int _class;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n){
        _class = d_classes[i];
        atomicAdd(&d_num_classes[_class], 1);
        //d_num_classes[_class] += 1;
	}
}

void cuda_count_classes(int n, int k, int sync=1){
	cudaError_t cuerr = cudaSuccess; // use with cudaGetErrorString(cuerr);
	int err;
    
    // copy data to device
    if(sync){
        err = 1;
        //err &= cudaMemcpy(D_POINTS, POINTS, S_POINTS, cudaMemcpyHostToDevice) == cudaSuccess;
        //err &= cudaMemcpy(D_CLASSES, CLASSES, S_CLASSES, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from host to device\n");
            exit(EXIT_FAILURE);
        }
    }
    
    cuda_count_classes_kernel_clean<<<BlocksPerGridK, ThreadsPerBlock>>>(D_NUM_CLASSES, k);
    cuda_count_classes_kernel_sum<<<BlocksPerGridN, ThreadsPerBlock>>>(D_CLASSES, D_NUM_CLASSES, n);
	
	// copy result to host
    if(sync){
        err = 1;
        err &= (cuerr = cudaMemcpy(NUM_CLASSES, D_NUM_CLASSES, S_NUM_CLASSES, cudaMemcpyDeviceToHost)) == cudaSuccess;
        //printf("err code %s %d\n", cudaGetErrorString(cuerr), err);
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from device to host\n");
            exit(EXIT_FAILURE);
        }
    }
}
    
void update_clusters(int n, int k){ // based on CLASSES
    int i;
	int _class;
    // clean
    for(i=0;i<k;i++){
        CLUSTERS[i*2]=0;
        CLUSTERS[i*2+1]=0;
        NUM_CLASSES[i]=0;
    }
    // sum
    for(i=0;i<n;i++){
        _class = CLASSES[i];
        NUM_CLASSES[_class]++;
        CLUSTERS[_class*2] += POINTS[i*2];
        CLUSTERS[_class*2+1] += POINTS[i*2+1];
    }
    // divide
    for(i=0;i<k;i++){
        //if(NUM_CLASSES[i]!=0){
            CLUSTERS[i*2] /= NUM_CLASSES[i]; // produce nan when divided by 0
            CLUSTERS[i*2+1] /= NUM_CLASSES[i];
        //}
    }    
}


__global__ void cuda_update_clusters_kernel_clean(float *d_clusters, 
                                                  int *d_num_classes, 
                                                  int k){
    int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < k){
        d_clusters[i*2]=0;
        d_clusters[i*2+1]=0;
        d_num_classes[i]=0;
	}
}


__global__ void cuda_update_clusters_kernel_sum(const float *d_points,
                                                const int *d_classes,
                                                float *d_clusters,
                                                int *d_num_classes,
                                                int n){
    int i;
    int _class;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n){
        _class = d_classes[i];
        atomicAdd(&d_num_classes[_class], 1);
        //d_num_classes[_class] += 1;
        atomicAdd(&d_clusters[_class*2], d_points[i*2]);
        //d_clusters[_class*2] += d_points[i*2];
        atomicAdd(&d_clusters[_class*2+1], d_points[i*2+1]);
        //d_clusters[_class*2+1] += d_points[i*2+1];
	}
}


__global__ void cuda_update_clusters_kernel_sum_reduce(const float *d_points,
                                                       const int *d_classes,
                                                       float *d_clusters,
                                                       int *d_num_classes,
                                                       int n, int k){
    extern __shared__ int shared[];
	int *shared_num_classes = shared;
	float *shared_clusters_x = (float*)&shared_num_classes[blockDim.x*k];
	float *shared_clusters_y = (float*)&shared_clusters_x[blockDim.x*k];
	
	int i,tid;
    int _class, cluster, cluster_mem, stride;
	tid = threadIdx.x;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < n){
        _class = d_classes[i];
		for(cluster=0;cluster<k;cluster++){
			if(cluster==_class){
				shared_num_classes[cluster_mem]=1;
				shared_clusters_x[cluster_mem] = d_points[i*2];
				shared_clusters_y[cluster_mem] = d_points[i*2+1];
			}else{
				shared_num_classes[cluster_mem]=0;
				shared_clusters_x[cluster_mem] = 0;
				shared_clusters_y[cluster_mem] = 0;
			}
		}
		__syncthreads();
		
		for(cluster=0;cluster<k;cluster++){
			cluster_mem = cluster * blockDim.x + tid;
			for(stride=blockDim.x/2;stride>0;stride>>=1){
				if(tid<stride){
					shared_num_classes[cluster_mem]+=shared_num_classes[cluster_mem+stride];
					shared_clusters_x[cluster_mem]+=shared_clusters_x[cluster_mem+stride];
					shared_clusters_y[cluster_mem]+=shared_clusters_y[cluster_mem+stride];
				}
				__syncthreads();
			}
		}
		
		if(tid<k){
			d_num_classes[tid] += shared_num_classes[tid*blockDim.x];
			d_clusters[tid*2] += shared_clusters_x[tid*blockDim.x];
			d_clusters[tid*2+1]+=shared_clusters_y[tid*blockDim.x];
		}			
		
	}
	
}




__global__ void cuda_update_clusters_kernel_divide(float *d_clusters,
                                                   const int *d_num_classes, 
                                                   int k){
    int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < k){
        d_clusters[i*2] /= d_num_classes[i];
        d_clusters[i*2+1] /= d_num_classes[i];
	}
}

void cuda_update_clusters(int n, int k, int sync=1){ // based on CLUSTERS, sync: synchronize between host and device
	cudaError_t cuerr = cudaSuccess; // use with cudaGetErrorString(cuerr);
	int err;
    
    // copy data to device
    if(sync){
        err = 1;
        //err &= cudaMemcpy(D_POINTS, POINTS, S_POINTS, cudaMemcpyHostToDevice) == cudaSuccess;
        //err &= cudaMemcpy(D_CLASSES, CLASSES, S_CLASSES, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from host to device\n");
            exit(EXIT_FAILURE);
        }
    }

    cuda_update_clusters_kernel_clean<<<BlocksPerGridK, ThreadsPerBlock>>>(D_CLUSTERS, D_NUM_CLASSES, k);
    //cuda_update_clusters_kernel_sum<<<BlocksPerGridN, ThreadsPerBlock>>>(D_POINTS, D_CLASSES, D_CLUSTERS, D_NUM_CLASSES, n);
	cuda_update_clusters_kernel_sum_reduce<<<BlocksPerGridN, ThreadsPerBlock, sizeof(int)*ThreadsPerBlock*k+sizeof(float)*ThreadsPerBlock*k*2>>>(D_POINTS, D_CLASSES, D_CLUSTERS, D_NUM_CLASSES, n, k);
    cuda_update_clusters_kernel_divide<<<BlocksPerGridK, ThreadsPerBlock>>>(D_CLUSTERS, D_NUM_CLASSES, k);
	
	// copy result to host
    if(sync){
        err = 1;
        err &= (cuerr = cudaMemcpy(CLUSTERS, D_CLUSTERS, S_CLUSTERS, cudaMemcpyDeviceToHost)) == cudaSuccess;
        err &= (cuerr = cudaMemcpy(NUM_CLASSES, D_NUM_CLASSES, S_NUM_CLASSES, cudaMemcpyDeviceToHost)) == cudaSuccess;
        //printf("err code %s %d\n", cudaGetErrorString(cuerr), err);
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from device to host\n");
            exit(EXIT_FAILURE);
        }
    }
}
    
void clean_clusters_0(int n, int *K){ // remove empty clusters, CLASSES are invalid after this process
    int i = 0;
    while(i<*K){
        if(NUM_CLASSES[i]==0){
            CLUSTERS[i*2] = CLUSTERS[*K * 2];
            CLUSTERS[i*2+1] = CLUSTERS[*K * 2 + 1];
            NUM_CLASSES[i]= NUM_CLASSES[*K];
            (*K)--;
        }else{
            i++;
        }
    }
}


void clean_clusters(int n, int *K=NULL){ // use old positions for empty clusters 
    int i = 0;
    while(i<*K){
        if(NUM_CLASSES[i]==0){
            printf("cluster %d empty, use old value\n", i);
            CLUSTERS[i*2] = OLD_CLUSTERS[i * 2];
            CLUSTERS[i*2+1] = OLD_CLUSTERS[i * 2 + 1];
        }
        i++;
    }
    memcpy(OLD_CLUSTERS, CLUSTERS, S_CLUSTERS);
}

__global__ void cuda_clean_clusters_kernel(float *d_clusters, 
                                           float *d_old_clusters,
                                           int *d_num_classes,
                                           int k){
    int i;
	i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<k){
        if(d_num_classes[i]==0){
            d_clusters[i*2] = d_old_clusters[i*2];
            d_clusters[i*2+1] = d_old_clusters[i*2+1];
        }else{
            d_old_clusters[i*2] = d_clusters[i*2];
            d_old_clusters[i*2+1] = d_clusters[i*2+1];
        }
    }    
}

void cuda_clean_clusters(int n, int *K=NULL, int sync=1){ // use old positions for empty clusters 
	cudaError_t cuerr = cudaSuccess; // use with cudaGetErrorString(cuerr);
	int err;
    
    // copy data to device
    if(sync){
        err = 1;
        err &= cudaMemcpy(D_CLUSTERS, CLUSTERS, S_CLUSTERS, cudaMemcpyHostToDevice) == cudaSuccess;
        err &= cudaMemcpy(D_OLD_CLUSTERS, OLD_CLUSTERS, S_CLUSTERS, cudaMemcpyHostToDevice) == cudaSuccess;
        err &= cudaMemcpy(D_NUM_CLASSES, NUM_CLASSES, S_NUM_CLASSES, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from host to device\n");
            exit(EXIT_FAILURE);
        }
    }
    
    cuda_clean_clusters_kernel<<<BlocksPerGridK, ThreadsPerBlock>>>(D_CLUSTERS, D_OLD_CLUSTERS, D_NUM_CLASSES, *K);
	
	// copy result to host
    if(sync){
        err = 1;
        err &= (cuerr = cudaMemcpy(CLUSTERS, D_CLUSTERS, S_CLUSTERS, cudaMemcpyDeviceToHost)) == cudaSuccess;
        err &= (cuerr = cudaMemcpy(OLD_CLUSTERS, D_OLD_CLUSTERS, S_CLUSTERS, cudaMemcpyDeviceToHost)) == cudaSuccess;
        //printf("err code %s %d\n", cudaGetErrorString(cuerr), err);
        if (!err)
        {
            fprintf(stderr, "Failed to copy data from device to host\n");
            exit(EXIT_FAILURE);
        }
    }
}

void clean_clusters_2(int n, int *K=NULL){ // random choose from points
    int i=0,p;
    while(i<*K){
        if(NUM_CLASSES[i]==0){
            p = (rand()) % n;
            printf("cluster %d empty, replace with point %d\n", i, p);
            CLUSTERS[i*2] = POINTS[p * 2];
            CLUSTERS[i*2+1] = POINTS[p * 2 + 1];
        }
        i++;
    }
    memcpy(OLD_CLUSTERS, CLUSTERS, S_CLUSTERS);
}

void init(int n, int k, char *input, int updateClasses){ // malloc and read points (and clusters)
    FILE *inputFile;
    int i;
    float x,y;
    
    // read points
    S_POINTS = n * 2 * sizeof(float);
    POINTS = (float*)malloc(S_POINTS);
    
    inputFile = fopen(input, "r");
    for(i=0;i<n;i++){
        if(fscanf(inputFile, "%f,%f\n", &x, &y)==2){
            POINTS[i*2] = x;
            POINTS[i*2+1] = y;
        }
    }
    fclose(inputFile);
    
    // classes init
    S_CLASSES = n * sizeof(int);
    CLASSES = (int*)malloc(S_CLASSES);
    // clusters init
    S_NUM_CLASSES = k * sizeof(int);
    S_CLUSTERS = k * 2 * sizeof(float);
    NUM_CLASSES = (int*)malloc(S_NUM_CLASSES);
    CLUSTERS = (float*)malloc(S_CLUSTERS);
    OLD_CLUSTERS = (float*)malloc(S_CLUSTERS);
    for(i=0;i<k;i++){
        CLUSTERS[i*2]=POINTS[i*2];
        CLUSTERS[i*2+1]=POINTS[i*2+1];
    }
    
    // update classes
    if(updateClasses){
        update_classes(n, k);
        count_classes(n, k);
    }
}

void cuda_init(int n, int k){ // malloc and copy data to device
	cudaError_t cuerr = cudaSuccess; // use with cudaGetErrorString(cuerr);
	int noerr = 1;
	// malloc
	noerr &= (cuerr = cudaMalloc((void **)&D_POINTS, S_POINTS)) == cudaSuccess;
	//printf("err code %s\n", cudaGetErrorString(cuerr));
	noerr &= (cuerr = cudaMalloc((void **)&D_CLASSES, S_CLASSES)) == cudaSuccess;
	noerr &= (cuerr = cudaMalloc((void **)&D_NUM_CLASSES, S_NUM_CLASSES)) == cudaSuccess;
	noerr &= (cuerr = cudaMalloc((void **)&D_CLUSTERS, S_CLUSTERS)) == cudaSuccess;
    noerr &= (cuerr = cudaMalloc((void **)&D_OLD_CLUSTERS, S_CLUSTERS)) == cudaSuccess;
    
    
    
    if (!noerr)
    {
        fprintf(stderr, "Failed to allocate device vector\n");
        exit(EXIT_FAILURE);
    }
	
	// copy data
	noerr = 1;
	noerr &= cudaMemcpy(D_POINTS, POINTS, S_POINTS, cudaMemcpyHostToDevice) == cudaSuccess;
	noerr &= cudaMemcpy(D_CLUSTERS, CLUSTERS, S_CLUSTERS, cudaMemcpyHostToDevice) == cudaSuccess;
	noerr &= cudaMemcpy(D_OLD_CLUSTERS, D_CLUSTERS, S_CLUSTERS, cudaMemcpyDeviceToDevice) == cudaSuccess;
    if (!noerr)
    {
        fprintf(stderr, "Failed to copy data from host to device\n");
        exit(EXIT_FAILURE);
    }
	
	// blocksPerGrid
	BlocksPerGridN = (n + ThreadsPerBlock - 1) / ThreadsPerBlock;
	BlocksPerGridK = (k + ThreadsPerBlock - 1) / ThreadsPerBlock;
	printf("Using %d blocks of %d threads\n", BlocksPerGridN, ThreadsPerBlock);
	
    // update classes
    cuda_update_classes(n, k);
    cuda_count_classes(n, k);
}

void cuda_toHost(int n, int k){
	cudaError_t cuerr = cudaSuccess; // use with cudaGetErrorString(cuerr);
	int noerr = 1;
    noerr = 1;
	noerr &= (cuerr = cudaMemcpy(CLUSTERS, D_CLUSTERS, S_CLUSTERS, cudaMemcpyDeviceToHost)) == cudaSuccess;
	noerr &= (cuerr = cudaMemcpy(CLASSES, D_CLASSES, S_CLASSES, cudaMemcpyDeviceToHost)) == cudaSuccess;
    if (!noerr)
    {
        fprintf(stderr, "Failed to copy data from host to device\n");
        exit(EXIT_FAILURE);
    }
}

int data_count(char *fileName){
    FILE *inputFile;
    float x, y;
    int count=0;
    inputFile = fopen(fileName, "r");
    while(fscanf(inputFile, "%f,%f\n", &x, &y)==2){
        count++;    
        //printf("%f,%f\n",tmp1,tmp2);
    }
    fclose(inputFile);
    return count;
}

int cmd_parser(int argc, char **argv, int *n, int *k, int *t, char *input){
    int invalid;
    int valid;
    char ch;
    char usage[] = "Usage: %s -n N -k K -t T -i Input.txt [-g]\n"
                   "    N: Number_of_Points, default: the number of lines in Input_File\n"
                   "    K: default: 2\n"
                   "    T: max iterations for the kmeans algorithm\n"
                   "    Input: should be n lines, two floats in each line and split by ','\n"
                   "    -g: Use GPU, otherwise, use CPU only.\n"
				   "    -s: synchronize after each step (for debug).\n"
		           "    Results will be in Classes.txt and Clusters.txt\n";
    invalid = 0;
    valid = 0;
    if(argc==1){
        invalid = 1;
    }
    
    //default values
    *n = -1;
    *k = 2;
    *t = 1;
    USEGPU = 0;
    SYNC = 0;
    while((ch = getopt(argc, argv, "n:k:t:i:gsh")) != -1) {
        switch(ch) {
            case 'n':
                sscanf(optarg, "%d", n);
                break;
            case 'k':
                sscanf(optarg, "%d", k);
                break;
            case 't':
                sscanf(optarg, "%d", t);
                break;
            case 'i':
                strncpy(input, optarg, 256);
                valid = 1;
                break;
            case 'g':
                USEGPU = 1;
                break;
            case 's':
                SYNC = 1;
                break;
            case 'h':  //print help
                invalid = 1;
                break;
            case '?':
                invalid = 1;
            default:
                ;
        }
    }
	
    if(valid && *n==-1){
        *n = data_count(input);
    }
    
    
    invalid = invalid || !valid;
    if(invalid){
        printf(usage, argv[0]);
    }
    
    if(*n>MAXN){
        invalid = 1;
        printf("N is too large\n");
    }
    
    
    //printf("option N: %d\n", *n);
    //printf("option K: %d\n", *k);
    //printf("option T: %d\n", *t);
    //printf("option Input: %s\n", input);
    //printf("invalid %d\n", invalid);
	
    return invalid;    
}

int main(int argc, char **argv) {
    int t;
    srand(time(0));
    if(cmd_parser(argc, argv, &N, &K, &T, INPUT_FILE)){ // not enough parameters
        return 1;
    }
    
    if(USEGPU){
        init(N, K, INPUT_FILE, 0);
        cuda_init(N, K);
    }else{
        init(N, K, INPUT_FILE, 1);
    }
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(t=0;t<T;t++){ 
        
        if(USEGPU){
            cuda_clean_clusters(N, &K, SYNC);
		    cuda_update_classes(N, K, SYNC);
            cuda_update_clusters(N, K, SYNC);
        }else{   
            clean_clusters(N, &K);
	        update_classes(N, K);
            update_clusters(N, K);
        }
		if(SYNC){
			printf("NUM CLASSES ");
			for(int i=0;i<k;i++){
				printf("%d, ",NUM_CLASSES[i]);
			}
			printf("\n");
		}
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    printf("Time for core computation: %f ms\n", time);
    if(USEGPU){
        cuda_toHost(N, K);
    }
	
    write_results(N, K);
    return 0;
}
