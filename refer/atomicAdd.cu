#include <stdio.h>

#include "cuda.h"
// https://devtalk.nvidia.com/default/topic/411598/can-you-give-me-sample-code-for-atomicadd-/


__global__ void Sum( int *sum , int size, int* index)
{
	register int i = atomicAdd(index,1);
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	sum[i] = idx;
}

int main(int argc, char* argv[])

{

	int W = 256;
	int H = 256;
	int *hSum ,*dSum , size = 50;
	int* d_index=0;
	int h_index=0;
	hSum = (int*)malloc(sizeof(int)*W*H);
	memset( hSum, 0, sizeof(int)*W*H);
	cudaMalloc( (void**) &dSum, sizeof(int)*W*H );
	cudaMalloc( (void**) &d_index, sizeof(int) );
	cudaMemcpy(dSum, hSum , sizeof(int)*W*H, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index, &h_index , sizeof(int), cudaMemcpyHostToDevice);
	Sum<<<W,H>>>( dSum , size, d_index );
	
	cudaMemcpy(hSum, dSum, sizeof(int)*W*H, cudaMemcpyDeviceToHost);

	cudaMemcpy(&h_index , d_index, sizeof(int), cudaMemcpyDeviceToHost);

	fprintf(stderr, "%d\n", h_index);
	for( int i=0; i<W*H; ++i )
		fprintf( stdout, " %d %d\n", i, hSum[i] );

	free(hSum);
	cudaFree(dSum);
	cudaFree(d_index);
	
	return 0;
}






