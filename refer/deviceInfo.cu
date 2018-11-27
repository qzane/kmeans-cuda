#include<stdio.h>
#include<cuda_runtime.h>
#include<iostream>
int main(){
	int dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);
	std::cout << "GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "Number of Multiprocessors(SM)：" << devProp.multiProcessorCount << std::endl;
	std::cout << "sharedMemPerBlock：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "maxThreadsPerBlock：" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "maxThreadsPerMultiProcessor(SM)：" << devProp.maxThreadsPerMultiProcessor << std::endl;
}