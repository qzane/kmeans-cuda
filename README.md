# kmeans-cuda
COMP 633 project: kmeans in cuda

# todo
0. write cuda version～～done
1. read clusters from file～～done
2. the choise for init clusters～～done
3. the ending criteria
4. data maker (maybe using python)～～done
5. visiualization (maybe using python)～～done
6. performance profile (running time)～～done
8. replace atomicAdd with parallel reduction～～done
9. optimize the program using skills in [reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

# cuda
* the compute capability of Titan V is 7.0 (on phaedra)
* the compute capability of GTX 1080 Ti is 6.1 (my desktop)
* the cuda version I'm using is 9.2 (on phaedra)
* Maximum x-dimension of a grid of thread blocks: 2^31-1, starting from cc3.0
* Maximum number of threads per block: 1024, starting from cc2.0
* Maximum x- or y-dimension of a block: 1024, starting from cc2.0
* atomicAdd_system, (atomic through all CPUs and GPUs, may not need it)starting from cc6.0
* 32-bit floating-point version of atomicAdd(), starting from cc2.0
* 64-bit floating-point version of atomicAdd(), starting from cc6.0

# refer: 
[compute capability](https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications)
[atomic functions](https://docs.nvidia.com/cuda/archive/9.2/cuda-c-programming-guide/index.html#atomic-functions)
[reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
[shared memory](https://stackoverflow.com/questions/8011376/when-is-cudas-shared-memory-useful)
