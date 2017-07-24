
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N  512 

__global__ void add(float *a, float *b,float *c)
{
	int tid = blockIdx.x;

	c[tid] = a[tid] * b[tid];
}

int main(void)
{
	float host_a[N], host_b[N], host_c[N];
	float *dev_a, *dev_b, *dev_c;

	srand((unsigned)time(NULL));
	for (int i= 0;i	<N; i++)
	{
		host_a[i] = floorf(1000 * (rand() / (float)RAND_MAX));
		host_b[i] = floorf(1000 * (rand() / (float)RAND_MAX));
	}

	cudaMalloc((void **) &dev_a, N*sizeof(float));
	cudaMalloc((void **) &dev_b, N*sizeof(float));
	cudaMalloc((void **) &dev_c, N*sizeof(float));

	cudaMemcpy(dev_a, host_a, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, N*sizeof(float), cudaMemcpyHostToDevice);

	add<<<N,1>>>(dev_a, dev_b, dev_c);

	cudaMemcpy(host_c, dev_c, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i= 0;i	<N; i++)
		printf("%f * %f= %f\n", host_a[i], host_b[i], host_c[i]);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}