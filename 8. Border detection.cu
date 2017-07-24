
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

#define ERROR_CHECK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
    printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__);}}

// Tamaño de las máscaras
#define N 3

// Hebras por dimensión de cada bloque
#define THREADS 32

// Máscara Vertical en el Device
__constant__ int maskV[N][N] = {{-1, 0, 1},
							    {-2, 0, 2},
							    {-1, 0, 1}};
// Máscara Horizontal en el Device
__constant__ int maskH[N][N] = {{-1, -2, -1},
							    { 0,  0,  0},
							    { 1,  2,  1}};

// Máscara Vertical en el Host
const int maskVSec[N][N] = {{-1, 0, 1},
	  				        {-2, 0, 2},
						    {-1, 0, 1}};
// Máscara Horizontal en el Host
const int maskHSec[N][N] = {{-1, -2, -1},
					        { 0,  0,  0},
						    { 1,  2,  1}};

// Cabecera de fichero BMP sin la signature
struct BMPHEADER
{
	unsigned int iFileSize;
	unsigned int iReserved;
	unsigned int iFileOffsetToPixelArray;

	unsigned int iDIBHeaderSize;
	unsigned int iWidth;
	unsigned int iHeigth;
	unsigned short iColorPlanes;
	unsigned short iBitsPerPixel;
	unsigned int iCompression;
	unsigned int iSizeOfDataWithPadding;
	unsigned int iPrintResolutionHorizontal;
	unsigned int iPrintResolutionVertical;
	unsigned int iColorsInPalete;
	unsigned int iImportantColors;
};

// Estructura para un pixel (B G R) (1 byte x 1 byte x 1 byte)
struct PIXEL
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};

// Estructura para un pixel con signo
struct SIGNED_PIXEL
{
	int B;
	int G;
	int R;
};


// Filtrado paralelo
__global__ void Filtro2DParalelo(SIGNED_PIXEL *dev_input, SIGNED_PIXEL *dev_output, unsigned iWidth, unsigned int iHeigth)
{
	/* COMPLETAR */
	int iCol = threadIdx.x + blockDim.x*blockIdx.x;
	int iFila = threadIdx.y + blockDim.y*blockIdx.y;

	int pos = iFila*(int)iWidth + iCol;
	int posA = (iFila - 1)*(int)iWidth + iCol - 1;
	int posB = (iFila)*(int)iWidth + iCol - 1;
	int posC = (iFila + 1)*(int)iWidth + iCol - 1;
	int posD = (iFila - 1)*(int)iWidth + iCol;
	int posF = (iFila + 1)*(int)iWidth + iCol;
	int posG = (iFila - 1)*(int)iWidth + iCol + 1;
	int posH = (iFila)*(int)iWidth + iCol + 1;
	int posI = (iFila + 1)*(int)iWidth + iCol + 1;

	dev_output[pos].R = 0;
	dev_output[pos].G = 0;
	dev_output[pos].B = 0;

	// Pixel a
	if (posA < 0){

	} else {
		dev_output[pos].R = dev_output[pos].R + maskV[0][0] * dev_input[posA].R +
			maskH[0][0] * dev_input[posA].R;
		dev_output[pos].G = dev_output[pos].G + maskV[0][0] * dev_input[posA].G +
			maskH[0][0] * dev_input[posA].G;
		dev_output[pos].B = dev_output[pos].B + maskV[0][0] * dev_input[posA].B +
			maskH[0][0] * dev_input[posA].B;
	}

	// Pixel b

	if (posB < 0){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[0][1] * dev_input[posB].R +
			maskH[0][1] * dev_input[posB].R;
		dev_output[pos].G = dev_output[pos].G + maskV[0][1] * dev_input[posB].G +
			maskH[0][1] * dev_input[posB].G;
		dev_output[pos].B = dev_output[pos].B + maskV[0][1] * dev_input[posB].B +
			maskH[0][1] * dev_input[posB].B;
	}

	// Pixel c

	if (posC < 0){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[0][2] * dev_input[posC].R +
			maskH[0][2] * dev_input[posC].R;
		dev_output[pos].G = dev_output[pos].G + maskV[0][2] * dev_input[posC].G +
			maskH[0][2] * dev_input[posC].G;
		dev_output[pos].B = dev_output[pos].B + maskV[0][2] * dev_input[posC].B +
			maskH[0][2] * dev_input[posC].B;
	}

	// Pixel d

	if (posD < 0){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[1][0] * dev_input[posD].R +
			maskH[1][0] * dev_input[posD].R;
		dev_output[pos].G = dev_output[pos].G + maskV[1][0] * dev_input[posD].G +
			maskH[1][0] * dev_input[posD].G;
		dev_output[pos].B = dev_output[pos].B + maskV[1][0] * dev_input[posD].B +
			maskH[1][0] * dev_input[posD].B;
	}

	// Pixel e

	dev_output[pos].R = dev_output[pos].R + maskV[1][1] * dev_input[pos].R +
		maskH[1][1] * dev_input[pos].R;
	dev_output[pos].G = dev_output[pos].G + maskV[1][1] * dev_input[pos].G +
		maskH[1][1] * dev_input[pos].G;
	dev_output[pos].B = dev_output[pos].B + maskV[1][1] * dev_input[pos].B +
		maskH[1][1] * dev_input[pos].B;

	// Pixel f

	if (posF < 0){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[1][2] * dev_input[posF].R +
			maskH[1][2] * dev_input[posF].R;
		dev_output[pos].G = dev_output[pos].G + maskV[1][2] * dev_input[posF].G +
			maskH[1][2] * dev_input[posF].G;
		dev_output[pos].B = dev_output[pos].B + maskV[1][2] * dev_input[posF].B +
			maskH[1][2] * dev_input[posF].B;
	}

	// Pixel g

	if (posG > iWidth*iHeigth){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[2][0] * dev_input[posG].R +
			maskH[2][0] * dev_input[posG].R;
		dev_output[pos].G = dev_output[pos].G + maskV[2][0] * dev_input[posG].G +
			maskH[2][0] * dev_input[posG].G;
		dev_output[pos].B = dev_output[pos].B + maskV[2][0] * dev_input[posG].B +
			maskH[2][0] * dev_input[posG].B;
	}

	// Pixel h

	if (posH > iWidth*iHeigth){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[2][1] * dev_input[posH].R +
			maskH[2][1] * dev_input[posH].R;
		dev_output[pos].G = dev_output[pos].G + maskV[2][1] * dev_input[posH].G +
			maskH[2][1] * dev_input[posH].G;
		dev_output[pos].B = dev_output[pos].B + maskV[2][1] * dev_input[posH].B +
			maskH[2][1] * dev_input[posH].B;
	}

	// Pixel i

	if (posI > iWidth*iHeigth){

	}
	else {
		dev_output[pos].R = dev_output[pos].R + maskV[2][2] * dev_input[posI].R +
			maskH[2][2] * dev_input[posI].R;
		dev_output[pos].G = dev_output[pos].G + maskV[2][2] * dev_input[posI].G +
			maskH[2][2] * dev_input[posI].G;
		dev_output[pos].B = dev_output[pos].B + maskV[2][2] * dev_input[posI].B +
			maskH[2][2] * dev_input[posI].B;
	}
}

// 



// Cálculo paralelo del valor mínimo y máximo en cada componente de los píxeles de la imagen filtrada
__global__ void calculoMinMaxParalelo(SIGNED_PIXEL *dev_output, SIGNED_PIXEL *dev_Min, SIGNED_PIXEL *dev_Max, unsigned int iWidth, unsigned int iHeigth)
{
	/* COMPLETAR */
	unsigned int iCol = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int iFila = threadIdx.y + blockDim.y*blockIdx.y;

	unsigned pos = iFila*iWidth + iCol;

	atomicMin(&dev_Min->R, dev_output[pos].R);
	atomicMin(&dev_Min->G, dev_output[pos].G);
	atomicMin(&dev_Min->B, dev_output[pos].B);
	atomicMax(&dev_Max->R, dev_output[pos].R);
	atomicMax(&dev_Max->G, dev_output[pos].G);
	atomicMax(&dev_Max->B, dev_output[pos].B);
}


// Cálculo paralelo de la normalización de los valores de los píxeles entre [0,255] para cada componente
__global__ void normalizacion2DParalelo(SIGNED_PIXEL *dev_output, SIGNED_PIXEL *dev_Min, float fFactorR, float fFactorG, float fFactorB, unsigned iWidth, unsigned int iHeigth)
{
	/* COMPLETAR */
	unsigned int iCol = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int iFila = threadIdx.y + blockDim.y*blockIdx.y;

	unsigned pos = iFila*iWidth + iCol;

	dev_output[pos].R = (dev_output[pos].R - dev_Min->R) * fFactorR;
	dev_output[pos].G = (dev_output[pos].G - dev_Min->G) * fFactorG;
	dev_output[pos].B = (dev_output[pos].B - dev_Min->B) * fFactorB;
}


void Filtro2DSecuencial(SIGNED_PIXEL *host_input, SIGNED_PIXEL *sec_output, unsigned iWidth, unsigned int iHeigth)
{
	/* COMPLETAR */
	SIGNED_PIXEL max, min;
	max.B = INT_MIN;
	max.R = INT_MIN;
	max.G = INT_MIN;
	min.B = INT_MAX;
	min.R = INT_MAX;
	min.G = INT_MAX;
	for (unsigned int c = 0; c < iWidth*iHeigth; c++)
	{
		sec_output[c].B = 0;
		sec_output[c].G = 0;
		sec_output[c].R = 0;

	}
	for (int i = 0; i < (int)iWidth*(int)iHeigth; i++)
	{
		// Máscaras
		// Pixel a
		if (i - (int)iWidth - 1 < 0){
			
		} else {
			sec_output[i].B = sec_output[i].B + (host_input[i - iWidth - 1].B*maskVSec[0][0]) +
				(host_input[i - iWidth - 1].B*maskHSec[0][0]);
			sec_output[i].G = sec_output[i].G + (host_input[i - iWidth - 1].G*maskVSec[0][0]) +
				(host_input[i - iWidth - 1].G*maskHSec[0][0]);
			sec_output[i].R = sec_output[i].R + (host_input[i - iWidth - 1].R*maskVSec[0][0]) +
				(host_input[i - iWidth - 1].R*maskHSec[0][0]);
		}
		// Pixel b
		if (i - (int)iWidth < 0){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i - iWidth].B*maskVSec[0][1]) +
				(host_input[i - iWidth].B*maskHSec[0][1]);
			sec_output[i].G = sec_output[i].G + (host_input[i - iWidth].G*maskVSec[0][1]) +
				(host_input[i - iWidth - 1].G*maskHSec[0][1]);
			sec_output[i].R = sec_output[i].R + (host_input[i - iWidth].R*maskVSec[0][1]) +
				(host_input[i - iWidth].R*maskHSec[0][1]);
		}
		// Pixel c
		if (i - (int)iWidth + 1 < 0){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i - iWidth + 1].B*maskVSec[0][2]) +
				(host_input[i - iWidth + 1].B*maskHSec[0][2]);
			sec_output[i].G = sec_output[i].G + (host_input[i - iWidth + 1].G*maskVSec[0][2]) +
				(host_input[i - iWidth + 1].G*maskHSec[0][2]);
			sec_output[i].R = sec_output[i].R + (host_input[i - iWidth + 1].R*maskVSec[0][2]) +
				(host_input[i - iWidth + 1].R*maskHSec[0][2]);
		}
		// Pixel d
		if (i - 1 < 0){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i - 1].B*maskVSec[1][0]) +
				(host_input[i - 1].B*maskHSec[1][0]);
			sec_output[i].G = sec_output[i].G + (host_input[i - 1].G*maskVSec[1][0]) +
				(host_input[i - 1].G*maskHSec[1][0]);
			sec_output[i].R = sec_output[i].R + (host_input[i - 1].R*maskVSec[1][0]) +
				(host_input[i - 1].R*maskHSec[1][0]);
		}
		// Pixel e
		sec_output[i].B = sec_output[i].B + (host_input[i].B*maskVSec[1][1]) +
			(host_input[i].B*maskHSec[1][1]);
		sec_output[i].G = sec_output[i].G + (host_input[i].G*maskVSec[1][1]) +
			(host_input[i].G*maskHSec[1][1]);
		sec_output[i].R = sec_output[i].R + (host_input[i].R*maskVSec[1][1]) +
			(host_input[i].R*maskHSec[1][1]);
		// Pixel f
		if (i + 1 < 0){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i + 1].B*maskVSec[1][2]) +
				(host_input[i + 1].B*maskHSec[1][2]);
			sec_output[i].G = sec_output[i].G + (host_input[i + 1].G*maskVSec[1][2]) +
				(host_input[i + 1].G*maskHSec[1][2]);
			sec_output[i].R = sec_output[i].R + (host_input[i + 1].R*maskVSec[1][2]) +
				(host_input[i + 1].R*maskHSec[1][2]);
		}
		// Pixel g
		if (i + (int)iWidth - 1 > (int)iWidth*(int)iHeigth){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i + iWidth - 1].B*maskVSec[2][0]) +
				(host_input[i + iWidth - 1].B*maskHSec[2][0]);
			sec_output[i].G = sec_output[i].G + (host_input[i + iWidth - 1].G*maskVSec[2][0]) +
				(host_input[i + iWidth - 1].G*maskHSec[2][0]);
			sec_output[i].R = sec_output[i].R + (host_input[i + iWidth - 1].R*maskVSec[2][0]) +
				(host_input[i + iWidth - 1].R*maskHSec[2][0]);
		}
		// Pixel h
		if (i + (int)iWidth > (int)iWidth*(int)iHeigth){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i + iWidth].B*maskVSec[2][1]) +
				(host_input[i + iWidth].B*maskHSec[2][1]);
			sec_output[i].G = sec_output[i].G + (host_input[i + iWidth].G*maskVSec[2][1]) +
				(host_input[i + iWidth].G*maskHSec[2][1]);
			sec_output[i].R = sec_output[i].R + (host_input[i + iWidth].R*maskVSec[2][1]) +
				(host_input[i + iWidth].R*maskHSec[2][1]);
		}
		// Pixel i
		if (i + (int)iWidth + 1 > (int)iWidth*(int)iHeigth){

		}
		else {
			sec_output[i].B = sec_output[i].B + (host_input[i + iWidth + 1].B*maskVSec[2][2]) +
				(host_input[i + iWidth + 1].B*maskHSec[2][2]);
			sec_output[i].G = sec_output[i].G + (host_input[i + iWidth + 1].G*maskVSec[2][2]) +
				(host_input[i + iWidth + 1].G*maskHSec[2][2]);
			sec_output[i].R = sec_output[i].R + (host_input[i + iWidth + 1].R*maskVSec[2][2]) +
				(host_input[i + iWidth + 1].R*maskHSec[2][2]);
		}
	}

	// Calculo de máximos y mínimos

	for (unsigned int f = 0; f < iWidth*iHeigth; f++)
	{
		if (sec_output[f].R > max.R)
			max.R = sec_output[f].R;
		if (sec_output[f].G > max.G)
			max.G = sec_output[f].G;
		if (sec_output[f].B > max.B)
			max.B = sec_output[f].B;
		if (sec_output[f].R < min.R)
			min.R = sec_output[f].R;
		if (sec_output[f].G < min.G)
			min.G = sec_output[f].G;
		if (sec_output[f].B < min.B)
			min.B = sec_output[f].B;
	}

	// Normalizar
	for (unsigned int z = 0; z < iWidth*iHeigth; z++)
	{
		sec_output[z].R = ((sec_output[z].R - min.R) * 255) / (max.R - min.R);
		sec_output[z].G = ((sec_output[z].G - min.G) * 255) / (max.G - min.G);
		sec_output[z].B = ((sec_output[z].B - min.B) * 255) / (max.B - min.B);
	}
}


int main(int argc, char **argv)
{
	struct BMPHEADER BMPheader;
	char signature[2];
	size_t dataToRead;
	struct PIXEL pixel;

	struct SIGNED_PIXEL *host_input, *host_output, *sec_output, host_Min, host_Max;
	struct SIGNED_PIXEL *dev_input, *dev_output, *dev_Max, *dev_Min;

	if (argc != 4)
	{
		printf("USO:\nFiltro2D <fichero imagen> <fichero salida paralelo> <fichero salida secuencial\n");
		return -1;
	}

	// Se abre el fichero de la imagen
	FILE *fData = fopen(argv[1], "rb");
	if (fData == NULL)
	{
		printf("Error abriendo el fichero de datos\n");
		return -1;
	}

	// Se lee la signature y se comprueba que realmente es un fichero BMP
	fread((void *) signature, sizeof(char), 2, fData);
	if (signature[0] != 'B' || signature[1] != 'M')
	{
		printf("El fichero de la imagen no es BMP\n");
		fclose(fData);
		return -1;
	}

	// Se lee la cabecera del fichero BMP
	fread((void *) &BMPheader, sizeof(BMPHEADER), 1, fData);

	// Los datos a leer serán el alto x ancho
	dataToRead = BMPheader.iWidth * BMPheader.iHeigth;

	// Se pide memoria para las variables en el host
	host_input =  (SIGNED_PIXEL *) malloc(dataToRead * sizeof(SIGNED_PIXEL));
	host_output = (SIGNED_PIXEL *) malloc(dataToRead * sizeof(SIGNED_PIXEL));
	sec_output =  (SIGNED_PIXEL *) malloc(dataToRead * sizeof(SIGNED_PIXEL));

	// Se cargan los datos de cada pixel en memoria del Host
	for (int iPos = 0; iPos < (int)dataToRead; iPos++)
	{
		fread((void *) &pixel, sizeof(PIXEL), 1, fData);
		host_input[iPos].R = pixel.R;
		host_input[iPos].G = pixel.G;
		host_input[iPos].B = pixel.B;
	}
	fclose(fData);

	// Se inicializan los valores mínimos
	host_Min.R = INT_MAX;
	host_Min.G = INT_MAX;
	host_Min.B = INT_MAX;

	// Se inicializan los valores máximos
	host_Max.R = INT_MIN;
	host_Max.G = INT_MIN;
	host_Max.B = INT_MIN;

	// Pedir memoria en el Device
	cudaMalloc((void **) &dev_input, dataToRead * sizeof(SIGNED_PIXEL));
	cudaMalloc((void **) &dev_output, dataToRead * sizeof(SIGNED_PIXEL));
	cudaMalloc((void **) &dev_Min, sizeof(SIGNED_PIXEL));
	cudaMalloc((void **) &dev_Max, sizeof(SIGNED_PIXEL));
	ERROR_CHECK;
	
	// Transferir los datos del Host al Device
	cudaMemcpy(dev_input, host_input, dataToRead * sizeof(SIGNED_PIXEL), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Min, &host_Min, sizeof(SIGNED_PIXEL), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Max, &host_Max, sizeof(SIGNED_PIXEL), cudaMemcpyHostToDevice);
	ERROR_CHECK;

	dim3 grid(BMPheader.iWidth / THREADS, BMPheader.iHeigth / THREADS);
	dim3 block(THREADS,THREADS);
	
	// Se llama al filtrado paralelo CUDA
	Filtro2DParalelo<<<grid,block>>>(dev_input, dev_output, BMPheader.iWidth, BMPheader.iHeigth);
	cudaDeviceSynchronize();
	ERROR_CHECK;

	// Se llama al cálculo paralelo CUDA del mínimo y máximo
	calculoMinMaxParalelo<<<grid,block>>>(dev_output, dev_Min, dev_Max, BMPheader.iWidth, BMPheader.iHeigth);
	cudaDeviceSynchronize();
	ERROR_CHECK;

	// Se copia el resultado del mínimo y máximo valor
	cudaMemcpy(&host_Min, dev_Min, sizeof(SIGNED_PIXEL), cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_Max, dev_Max, sizeof(SIGNED_PIXEL), cudaMemcpyDeviceToHost);
	ERROR_CHECK;

	// Se calculan los factores de normalización
	float fFactorR = 255.0f / ((float) host_Max.R - (float)host_Min.R);
	float fFactorG = 255.0f / ((float) host_Max.G - (float)host_Min.G);
	float fFactorB = 255.0f / ((float) host_Max.B - (float)host_Min.B);

	// Se llama a la normalización paralela CUDA
	normalizacion2DParalelo<<<grid,block>>>(dev_output, dev_Min, fFactorR, fFactorG, fFactorB, BMPheader.iWidth, BMPheader.iHeigth);
	cudaDeviceSynchronize();
	ERROR_CHECK;

	// Copiar el resultado final del Device al Host
	cudaMemcpy(host_output, dev_output, dataToRead * sizeof(SIGNED_PIXEL), cudaMemcpyDeviceToHost);
	ERROR_CHECK;

	// Llamada al filtro secuencial
	Filtro2DSecuencial(host_input, sec_output, BMPheader.iWidth, BMPheader.iHeigth);

	// Se abre el fichero de salida paralela
	FILE *fDataOut = fopen(argv[2], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escribe la signature
	fwrite((void *) &signature, sizeof(char), 2, fDataOut);
	// Se escribe la cabecera
	fwrite((void *) &BMPheader, sizeof(BMPHEADER), 1, fDataOut);
	// Se escriben los datos del filtrado paralelo
	for (int iPos = 0; iPos < (int) dataToRead; iPos++)
	{
		pixel.R = host_output[iPos].R;
		pixel.G = host_output[iPos].G;
		pixel.B = host_output[iPos].B;
		fwrite((void *) &pixel, sizeof(PIXEL), 1, fDataOut);
	}
	fclose(fDataOut);

	// Se abre el fichero de salida secuencial
	fDataOut = fopen(argv[3], "wb");
	if (fDataOut == NULL)
	{
		printf("Error abriendo el fichero de salida\n");
		return -1;
	}

	// Se escribe la signature
	fwrite((void *) &signature, sizeof(char), 2, fDataOut);
	// Se escribe la cabecera
	fwrite((void *) &BMPheader, sizeof(BMPHEADER), 1, fDataOut);
	// Se escriben los datos del filtrado secuencial
	for (int iPos = 0; iPos < (int) dataToRead; iPos++)
	{
		pixel.R = sec_output[iPos].R;
		pixel.G = sec_output[iPos].G;
		pixel.B = sec_output[iPos].B;
		fwrite((void *) &pixel, sizeof(PIXEL), 1, fDataOut);
	}
	fclose(fDataOut);

	// Liberar la memoria del Host
	free(host_input);
	free(host_output);
	free(sec_output);

	// Librerar la memoria solicitada en el Device
	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaFree(dev_Min);
	cudaFree(dev_Max);

	return 0;
}
