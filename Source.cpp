#include <iostream>
#include <iomanip>
#include <Windows.h>
#include <time.h>
#include <random>
#include <emmintrin.h>
#include <immintrin.h>

using namespace std;

#define L_value 2816
#define M_value 2816
#define N_value 2816

#define l_value 704
#define m_value 704
#define n_value 704

double matrixA[L_value][M_value];
double matrixB[M_value][N_value];
double matrixC[L_value][N_value];
double matrixC_Vectorized[L_value][N_value];
double matrixC_Cached[L_value][N_value];
double matrixC_SSE[L_value][N_value];

bool checkEquality()
{
	bool flag = 0;
	for (int i = 0; i < L_value; i++)
	{
		for (int j = 0; j < N_value; j++)
		{
			if (matrixC[i][j] != matrixC_Vectorized[i][j] || matrixC[i][j] != matrixC_Cached[i][j])
				flag = 1;
		}
	}
	if (flag)
		return false;
	else
		return true;
}

void clear(double matrix[L_value][N_value])
{
	for (int i = 0; i < L_value; i++)
	{
		for (int j = 0; j < N_value; j++)
		{
			matrix[i][j] = 0;
		}
	}
}

void main()
{
	srand(time(NULL));

	for (int i = 0; i < L_value; i++)
	{
		for (int j = 0; j < M_value; j++)
		{
			matrixA[i][j] = 0 + rand() % 10;
		}
	}

	for (int i = 0; i < M_value; i++)
	{
		for (int j = 0; j < N_value; j++)
		{
			matrixB[i][j] = 0 + rand() % 10;
		}
	}

	ULONGLONG  startTime;
	ULONGLONG  resultTimeVectorized;
	ULONGLONG  resultTimeCache;
	ULONGLONG  resultTimeSSE;


	clear(matrixC_Vectorized);
	startTime = GetTickCount64();
	for (int i = 0; i < L_value; i++)
	{
		for (int j = 0; j < N_value; j++)
		{
			for (int e = 0; e < M_value; e++)
			{
				matrixC_Vectorized[i][j] += matrixA[i][e] * matrixB[e][j];
			}
		}
	}
	resultTimeVectorized = GetTickCount64() - startTime;

	clear(matrixC_Cached);
	startTime = GetTickCount64();
	__m256d reg1;
	__m256d reg2;
	__m256d reg3;
	__m256d reg4;
	__m256d reg5;
	for (int k = 0; k < L_value / l_value; k++)
	{
		for (int h = 0; h < M_value / m_value; h++)
		{
			for (int p = 0; p < N_value / n_value; p++)
			{
				for (int i = 0; i < l_value; i++)
				{
					for (int j = 0; j < m_value; j++)
					{
						reg1 = _mm256_set1_pd(matrixA[i + k * l_value][j + h * m_value]);
						for (int e = 0; e < n_value / 4; e++)
						{
							reg3 = _mm256_load_pd(&matrixC_Cached[i + k * l_value][e * 4 + p * n_value]);
							reg2 = _mm256_load_pd(&matrixB[j + h * m_value][e * 4 + p * n_value]);

							reg5 = _mm256_mul_pd(reg1, reg2);
							reg4 = _mm256_add_pd(reg3, reg5);
							_mm256_store_pd(&matrixC_Cached[i + k * l_value][e * 4 + p * n_value], reg4);
						}
					}
				}
			}
		}
	}
	resultTimeCache = GetTickCount64() - startTime;

	clear(matrixC);
	startTime = GetTickCount64();

	for (int i = 0; i < L_value; i++)
	{
		for (int j = 0; j < M_value; j++)
		{
			reg1 = _mm256_set1_pd(matrixA[i][j]);
			for (int e = 0; e < N_value / 4; e++)
			{
				reg3 = _mm256_load_pd(&matrixC[i][e * 4]);
				reg2 = _mm256_load_pd(&matrixB[j][e * 4]);

				reg5 = _mm256_mul_pd(reg1, reg2);
				reg4 = _mm256_add_pd(reg3, reg5);
				_mm256_store_pd(&matrixC[i][e * 4], reg4);
			}
		}
	}
	resultTimeSSE = GetTickCount64() - startTime;

	if (checkEquality())
		cout << "Matrixes are equal" << endl;
	else
		cout << "Matrixes are not equal" << endl;
	cout << "Time with simple vectorization: " << resultTimeVectorized << endl;
	cout << "Time with manual vectirzation: " << resultTimeSSE << endl;
	cout << "Time with cache optimization: " << resultTimeCache << endl;

	system("pause");
}