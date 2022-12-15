// Задание: найти такое значение вектора X = (x_1, x_2, y, phi1, phi2), чтобы 
// при подставлении его в систему уравнений F(X) = (f_1(X), f_2(X), f_3(X), f_4(X), f_5(X)), каждое уравнение в ней равнялось нулю

#include <iostream>
#include <cmath>
#include <cuda.h>

using namespace std;

float *X, *X_next; // искомые данные (x_1, x_2, y, phi1, phi2)

// __device__ чтобы был виден в девайс коде
__device__ int counter = 0; // счётчик того, что все уравнения стали меньше порога e (пока он меньше 5 сидим в цикле)
__device__ const float e = 0.0001; // порог для уравнений
__device__ const float delta_t = 0.01; // шаг по времени для алгоритма

int thread_count = 5; // число потоков (в блоке), 1 поток на 1 уроавнение
int block_count = 1; // число блоков

__device__ float F(float X[]){
    float res = 0;
    switch ( threadIdx.x ) { // возвращаем значение в зависимости от ID потока
        case 0:
            res = X[0] + X[2] * cos(3*3.14159265/2 - X[3]) + 0.353; // x_1 + y * cos(3*Pi/2 - phi1) - A_x, где A_x = -0.353
            break;
        case 1:
            res = X[1] + X[2] * cos(3*3.14159265/2 + X[4]) - 0.353; // x_2 + y * cos(3*Pi/2 + phi2) - B_x, где B_x = 0.353
            break;
        case 2:
            res = X[2] + X[2] * sin(3*3.14159265/2 - X[3]) - 0.3; // y + y * sin(3*Pi/2 - phi1) - A_y, где A_y = 0.3
            break;
        case 3:
            res = (X[3] + X[4]) * X[2] + (X[1] - X[0]) - 1.178; // (phi1 + phi2) * y + (x_2 - x_1) - C, где C = 3*Pi/8=1.178
            break;
        case 4:
            res = X[2] + X[2] * sin(3*3.14159265/2 + X[4]) - 0.3; // y + y * sin(3*Pi/2 + phi2) - B_y, где B_y = 0.3
            break;
    }
    if (abs(res) < e) // обновляем счётчик
        atomicAdd(&counter, 1);
    return res;
}

__global__ void Routine(float X[], float X_next[]){ // функция, запускаемая на девайсе
    while (counter < 5) { // проверка, что все целевые функции меньше порога e (счётчик должен равняться 5)
        counter = 0; // обнуляем счётчик
        __syncthreads(); // синхронизируемся после обнуления счётчика, чтобы какой-нибудь поток случайно не занулил его после инкрементации в F(X)
        X_next[threadIdx.x] = X[threadIdx.x] - F(X) * delta_t; // получаем значение для следующей итерации
        __syncthreads(); // синхронизируемся перед обновлением текущих значений (чтобы раньше времени не перезаписать где-то читаемые данные)
        X[threadIdx.x] = X_next[threadIdx.x]; // обновляем текущее значение
    }
}

int main() {
    cudaMallocManaged(&X, 5 * sizeof(float)); // выделение памяти под массив X (текущий) из 5 элементов типа float
    cudaMallocManaged(&X_next, 5 * sizeof(float)); // выделение памяти под массив X_next (после шага) из 5 элементов типа float
    for (int i=0; i<5; i++){ // заполнение созданных масивов
        X[i] = 0; // ==> X=[0,0,0,0,0]
        X_next[i] = 0; // ==> X_next=[0,0,0,0,0]
    }

    Routine<<<block_count, thread_count>>>(X, X_next); // вызов девайсной функции (передаём число блоков и число потоков в блоке)
    cudaDeviceSynchronize();

    for (int i=0; i<5; ++i) // вывод посчитанного массива
        printf("%f\n", X[i]);

    cudaFree(X); // освобождение памяти
    cudaFree(X_next);
    return 0;
}