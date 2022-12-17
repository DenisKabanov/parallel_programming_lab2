// Задание: найти такое значение вектора X = (x_1, x_2, y, phi1, phi2), чтобы 
// при подставлении его в систему уравнений F(X) = (f_1(X), f_2(X), f_3(X), f_4(X), f_5(X)), каждое уравнение в ней равнялось нулю

// Примечание: A_x, B_x, A_y=B_y

#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cuda.h>

using namespace std;
using namespace std::chrono;

double *X, *X_next; // искомые данные (x_1, x_2, y, phi1, phi2)

// __device__ чтобы был виден в девайс коде
__device__ int counter = 0; // счётчик того, что все уравнения стали меньше порога e (пока он меньше 5 сидим в цикле)
__device__ const double e = 0.000001; // порог для уравнений
__device__ const double delta_tau = 0.005; // шаг по фиктивному времени для алгоритма (внутренний цикл)
__device__ const double delta_t = 0.01; // шаг по реальному времени для алгоритма (внешний цикл)

__device__ const double p = 2000; // давление внутри баллона
__device__ const double m = 100; // масса
__device__ const double g = 9.780; // ускорени свободного падения
__device__ const double t_max = 2.5; // предел времени моделирования
__device__ double n = 0;
__device__ double l; // расстояние между точками x_1 и x_2

__device__ double A_y = 0.3; // значения A_y для задания на 5
__device__ double v = 0, v_next = 0; // скорость для залания 5
__device__ double A_x = -0.353, B_x = 0.353, B_y = 0.3, C = 1.178;

int thread_count = 1; // число потоков (в блоке), 1 поток на 1 уроавнение
int block_count = 1; // число блоков

__device__ double F(double X[], const int& i){
    double res = 0;
    switch ( i ) { // возвращаем значение в зависимости от ID потока
        case 0:
            res = X[0] + X[2] * cos(3*3.141592/2 - X[3]) - A_x; // x_1 + y * cos(3*Pi/2 - phi1) - A_x, где A_x = -0.353
            break;
        case 1:
            res = X[1] + X[2] * cos(3*3.141592/2 + X[4]) - B_x; // x_2 + y * cos(3*Pi/2 + phi2) - B_x, где B_x = 0.353
            break;
        case 2:
            res = X[2] + X[2] * sin(3*3.14159265/2 - X[3]) - A_y; // y + y * sin(3*Pi/2 - phi1) - A_y, где изначально A_y = 0.3, а потом изменяется
            break;
        case 3:
            res = (X[3] + X[4]) * X[2] + (X[1] - X[0]) - C; // (phi1 + phi2) * y + (x_2 - x_1) - C, где C = 3*Pi/8=1.178
            break;
        case 4:
            res = X[2] + X[2] * sin(3*3.14159265/2 + X[4]) - B_y; // y + y * sin(3*Pi/2 + phi2) - B_y, где B_y = 0.3
            break;
    }
    if (abs(res) < e) // обновляем счётчик
        counter++;
    return res;
}

__global__ void Routine(double X[], double X_next[]){ // функция, запускаемая на девайсе
    while (n < t_max) {
        while (counter < 5) { // проверка, что все целевые функции меньше порога e (счётчик должен равняться 5)
            counter = 0; // обнуляем счётчик
            for(int i=0; i<5; ++i)
                X_next[i] = X[i] - F(X, i) * delta_tau; // получаем значение для следующей итерации
            for(int i=0; i<5; ++i)
                X[i] = X_next[i]; // обновляем текущее значение
        }
        // printf("%f %f %f %f %f %f\n", X[0], X[1], X[2], X[3], X[4], A_y);
        
        A_y += v * delta_t;
        B_y = A_y;
        l = X[1] - X[0];
        v += delta_t * (p * l - m * g) / m;
        n += delta_t;
        counter = 0;
    }
}

int main() {
    cudaMallocManaged(&X, 5 * sizeof(double)); // выделение памяти под массив X (текущий) из 5 элементов типа double
    cudaMallocManaged(&X_next, 5 * sizeof(double)); // выделение памяти под массив X_next (после шага) из 5 элементов типа double
    for (int i=0; i<5; i++){ // заполнение созданных масивов
        X[i] = 0; // ==> X=[0,0,0,0,0]
        X_next[i] = 0; // ==> X_next=[0,0,0,0,0]
    }

    auto start = high_resolution_clock::now();

    Routine<<<block_count, thread_count>>>(X, X_next); // вызов девайсной функции (передаём число блоков и число потоков в блоке)
    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    cout << fixed << setprecision(12) << duration.count() * 1e-9 << endl;

    // for (int i=0; i<5; ++i) // вывод посчитанного массива
    //     printf("%f\n", X[i]);

    cudaFree(X); // освобождение памяти
    cudaFree(X_next);
    return 0;
}