#include <iostream>
#include <cmath>

void add(int n, float *x, float *y)
{
    for(int i = 0; i < n; ++i)
    {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char** argv)
{
    int N = 1 << 20; //1M elements

    float *x = new float[N];
    float *y = new float[N];

    for( int i = 0; i < N; ++i)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add(N, x, y);

    //Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    
    for( int i = 0; i < N; ++i )
    {
        maxError = std::max(maxError, std::abs(y[i] - 3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    delete[] x;
    delete[] y;

    return 0;
}
