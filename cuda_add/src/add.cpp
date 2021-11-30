#include <iostream>
#include <cmath>

// Vector element type
using vector_t = float;

void add(int n, vector_t *x, vector_t *y)
{
    for(int i = 0; i < n; ++i)
    {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char** argv)
{
    int N = 1 << 20; //1M elements

    vector_t *x = new vector_t[N];
    vector_t *y = new vector_t[N];

    for(int i = 0; i < N; ++i)
    {
        x[i] = vector_t(1);
        y[i] = vector_t(2);
    }

    add(N, x, y);

    //Check for errors (all values should be 3.0)
    vector_t maxError = vector_t(0);
    
    for( int i = 0; i < N; ++i )
    {
        maxError = std::max(maxError, std::abs(y[i] - vector_t(3)));
    }

    std::cout << "Max error: " << maxError << std::endl;

    delete[] x;
    delete[] y;

    return 0;
}
