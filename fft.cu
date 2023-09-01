#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// parameters
#define PI 3.14159265

// functions
__global__ void fft2(float* xr, float* xi, int steps) {
    int L = 1;
    double wr;
    double wi;
    double xrt1;
    double xit1;
    double br[2];
    double bi[2];
    //__shared__ float xr[2048];
    //__shared__ float xi[2048];

    int zero;
    int one;
    int q;
    int idx = tx + bx * 1024 + by * 32768 * 1024;

    /*xr[2 * tx] = x_r_d[2 * tx + bx * 2048 + by * 32768 * 2048];
    xi[2 * tx] = x_i_d[2 * tx + bx * 2048 + by * 32768 * 2048];

    xr[2 * tx + 1] = x_r_d[2 * tx + 1 + bx * 2048 + by * 32768 * 2048];
    xi[2 * tx + 1] = x_i_d[2 * tx + 1 + bx * 2048 + by * 32768 * 2048];

    __syncthreads();*/

    for (int i = 0; i < steps; ++i) {
        q = idx % L;
        zero = (idx / L) * (2 * L) + q;
        one = (idx / L) * (2 * L) + q + 1 * L;
        
        wr = cos(q * 2 * PI / (2 * L));
        //wi = -sin(q * 2 * PI / (2 * L));
        wi = -sqrt(1 - wr * wr);

        xrt1 = xr[one] * wr - xi[one] * wi;
        xit1 = xi[one] * wr + xr[one] * wi;


        br[0] = xr[zero] + xrt1;
        bi[0] = xi[zero] + xit1;

        br[1] = xr[zero] - xrt1;
        bi[1] = xi[zero] - xit1;


        xr[zero] = br[0];
        xi[zero] = bi[0];

        xr[one] = br[1];
        xi[one] = bi[1];

        L = L * 2;
        __syncthreads();
    }
    
    /*x_r_d[2 * tx + bx * 2048 + by * 32768 * 2048] = xr[2 * tx];
    x_i_d[2 * tx + bx * 2048 + by * 32768 * 2048] = xi[2 * tx];

    x_r_d[2 * tx + 1 + bx * 2048 + by * 32768 * 2048] = xr[2 * tx + 1];
    x_i_d[2 * tx + 1 + bx * 2048 + by * 32768 * 2048] = xi[2 * tx + 1];*/

}


__global__ void onestepfft2(float* xr, float* xi, int step) {
    int L = 1 << step;
    double wr;
    double wi;
    double xrt1;
    double xit1;
    double br[2];
    double bi[2];

    int zero;
    int one;
    int q;
    int idx = tx + bx * 1024 + by * 32768 * 1024;

    q = idx % L;
    zero = (idx / L) * (2 * L) + q;
    one = (idx / L) * (2 * L) + q + 1 * L;

    wr = cos(q * 2 * PI / (2 * L));
    //wi = -sin(q * 2 * PI / (2 * L));
    wi = -sqrt(1 - wr * wr);

    xrt1 = xr[one] * wr - xi[one] * wi;
    xit1 = xi[one] * wr + xr[one] * wi;


    br[0] = xr[zero] + xrt1;
    bi[0] = xi[zero] + xit1;

    br[1] = xr[zero] - xrt1;
    bi[1] = xi[zero] - xit1;


    xr[zero] = br[0];
    xi[zero] = bi[0];

    xr[one] = br[1];
    xi[one] = bi[1];

}


__global__ void bitrev2(float* xr, float* xi, int bits) {
    float tmpr;
    float tmpi;
    long long idx;
    long long revidx;
    idx = tx + bx * 1024 + by * 32768 * 1024;
    revidx = 0;

    for (int j = 0; j < bits / 2; ++j) {
        revidx |= (idx & (1 << j)) << (bits - 1 - 2 * j);
        revidx |= (idx & (1 << (bits - 1 - j))) >> (bits - 1 - 2 * j);
    }
    if (bits % 2 == 1) {
        revidx |= idx & (1 << (bits / 2));
    }

    if (revidx > idx) {
        tmpr = xr[revidx];
        tmpi = xi[revidx];

        xr[revidx] = xr[idx];
        xi[revidx] = xi[idx];

        xr[idx] = tmpr;
        xi[idx] = tmpi;
    }
}


__global__ void fft4(float* x_r_d, float* x_i_d, int steps) {
    int L = 1;
    double wr1, wr2, wr3;
    double wi1, wi2;
    double xrt[4];
    double xit[4];
    double br[4];
    double bi[4];
    __shared__ float xr[1024];
    __shared__ float xi[1024];

    int zero;
    int one;
    int two;
    int three;
    int q;
    int idx = tx + bx * 256 + by * 32768 * 256;

    xr[4 * tx] = x_r_d[4 * idx];
    xi[4 * tx] = x_i_d[4 * idx];

    xr[4 * tx + 1] = x_r_d[4 * idx + 1];
    xi[4 * tx + 1] = x_i_d[4 * idx + 1];

    xr[4 * tx + 2] = x_r_d[4 * idx + 2];
    xi[4 * tx + 2] = x_i_d[4 * idx + 2];

    xr[4 * tx + 3] = x_r_d[4 * idx + 3];
    xi[4 * tx + 3] = x_i_d[4 * idx + 3];

    __syncthreads();

    for (int i = 0; i < steps; ++i) {
        q = idx % L;
        zero = (tx / L) * (4 * L) + q;
        one = (tx / L) * (4 * L) + q + 1 * L;
        two = (tx / L) * (4 * L) + q + 2 * L;
        three = (tx / L) * (4 * L) + q + 3 * L;


        xrt[0] = xr[zero];
        xit[0] = xi[zero];

        wr1 = cos(q * 2 * PI / (4 * L));
        wi1 = -sqrt(1 - wr1 * wr1);
        xrt[1] = xr[one] * wr1 - xi[one] * wi1;
        xit[1] = xi[one] * wr1 + xr[one] * wi1;

        wr2 = 2 * wr1 * wr1 - 1;
        wi2 = 2 * wr1 * wi1;
        xrt[2] = xr[two] * wr2 - xi[two] * wi2;
        xit[2] = xi[two] * wr2 + xr[two] * wi2;

        wr3 = wr1 * wr2 - wi1 * wi2;
        wi2 = wi2 * wr1 + wi1 * wr2;
        xrt[3] = xr[three] * wr3 - xi[three] * wi2;
        xit[3] = xi[three] * wr3 + xr[three] * wi2;


        br[0] = xrt[0] + xrt[1] + xrt[2] + xrt[3];
        bi[0] = xit[0] + xit[1] + xit[2] + xit[3];

        br[1] = xrt[0] + xit[1] - xrt[2] - xit[3];
        bi[1] = xit[0] - xrt[1] - xit[2] + xrt[3];

        br[2] = xrt[0] - xrt[1] + xrt[2] - xrt[3];
        bi[2] = xit[0] - xit[1] + xit[2] - xit[3];

        br[3] = xrt[0] - xit[1] - xrt[2] + xit[3];
        bi[3] = xit[0] + xrt[1] - xit[2] - xrt[3];


        xr[zero] = br[0];
        xi[zero] = bi[0];

        xr[one] = br[1];
        xi[one] = bi[1];

        xr[two] = br[2];
        xi[two] = bi[2];

        xr[three] = br[3];
        xi[three] = bi[3];

        L = L * 4;
        __syncthreads();

    }

    x_r_d[4 * idx] = xr[4 * tx];
    x_i_d[4 * idx] = xi[4 * tx];

    x_r_d[4 * idx + 1] = xr[4 * tx + 1];
    x_i_d[4 * idx + 1] = xi[4 * tx + 1];

    x_r_d[4 * idx + 2] = xr[4 * tx + 2];
    x_i_d[4 * idx + 2] = xi[4 * tx + 2];

    x_r_d[4 * idx + 3] = xr[4 * tx + 3];
    x_i_d[4 * idx + 3] = xi[4 * tx + 3];

}


__global__ void onestepfft4(float* xr, float* xi, int step) {
    int L = 1 << step;
    double wr1, wr2, wr3;
    double wi1, wi2;
    double xrt[4];
    double xit[4];
    double br[4];
    double bi[4];

    int zero;
    int one;
    int two;
    int three;
    int q;
    int idx = tx + bx * 256 + by * 32768 * 256;

    q = idx % L;
    zero = (idx / L) * (4 * L) + q;
    one = (idx / L) * (4 * L) + q + 1 * L;
    two = (idx / L) * (4 * L) + q + 2 * L;
    three = (idx / L) * (4 * L) + q + 3 * L;


    xrt[0] = xr[zero];
    xit[0] = xi[zero];

    wr1 = cos(q * 2 * PI / (4 * L));
    wi1 = -sqrt(1 - wr1 * wr1);
    xrt[1] = xr[one] * wr1 - xi[one] * wi1;
    xit[1] = xi[one] * wr1 + xr[one] * wi1;

    wr2 = 2 * wr1 * wr1 - 1;
    wi2 = 2 * wr1 * wi1;
    xrt[2] = xr[two] * wr2 - xi[two] * wi2;
    xit[2] = xi[two] * wr2 + xr[two] * wi2;

    wr3 = wr1 * wr2 - wi1 * wi2;
    wi2 = wi2 * wr1 + wi1 * wr2;
    xrt[3] = xr[three] * wr3 - xi[three] * wi2;
    xit[3] = xi[three] * wr3 + xr[three] * wi2;


    br[0] = xrt[0] + xrt[1] + xrt[2] + xrt[3];
    bi[0] = xit[0] + xit[1] + xit[2] + xit[3];

    br[1] = xrt[0] + xit[1] - xrt[2] - xit[3];
    bi[1] = xit[0] - xrt[1] - xit[2] + xrt[3];

    br[2] = xrt[0] - xrt[1] + xrt[2] - xrt[3];
    bi[2] = xit[0] - xit[1] + xit[2] - xit[3];

    br[3] = xrt[0] - xit[1] - xrt[2] + xit[3];
    bi[3] = xit[0] + xrt[1] - xit[2] - xrt[3];


    xr[zero] = br[0];
    xi[zero] = bi[0];

    xr[one] = br[1];
    xi[one] = bi[1];

    xr[two] = br[2];
    xi[two] = bi[2];

    xr[three] = br[3];
    xi[three] = bi[3];

}


__global__ void bitrev4(float* xr, float* xi, int bits) {
    float tmpr;
    float tmpi;
    long long idx;
    long long revidx;
    idx = tx + bx * 1024 + by * 32768 * 1024;
    revidx = 0;
    
    for (int j = 0; j < bits / 2; ++j) {
        revidx |= (idx & (3 << (2 * j))) << (2 * (bits - 1 - 2 * j));
        revidx |= (idx & (3 << (2 * (bits - 1 - j)))) >> (2 * (bits - 1 - 2 * j));
    }
    if (bits % 2 == 1) {
        revidx |= idx & (3 << (2 * (bits / 2)));
    }
    
    if (revidx > idx) {
        tmpr = xr[revidx];
        tmpi = xi[revidx];

        xr[revidx] = xr[idx];
        xi[revidx] = xi[idx];

        xr[idx] = tmpr;
        xi[idx] = tmpi;
    }
}


__global__ void mixedbitrev42(float* xr, float* xi, float* xr_cpy, float* xi_cpy, int bits) {
    long long idx;
    long long revidx;
    idx = tx + bx * 1024 + by * 32768 * 1024;
    revidx = 0;

    for (int j = 0; j < bits / 2; ++j) {
        revidx |= (idx & (3 << (2 * j))) << (2 * (bits - 1 - 2 * j));
        revidx |= (idx & (3 << (2 * (bits - 1 - j)))) >> (2 * (bits - 1 - 2 * j));
    }
    if (bits % 2 == 1) {
        revidx |= idx & (3 << (2 * (bits / 2)));
    }

    revidx = revidx << 1;
    revidx |= (idx & (1 << (bits * 2))) >> (bits * 2);

    xr[idx] = xr_cpy[revidx];
    xi[idx] = xi_cpy[revidx];
}


__global__ void arraycpy(float* xr, float* xi, float* xr_cpy, float* xi_cpy) {
    long long idx;
    idx = tx + bx * 1024 + by * 32768 * 1024;

    xr_cpy[idx] = xr[idx];
    xi_cpy[idx] = xi[idx];
}


//-----------------------------------------------------------------------------
void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M)
{
    float* xr_cpy;
    float* xi_cpy;


    dim3 dimGrid1(N / 1024 > 65535 ? 32768 : (N < 1024 ? 1 : N / 1024), N / 1024 > 65535 ? N / 1024 / 32768 : 1);
    dim3 dimBlock1(N > 1024 ? 1024 : N);

    dim3 dimGrid2(N / 2048 > 65535 ? 32768 : (N < 2048 ? 1 : N / 2048), N / 2048 > 65535 ? N / 2048 / 32768 : 1);
    dim3 dimBlock2(N / 2 > 1024 ? 1024 : N / 2);

    dim3 dimGrid3(N / 1024 > 65535 ? 32768 : (N < 1024 ? 1 : N / 1024), N / 1024 > 65535 ? N / 1024 / 32768 : 1);
    dim3 dimBlock3(N / 4 > 256 ? 256 : N / 4);


    if (M % 2 == 0) {
        bitrev4 <<<dimGrid1, dimBlock1>>> (x_r_d, x_i_d, M / 2);

        if (M < 12) {
            fft4 <<<dimGrid3, dimBlock3>>> (x_r_d, x_i_d, M / 2);
        }
        else {
            fft4 <<<dimGrid3, dimBlock3>>> (x_r_d, x_i_d, 10 / 2);
            for (unsigned int i = 10; i < M; i = i + 2) {
                onestepfft4 <<<dimGrid3, dimBlock3>>> (x_r_d, x_i_d, i);
            }
        }
    }
    else {
        HANDLE_ERROR(cudaMalloc((void**)&xr_cpy, N * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void**)&xi_cpy, N * sizeof(float)));
        arraycpy <<<dimGrid1, dimBlock1>>> (x_r_d, x_i_d, xr_cpy, xi_cpy);
        mixedbitrev42 <<<dimGrid1, dimBlock1>>> (x_r_d, x_i_d, xr_cpy, xi_cpy, (M - 1) / 2);
        HANDLE_ERROR(cudaFree(xr_cpy));
        HANDLE_ERROR(cudaFree(xi_cpy));

        if (M < 10) {
            fft4 <<<dimGrid3, dimBlock3>>> (x_r_d, x_i_d, M / 2);
            onestepfft2 <<<dimGrid2, dimBlock2>>> (x_r_d, x_i_d, M - 1);
        }
        else {
            fft4 <<<dimGrid3, dimBlock3>>> (x_r_d, x_i_d, 10 / 2);
            for (unsigned int i = 10; i < M - 1; i = i + 2) {
                onestepfft4 <<<dimGrid3, dimBlock3>>> (x_r_d, x_i_d, i);
            }
            onestepfft2 <<<dimGrid2, dimBlock2>>> (x_r_d, x_i_d, M - 1);
        }

    }

}
