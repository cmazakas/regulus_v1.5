#include "structures.h"

associated_arrays::associated_arrays(const int num_cart_points)
{
    capacity = 8 * num_cart_points;
    size     = num_cart_points;

    pa        = thrust::device_malloc<int>(capacity);
    ta        = thrust::device_malloc<int>(capacity);
    fs        = thrust::device_malloc<int>(capacity);
    la        = thrust::device_malloc<int>(capacity);
    nominated = thrust::device_malloc<int>(capacity);

    thrust::fill(pa, pa + capacity, -1);
    thrust::fill(ta, ta + capacity, -1);
    thrust::fill(fs, fs + capacity, -1);
    thrust::fill(la, la + capacity, -1);
    thrust::fill(nominated, nominated + capacity, 0);
}

associated_arrays::~associated_arrays(void)
{
    thrust::device_free(pa);
    thrust::device_free(ta);
    thrust::device_free(fs);
    thrust::device_free(la);
    thrust::device_free(nominated);
};

void associated_arrays::resize(const int N)
{
    size = N;
}
void associated_arrays::print(void) const
{
    const int print_width = 4;

    for (int i = 0; i < size; ++i)
    {
        std::cout << "{ pa : " << std::setw(print_width) << pa[i] << ", ta : " << std::setw(print_width) << ta[i] << ", fs : " << std::setw(print_width) << fs[i] << ", la : " << std::setw(print_width) << la[i] << " }" << std::endl;
    }
}

void associated_arrays::print_with_nominated(void) const
{
    const int print_width = 4;

    for (int i = 0; i < size; ++i)
    {
        std::cout << "{ pa : " << std::setw(print_width) << pa[i] << ", ta : " << std::setw(print_width) << ta[i] << ", fs : " << std::setw(print_width) << fs[i] << ", la : " << std::setw(print_width) << la[i] << ", nominated : " << nominated[i] << " }" << std::endl;
    }
}
