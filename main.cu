#include "structures.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Incorrect number of arguments. Only enter the length of the Cartesian grid.\n");
        return 1;
    }
    
    char *end = 0;
    long int status = strtol(argv[1], &end, 10);
    if (status == 0L || status == LONG_MIN || status == LONG_MAX)
    {
        printf("Enter a more realistic number\n");
        return 1;
    }

    const int box_length = status;

    regulus reg(box_length);

    reg.triangulate();

    return 0;
}
