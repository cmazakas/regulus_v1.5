#include "structures.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Incorrect number of arguments. Only enter the length of the Cartesian grid.\n");
        return 1;
    }
    
    int box_length = 0;

    char *end = 0;
    long int status = strtol(argv[1], &end, 10);
    if (status == 0L || status == LONG_MIN || status == LONG_MAX)
    {
        printf("Enter a more realistic number\n");
        return 1;
    }

    box_length = status;

    /* ----------------------------- */

    mesh m;
    m.create_input(box_length); // generate Cartesian set
    m.sort_by_peanokey(); // sort points by spatial locality
    m.triangulate(); // triangulate the point set

    return 0;
}
