#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: memory-user <memory> <time>\n");
        exit(EXIT_FAILURE);
    }

    printf("pid: %d\n", getpid());

    int memory = atoi(argv[1]) * 1024 * 1024;
    int length = (int)(memory / sizeof(int));
    int runTime = atoi(argv[2]);
    int *arr = malloc(memory);
    clock_t begin = clock();
    double time_spent;

    while (1)
    {
        time_spent = (double)(clock() - begin) / CLOCKS_PER_SEC;
        if (time_spent >= runTime)
            break;
        for (int i = 0; i < length; i++)
            arr[i] += 1;
    }

    free(arr);
    return 0;
}
