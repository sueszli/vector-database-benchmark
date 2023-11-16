#include "xor_model.h" // emlearn generated model

#include <stdio.h> // printf
#include <stdlib.h> // stdod

int
main(int argc, const char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Wrong number of arguments. Usage: xor_model A-number B-number \n");
        return -1;
    }

    const float a = strtod(argv[1], NULL);
    const float b = strtod(argv[2], NULL);
    const float features[] = { a, b };

    int out = xor_model_predict(features, 2); // Alternative A: "inline"
    out = eml_trees_predict(&xor_model, features, 2); // Alternative B: "loadable"
    if (out < 0) {
        return out; // error
    } else {
        printf("%d\n", out);
    }
    return 0;
}
