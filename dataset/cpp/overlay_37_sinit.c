#include "global.h"

extern void Poketch_InitApp(void *func1, void *func2);
extern void ov37_02254854();
extern void ov37_02254934();

static void ov37_02254840(void)
{
    Poketch_InitApp(ov37_02254854, ov37_02254934);
}

#define NitroStaticInit ov37_02254840
#include "sinit.h"
