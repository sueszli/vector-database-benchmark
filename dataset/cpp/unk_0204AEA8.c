#include "global.h"
#include "unk_0204AEA8.h"
#include "heap.h"
#include "task.h"
#include "unk_020040F4.h"

extern void ov05_021E3444(u32, struct FieldSystem *, u32);

BOOL sub_0204AEA8(struct TaskManager *taskManager)
{
    struct FieldSystem *fieldSystem = TaskManager_GetFieldSystem(taskManager);
    u32 *v1 = TaskManager_GetEnvironment(taskManager);

    switch (v1[0])
    {
        case 0:
            ov05_021E3444(v1[2], fieldSystem, (u32)v1 + 4);
            sub_0200433C(5, v1[3], 1);
            v1[0]++;
            break;
        case 1:
            if (v1[1] == 1)
            {
                FreeToHeap(v1);
                return TRUE;
            }
            break;
    }
    
    return FALSE;
}

void sub_0204AEF8(struct TaskManager *taskManager, u32 a1, u32 a2) //taskManager, effect, bgm
{
    u32 *v0 = AllocFromHeapAtEnd(HEAP_ID_FIELD, 4 * sizeof(u32)); //todo: make a struct
    
    v0[0] = 0; //state
    v0[1] = 0; //unk
    v0[2] = a1; //effect
    v0[3] = a2; //bgm
    
    TaskManager_Call(taskManager, sub_0204AEA8, v0);
}
