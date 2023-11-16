#include "SystemClass.h"

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    SystemClass* systemClass = new SystemClass();

    if (!FAILED(systemClass->InitWindow(nCmdShow)))
    {
        systemClass->Run();
    }

    systemClass->Shutdown();
    delete systemClass;
    systemClass = nullptr;

    return 0;
}