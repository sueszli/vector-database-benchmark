int Open() { return 0; }
int Write() { return 0; }
#define SUCCESS 1


auto Foo()
{
    if( auto ret = Open(); SUCCESS != ret )
    {
        return ret;
    } else if( auto ret = Write(); SUCCESS != ret ) {
        return ret;
    }

    // ...
    
    return SUCCESS;
}


void Fun()
{
    if(Open(); true) {}

    switch(Open(); 1) {
        default: break;
    }
}


int main()
{
    Foo();
}

