extern "C" void Foo();

int main()
{

    Foo();
}

extern "C" {
    void Foo() {}
}
