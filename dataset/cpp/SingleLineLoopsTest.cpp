void Foo() {}

int main()
{
  for(int i = 0; i < 2; ++i) Foo();
  
  int i=0;
  while( ++i < 5 ) Foo();

  do Foo(); while( ++i < 5 );
}
