namespace ExplicitClassTemplateSpecializationTest
{
  template<typename T>
  struct Base
  {
  };

  template<>
  struct Base<bool> {};

}

  template<typename T>
  struct Base
  {
  };

  template<>
  struct Base<bool> {};

int main()
{
}
