struct templateTypeAliasTest {
  template<typename T>
  using result = T;
};

int main()
{
    templateTypeAliasTest::result<int> x;

}
