int a[2];

int (&&f())[2]
{
  return static_cast<int(&&)[2]>(a);
}
