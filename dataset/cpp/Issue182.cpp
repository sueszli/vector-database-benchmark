typedef int my_cb(int dst, int len, int dat);
int f(my_cb cb, void *dat);

int f(my_cb cb, void *dat)
{
	return 0;
}
