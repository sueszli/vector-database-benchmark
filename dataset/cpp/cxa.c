/* Register a function to be called by exit or when a shared library
   is unloaded.  This function is only called from code generated by
   the C++ compiler.  */
int __cxa_atexit(void (*func)(void *), void *arg, void *d)
{
    // todo: 使用rust实现这个函数。参考：http://redox.longjin666.cn/xref/relibc/src/cxa.rs?r=c7d499d4&mo=323&fi=15#15
    return 0;
}
