#if defined __has_extension
#  if __has_extension(c_alignas)
      #error extension exists
#  endif
#  if __has_extension(does_not_exist)
      #error extension exists
#  endif
#endif

#define EXPECTED_ERRORS "__has_extension.c:3:8: error: extension exists"
