int main() {
    char buffer[2];

    [&]() {
        auto &[a, b] = buffer;
    }();
}

#if 0
int main() {
    char buffer[2];


     class __lambda_4
     {
       public: inline /*constexpr */ void operator()() const
       {
         auto __buffer5 = buffer;
         char a = __buffer5[0];
         char b = __buffer5[1];
       }
       
       private:
       char (&buffer)[2];
       
       public: __lambda_4(char (&_buffer)[2])
       : buffer{_buffer}
       {}
       
     };
     
         __lambda_4{buffer}();
}
#endif
