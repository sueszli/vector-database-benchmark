// https://mbevin.wordpress.com/2012/11/14/range-based-for/
class MyArrayWrapper {
   int* data;
   int size;

public:
   int* begin() { return size>0 ? &data[0] : nullptr; }
   int* end()   { return size>0 ? &data[size-1] : nullptr; }
};


int main() {
    MyArrayWrapper arr;
    for(int i: arr) {
   
    }

}
