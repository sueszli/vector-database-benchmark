#include <stdio.h>

class TwoPhaseCons 
{
private:
    TwoPhaseCons() // ��һ�׶ι��캯��
    {   
    }
    bool construct() // �ڶ��׶ι��캯��
    { 
        return true; 
    }
public:
    static TwoPhaseCons* NewInstance(); // ���󴴽�����
};

TwoPhaseCons* TwoPhaseCons::NewInstance() 
{
    TwoPhaseCons* ret = new TwoPhaseCons();

    // ���ڶ��׶ι���ʧ�ܣ����� NULL    
    if( !(ret && ret->construct()) ) 
    {
        delete ret;
        ret = NULL;
    }
        
    return ret;
}


int main()
{
    TwoPhaseCons* obj = TwoPhaseCons::NewInstance();
    
    printf("obj = %p\n", obj);

    delete obj;
    
    return 0;
}
