#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void csvline_populate(vector<string> &record, const string& line, char delimiter);
void  GetString( const string& Str, vector<string> &Value );


int main(int argc, char *argv[])
{
	
    vector<string> row;
    vector<string> label;
    string line;
    string temp_str;
    ifstream in("sour_lable.csv"); //�����ļ��� 
    if (in.fail())  { cout << "File not found" <<endl; return 0; }
     long t=1;
    while(getline(in, line)  && in.good() )
    {
          csvline_populate(row, line, ',');
          cout<< t<<"\t";
          cout<< row[1] << "\t";
          temp_str=row[1];
         // char* c;
         // const int len = temp_str.length();
         // c =new char[len+1];
         // strcpy(c,temp_str.c_str());
          //cout<<temp_str.size()<<"|";
         // char *tokenPtr=strtok(c,"");
          GetString(temp_str,label);
          //vector<string>::iterator it;
          //cout<<label.size()<<"|";
         for (int it=0;it<label.size();it++)
           cout<<label[it]<<"|";
         // while(tokenPtr!=NULL)��{
          //    cout<<tokenPtr<<"|";
          //    tokenPtr=strtok(NULL,"");
          //      }
          cout << endl;
          t++;
    
    }
    in.close();
    return 0;
}

void csvline_populate(vector<string> &record, const string& line, char delimiter)
{
    int linepos=0;               //�ַ��������� 
    int inquotes=false;          //���� 
    char c;                      //�ַ�����ÿ��Ԫ�� char 
    int i;
    int linemax=line.length();   //ÿ���ַ������ܳ��� 
    string curstring;
    record.clear();
       
    while(line[linepos]!=0 && linepos < linemax)//�ַ�����ÿ��Ԫ�� 
    {
       
        c = line[linepos];       //ÿ���ַ� 
       
        if (!inquotes && curstring.length()==0 && c=='"')//��һ������ 
        {
            //beginquotechar
            inquotes=true;//������ 
        }
        else if (inquotes && c=='"')//��������� 
        {
            //quotechar
            if ( (linepos+1 <linemax) && (line[linepos+1]=='"') )//˫���� 
            {
                //encountered 2 double quotes in a row (resolves to 1 double quote)
                curstring.push_back(c);
                linepos++;
            }
            else
            {
                //endquotechar
                inquotes=false;
            }
        }
        else if (!inquotes && c==delimiter)
        {
            //end of field
            record.push_back( curstring );
            curstring="";
        }
        else if (!inquotes && (c=='\r' || c=='\n') )
        {
            record.push_back( curstring );
            return;
        }
        else
        {
            curstring.push_back(c);
        }
        linepos++;
    }
    record.push_back( curstring );
    return;
}

//��ȡ���ַ��������ݣ����ݿո��ֲ�����������ȥ��
//Str��Դ�ַ�����Value��Ÿ���Ŀ���ַ����Ŀո���ȡ������ÿ���ַ������ݡ�
void  GetString( const string& Str, vector<string> &Value )
{
   string Temp="";                          //�ָ���ַ��� 
   int strlength=Str.length();              //Դ�ַ������� 
   int Index= 0;
   Value.clear();
   char  Tmp=' '; 
   while(Str[Index]!=0 && Index < strlength)//�ַ�����ÿ��Ԫ�� 
    {
      Tmp = Str[Index];               //�ַ����е�ÿ���ַ� 
      if( Tmp==' '|| Index == strlength-1)  //�ո�Ļ����ߵ����һ���ַ� 
      {
         if( Index == strlength-1) Temp.push_back(Tmp);
         if(Temp!="")       //�����ո��ˣ���� Temp���ַ��Ļ��Ͱ�Temp���������� 
         {
         	if (Temp=="10") Temp="A";
         	if (Temp=="11") Temp="B";
         	if (Temp=="12") Temp="C";
         	if (Temp=="13") Temp="D";
         	if (Temp=="14") Temp="E";
         Value.push_back(Temp);
         Temp="";                    //�����һ�α������ʱ�ַ��� 
         }
        Index++; 
        continue;
      }
      else {
      Temp.push_back(Tmp);
      Index++; 
      continue;
	  }
   }
   return;
}
