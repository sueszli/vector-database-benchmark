/*-----------------------------------
����    ��   C���Ի�ԭQQ �����ܰ桿 
����    ��   404name / ����N CTGU_LLZ
���ʱ�䣺   2021.11.22 
˵��    ��   �������������ʵ�ֵģ����ʺ�ѧϰ�����Կ�����ƣ��������100������100��ʵ�ַ�ʽ������
��������� + �㷨������ͳһ��ģ�塣����߶Ȼ�ԭ��qq��ͨ��wads���ƹ���ƶ����ո����ȷ�ϣ��س�����
���������˳���q����ҳ��������ٿء�

����ɽ��棺[QQ��ҳ][QQ�ѻ���][QQ��¼ҳ]   [��ϵ�˺Ϳռ��Ǿ�̬��] 
֧�ֲ�������¼/ע��  ��������+������Ϣ+��ѯ�������+�洢��Ϣ��¼
�������Ϊ [demo��][���ݰ���MD5���ܰ�]  
	  
	  
	���򲿷ֺ�����������˵�� 
	 appMap[20][12][7] : ����棬��ʾ��20��12*7�Ľ��� ������ƺ��ر���Ҫ����˵��
	 					�����Ȳ����� 
	 appMapMessage[100][5] ����ȡ����Ӧ�����ݵľ�����Ϣ 
	 appMapIcon[100][1000]������� 
	  
	��Ҫ�������� 
	printDemo(index,bool);  html��ҳ���������ƶ�����������ƿ��ĸ���ú������������״̬��
	checkAction �����Ƽ�����������·����תȫ���Ǳ����� 
	move���������ƶ� 
-----------------------------------*/

#include <conio.h>  //getch()
#include <stdio.h>
#include <stack>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <windows.h>
#define Is_Active_Color 115     //��ǰ����״̬������ɫ ǳ��ɫ�ҵ�
#define Is_Selected1_Color 128  //��ѡ�е���ɫ1 ��ɫ��ҵ�
#define Is_Selected2_Color 62   //��ѡ�е���ɫ2 ��ɫ����
#define Is_Selected3_Color \
    369  //��ѡ�е���ɫ3 ����ɫ��ɫ(���һЩ�ַ�����ʾ������)
#define Video_Color 3     //��Ƶ��ɫ ��ɫ�ڵ�
#define Normal_Color 112  //��������ɫ �ҵ�
#define Boder_Color 391   //�߿����ɫ ��ҵ�
using namespace std;
const int WIDTH = 53;     //��λ���
const int HEIGHT = 47;    //��λ�߶�
const int L = 4;          //�û�ͷ�񳤿�
const int LL = 20;        //��Ƶ���泤��
const int N = 50;         //���⡢�û����������
const int M = 1000000;    //��Ƶ�ղأ���������˿������
/*appMap������ӳ�䣬ȡ��������ֵ��Ӧ����ľ�����Ϣ
0�߽� 1��ͷ��(�����¼) 2������ 3��Ϣ
4 5 6 7 8 ֱ��/�Ƽ�/����/׷��/Ӱ��
9-14 6����Ƶ��
15-19 ��ҳ/�Ƽ�/��̬/��Ա��/�ҵ�
20 404ҳ��  */
/*ȫ�ֱ���
nowPage 0 1 2 3 404ҳ��/��ҳ/����/�ҵ�/����ҳ��/������
*/
int nowPage = 4;
int indexCnt = 0;// ��ҳ�ж�������Ϣ��¼ 
int startItem = 0,endItem = 0;
bool onPlay = false;   //�ر�����ɱ��
bool stopPlay = true;  //��������ֹͣ
bool isLogin = false;
stack<int> pageStack; 
int activeIndexBottom = 17;  //��ǰ��Ӧ��ҳ��(����
int activeIndexTop = 5;      //��ǰ��Ӧ��ҳ��(�ײ�
string username = "����N";
string pwd = "123456"; 

char bacground[48][55] = {
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                      i.,ii;;ii;  .: .i7:            ",
"                    :;iiir7i;X;XXX;i  :i,            ",
"                  7iii;;:iS;::iX7XS  ..              ",
"                 XSr,::i,r;i:i;::XXXi,               ",
"                .Srr;::,rr:.:iii;7X2S                ",
"                ;;:;r:iSSr;,r7;i;iXaa                ",
"                7r;rXX22XrrrXX2rr;22Z                ",
"                ;r;XXSS2XXr7XX: ,XS2S     ::         ",
"                ;r;XXXXXSSS7X:  :Xa@Z                ",
"               .7rXSXSXSXXS2S7S7r:7MMS               ",
"              SZ222S2SXS2X2aXS0a2XSZ8ZS              ",
"             0WZ20a2X2SaXX22aa2aXSSXSX2W, ,.         ",
"            XM8SSa2XaSSXXZZSa22X7aXS2aaMM  .         ",
"            aZXSXSXaSSXXXZ222S;XrZ82S22WMZ           ",
"           ,BXX8SXXXXXXSX2ZaZa27rZa7XXSWMM           ",
"           i8BZX28XXXX2arZZaaZ882araZXWMMW           ",
"            rr.  80ZZZ8a20BZ888ZaZ80r 7X0            ",
"                 iB008ZZaa88S: .XXWZ:                ",
"               7@8ZZZaZZ880ZXirXXraMMW               ",
"               SM228XZMM@BSZ8MM8Z8WMM0     .         ",
"              . r.;:ir2a,    .i::. :                 ",
"            i,.                                      ",
"            .. ,                     .               ",
"                                                     ",
"                     .r..;. :i.,;                    ",
"                     2    S:X   ,X                   ",
"                     7:  X7 X . Si                   ",
"                      ii:r:  ;:i7.                   ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     ",
"                                                     "
};
//��ά������   �ĸ����� ��������  ĳ�еڼ���ģ��
// appMap[x][0][0]��ʾ��ҳ���м���
int appMap[20][12][7] = {{{5, 0, 0, 0, 0, 0, 0},  // 0.404����ҳ��
                          {0, 1, 2, 2, 2, 3, 0},
                          {0, 20, 20, 20, 20, 20, 0},
                          {0, 17, 17, 18, 19, 19, 0},
                          {0, 0, 0, 0, 0, 0, 0}},
                         {{11, 0, 0, 0, 0, 0, 0},  // 1.��ҳ
                          {0, 1, 2, 2, 2, 3, 0},  
                          {0, 9, 9, 9, 9, 9, 0},
                          {0, 10, 10, 10, 10, 10, 0},
                          {0, 11, 11, 11, 11, 11, 0},
                          {0, 12, 12, 12, 12, 12, 0},
                          {0, 13, 13, 13, 13, 13, 0},
                          {0, 14, 14, 14, 14, 14, 0},
                          {0, 15, 15, 15, 15, 15, 0},
                          {0, 17, 17, 18, 19, 19, 0},
                          {0, 0, 0, 0, 0, 0, 0}},
                         {{9, 0, 0, 0, 0, 0, 0},  // 2.��ϵ�� 
                          {0, 1, 2, 2, 2, 3, 0}, 
                          {0, 21, 21, 21, 21, 21, 0},
                          {0, 22, 22, 22, 22, 22, 0},
                          {0, 23, 23, 23, 23, 23, 0},
                          {0, 4, 5, 6, 7, 8, 0},
                          {0, 24, 24, 24, 24, 24, 0},
                          {0, 17, 17, 18, 19, 19, 0},
                          {0, 0, 0, 0, 0, 0, 0}},
                         {{7, 0, 0, 0, 0, 0, 0},  // 3.��̬ 
                          {0, 1, 2, 2, 2, 3, 0},
                          {0, 25, 25, 26, 27, 28, 0},
                          {0, 29, 29, 29, 29, 29, 0},
                          {0, 30, 30, 30, 30, 30, 0},
                          {0, 17, 17, 18, 19, 19, 0},
                          {0, 0, 0, 0, 0, 0, 0}},
                         {{7, 0, 0, 0, 0, 0, 0},  // 4.��¼ע��ҳ�� (������ )
                          {0, 31, 31, 31, 31, 31, 0},
                          {0, 32, 32, 32, 32, 32, 0},
                          {0, 33, 33, 33, 33, 33, 0},
                          {0, 34, 34, 34, 34, 34, 0},
                          {0, 35, 35, 36, 37, 37, 0},
                          {0, 0, 0, 0, 0, 0, 0}},
                         {{12, 0, 0, 0, 0, 0, 0},  // 5.��Ϣҳ�� (������ )
                          {0, 47, 47, 47, 47, 47, 0},
                          {0, 38, 38, 38, 38, 38, 0},
                          {0, 39, 39, 39, 39, 39, 0},
                          {0, 40, 40, 40, 40, 40, 0},
                          {0, 41, 41, 41, 41, 41, 0},
                          {0, 42, 42, 42, 42, 42, 0},
                          {0, 43, 43, 43, 43, 43, 0},
                          {0, 44, 44, 44, 44, 44, 0},
                          {0, 45, 45, 45, 45, 45, 0},
                          {0, 46, 46, 46, 46, 46, 0},
                          {0, 0, 0, 0, 0, 0, 0}}};
/*appMapȡ�������ݶ�Ӧ������±꣬
��Ϣ������ӳ�䵽ʵ�ʴ�ӡ���� x/y/��/��/���
��� 0 �޹����� �������ȥ
           1 �й����� �������action
           2 ������
*/
int appMapMessage[100][5] = {
    {0, 0, 0, 0, 0},
    {1, 2, 2, 40, 1},  // 1-3 ������Ϣ�� ͷ��/������/��Ϣ
    {3, 2, 3, 49, 1},
    {1, 43, 1, 8, 1},
    {21, 2, 1, 9, 1},   // 4-8 ������ǩ������
    {21, 12, 1, 9, 1},  //�Ƽ�
    {21, 22, 1, 9, 1},  //����
    {21, 32, 1, 9, 1},  //
    {21, 42, 1, 9, 1},	
    {6, 1, 6, 51, 1},  // 9-14 6����Ϣ��¼�� 
    {11, 1, 4, 51, 1},
    {16, 1, 4, 51, 1},
    {21, 1, 4, 51, 1},
    {26, 1, 4, 51, 1},
    {31, 1, 4, 51, 1},
    {35, 1, 7, 51, 1}, // 15 17 �ײ���ǩ���� 
    {42, 1, 4, 9, 1},
    {42, 2, 4, 9, 1},  // 17-19 �ײ���ǩ������
    {42, 22, 4, 9, 1},
    {42, 40, 4, 9, 1},
    {6, 2, 35, 49, 0},  // 20.404ҳ��
    {7, 2, 7, 49, 1},  // 21-24. top���Ű�ҳ��
    {14, 2, 3, 49, 1},
    {17, 2, 3, 49, 1},
    {23, 2, 15, 49, 1},
    {6, 4, 4, 10, 1},  // 25-28 �м�4���������� ��Ϣ����
    {6, 16, 4, 10, 1},
    {6, 27, 4, 10, 1},
    {6, 39, 4, 10, 1},
    {10, 1, 11, 51, 1},  // 29-30 qq�ռ� 
    {23, 1, 17, 51, 1},  
    {8, 20, 4, 18, 1},//31-37 ��¼���� 
	{14, 2, 3, 49, 1},
    {17, 2, 3, 49, 1},
    {22, 20, 2, 14, 1}, 
    {43, 2, 1, 14, 1},  // 35-37 �ײ���ǩ������
    {43, 20, 1, 14, 1}, 
    {43, 38, 1, 14, 1}, 
    {3, 1, 4, 51, 1},  // 38-45 6����Ϣ��¼�� 
    {8, 1, 4, 51, 1},
    {13, 1, 4, 51, 1},
    {18, 1, 4, 51, 1},
    {23, 1, 4, 51, 1},
    {28, 1, 4, 51, 1},
    {33, 1, 4, 51, 1},
    {38, 1, 4, 51, 1},
    {42, 1, 3, 51, 1},  //46 ����� 
    {1, 1, 2, 51, 1}
};
char appMapIcon[100][1000] = {
    " ",
    "�����ǡ� �û���                         "
	"��ͷ�� QQ for Cmd���� >               ",  // 1-5
    "._______________________________________________."
    "|                    o ����                     |"
    "|_____________________\\_________________________|",
    " ���(+)",
    "�����ѡ� ",
    "�����顽 ",
    "��Ⱥ�ġ� ",  // 6-10
    "���豸�� ", 
    "��ͨѶ�� ",
    "._________________________________________________."//9-14
    "|  Img | ��ʱû����Ϣ                             |"
    "|  404 |                                          |"
    "|______|__________________________________________|",
    "._________________________________________________."
    "|  Img | ��ʱû����Ϣ                             |"
    "|  404 |                                          |"
    "|______|__________________________________________|",
    "._________________________________________________."
    "|  Img | ��ʱû����Ϣ                             |"
    "|  404 |                                          |"
    "|______|__________________________________________|",
    "._________________________________________________."
    "|  Img | ��ʱû����Ϣ                             |"
    "|  404 |                                          |"
    "|______|__________________________________________|",
    "._________________________________________________."
    "|  Img | ��ʱû����Ϣ                             |"
    "|  404 |                                          |"
    "|______|__________________________________________|",
    "._________________________________________________."
    "|  Img | ��ʱû����Ϣ                             |"
    "|  404 |                                          |"
    "|______|__________________________________________|",

    "._________________________________________________."
    "|@Bվ  | ��ϵͳ���桿                   2021-11-18|"
    "|����N | - ϵͳ���ݣ����ڼ����㷨��QQ for Cmd (1) |"
    "|______| - ϵͳ�汾��1.0.0 perview            (2) |"
    "|Q:1308| - ������ߣ�404name(CTGU_LLZ)        (3) |"
    "|964967| - �漰���ݣ�RSA���ԳƼ��ܡ�MD5       (4) |"
    "|______|__________________________________________|",
	"   �X�D�["
	"   ���D��"
	"   |/    "
	"   ��ϵ��",  
	"   �X�D�["
	"   ���D��"
	"    |/   "
	"   ��Ϣ  ", // 17-20

	"    ()   "
    "    ��   " 
    "   (  )  "
	"  ��ϵ�� ",
    "    ��� "
	"   (  )  "
	"    ��   "
	"   ��̬  ",
    " ",
    "#################################################"
    "#������ʶ���� >                                 #"
    "#[IMG ] ����                         [���] x #"
    "#[404 ] �������ʶ                              #"
    "#[Chin] ������                         [���] x #"
    "#[a�� ] �������ʶ                              #"
    "#################################################",  // 21-25
    "._______________________________________________."
    "|������|                                      > |"
    "|______|________________________________________|",
    "._______________________________________________."
    "|Ⱥ֪ͨ|                                      > |"
    "|______|________________________________________|",
    "#################################################"
    "# > �ر����                              5/5 > #"
    "#                                               #"
    "# > �ҵĺ���                           50/100 > #"
    "#                                               #"
    "# > �ҵ�ͬѧ                           98/201 > #"
    "#                                               #"
    "# > ��������                          300/405 > #"
    "#                                               #"
    "#################################################"
	"                                                 "
	"                                                 "
	"                                                 "
	"                  QQ for Cmd                     "
	"                    V1.0.0                       ",
    " +------+ "
	" |::  ��| "
	" +------+ "
	"   ���   ",
	"   �X�D�[ "
	"   ���D�� "
	"    |/    "
	"   ˵˵   ",// 26-30
	"    ��    "
	"  ����    "
	"  ������  "
	" ��Ϸ���� ",
    "     ��   "
	"    (��)  "
	"     ��-  "
	"  С��Ƶ  ",  
    "._________________________________________________."
    "|  Img | 404name                                  |"
    "|  404 | 2021��11��17�� 00:24                     |"
    "|-------------------------------------------------|"
    "| QQ for Cmd 1.0.0�汾�����ˣ�����·���������~   |"
    "| $ https://qq.404name.top                        |"
    "| ��� 10w��                     ����|����|ת��   |"
    "| 1.3���˾��ú���                                 |"
    "| 1981��ת��                                      |"
    "| 201�����ۻظ�                                   |"
    "|_________________________________________________|",
    "._________________________________________________."
    "|  Img | 404name                                  |"
    "|  404 | 2021��11��10�� 00:24                     |"
    "|-------------------------------------------------|"
    "| �ռ书�ܺ������ߣ������ڴ�                      |"
    "| ��� 1k��                      ����|����|ת��   |"
    "| 100�˾��ú���                                   |"
    "| 19��ת��                                        |"
    "| 20�����ۻظ�                                    |"
    "|_________________________________________________|"
	"                                                   "
	"                                                   "
	"                                                   "
	"                                                   "
	"                                                   "
	"                  QQ for Cmd                       "
	"                    V1.0.0                         ",
    "    ��            "
	"   (��)  QQ       "
	"  /(��)\\ for cmd  "
	"   -��-  V 1.0.0  ", //31-35 ��¼ע��ҳ�� 
    "._______________________________________________."
    "| �˺� |                                      > |"
    "|______|________________________________________|",
    "._______________________________________________."
    "| ���� |                                      > |"
    "|______|________________________________________|",
    "(��¼/ע�� ->)",  
    "���ֻ��ŵ�¼��",  
    " ���˺����롽 ", //36-40 
    "�����û�ע�᡽",
    ".______.___________________________________.______."//38-45
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    ".______.___________________________________.______."
    "|  Img | 2021-11-21                12:38:18|  Img |"
    "|  404 | hello_world                       |  404 |"
    "|______|___________________________________|______|",
    "._________________________________________________."
    "|  []  |                                        + |"  //46-47 
    "|______|__________________________________________|",
    "| <                                            �� |"
    "|_________________________________________________|",
};
string GetSystemTime()
{
	SYSTEMTIME m_time;
	GetLocalTime(&m_time);
	char szDateTime[100] = { 0 };
	sprintf(szDateTime, "%02d-%02d-%02d_%02d:%02d:%02d", m_time.wYear, m_time.wMonth,
		m_time.wDay, m_time.wHour, m_time.wMinute, m_time.wSecond);
	string time(szDateTime);
	return time;
}
string replaceBlank(string str){
	for(int i = 0; i < str.size() ; i++){
		if(str[i] == ' ') str[i] = '_';
	}
	return str;
} 
//�ṹ�����

class Chat
{
    public:
	    string fromUser;  
	    string toUser;  
	    string message;  
		string time;  
    public:
    	
    	static vector<Chat> chatList;
    	// ��ʼ�� 
    	static bool updateAndSave(){
    		ofstream osm("chat.txt");
    		for(Chat chat:chatList){
    			osm <<  chat.toString() << endl;
			}
			osm.close();
		}
		static void printAll(){
			for(Chat chat:chatList){
    			cout << chat.toString() << endl;
			}
		}
		static bool init(){
			//��ʼ���û� 
    		for(int i = 0; i < 5; i++){
    			chatList.push_back(Chat("admin","QQ�ٷ�","hello world"));
			}
		}
		static void addChat(string from,string to,string msg){
			chatList.push_back(Chat(from,to,msg));
			updateAndSave();
		}
    	static bool load(){
    		ifstream ism("chat.txt");
			if(!ism) 
				return false;
    		string s;
			while(getline(ism,s))//���ж�ȡ���ݲ�����s�У�ֱ������ȫ����ȡ
				chatList.push_back(Chat(s)); 
			ism.close();
		}
		Chat(){}
    	Chat(string from,string to,string msg){
			fromUser = replaceBlank(from);
			toUser = replaceBlank(to);
			if(msg.size() >= 30){
				msg = msg.substr(0,30) + "...";
			}
			message = replaceBlank(msg);
			time = GetSystemTime();
		}
		Chat(string str) {
			stringstream input(str);
			input >> this->fromUser;
			input >> this->toUser;
			input >> this->message;
			input >> this->time;
		}
        string toString() { 
			return fromUser+" "+toUser+" " + message + " " + time;
		} 
        
};
vector<Chat> Chat::chatList = vector<Chat>();
map<string,vector<Chat>> chatHistory;
string chatHis[6][4]; 
class User
{
    public:
	    string username;
	    string pwdmd5;
    public:
    	static vector<User> userList;
    	// ��ʼ����ɫ 
    	static bool updateAndSave(){
    		ofstream osm("user.txt");
    		for(User user:userList){
    			osm <<  user.toString() << endl;
			}
			osm.close();
		}
		static void printAll(){
			for(User user:userList){
    			cout << user.toString() << endl;
			}
		}
		static bool init(){
			//��ʼ���û� 
    		for(int i = 0; i < 10; i++){
    			userList.push_back(User(to_string(i),"admin"));
			}
		}
		static bool loginOrRegister(string u,string p){
			//������˺žͼ�� 
			for(User user:userList){
				if(user.getName() == u){
					return user.getPwd() == p;
				}
			}
			// û�о�ע��
			userList.push_back(User(u,p));
			//�־û������� 
			Chat::addChat("QQ�ٷ�",u,"��ӭʹ��QQ for CMD"); 
			updateAndSave();
			return true;
		}
    	static bool load(){
    		ifstream ism("user.txt");
    		string s;
			while(getline(ism,s))//���ж�ȡ���ݲ�����s�У�ֱ������ȫ����ȡ
				userList.push_back(User(s)); 
			ism.close();
		}
        User(string u,string p):username(u),pwdmd5(p) {
			//���ڶ�p���� 
			this->username = replaceBlank(u);
			this->pwdmd5 = replaceBlank(p);
		}
		User(string str) {
			stringstream input(str);
			input >> this->username;
			input >> this->pwdmd5;
		}
        User() { }
        string toString() { 
			return username+" "+pwdmd5;
		} 
		string getName(){
			return this->username;
		}
		string getPwd(){
			return this->pwdmd5;
		}
};
vector<User> User::userList = vector<User>(); 
User toUser;
User nowUser;
//һЩϵͳ���ܺ���
void HideCursor()  //�������
{
    CONSOLE_CURSOR_INFO cursor_info = {1, 0};  //��ߵ�0�����겻�ɼ�
    SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursor_info);
}
void gotoxy(int x, int y)  //�ù���ƶ�������ȥ��ӡ���ݺ���
{
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD pos;
    pos.X = x;
    pos.Y = y;
    SetConsoleCursorPosition(handle, pos);
}

void setColor(short x)  //�Զ��庯���ݲ����ı���ɫ
{
    // if (x >= 0 && x <= 15) //������0-15�ķ�Χ��ɫ
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE),
                            x);  //ֻ��һ���������ı�������ɫ
    /* else                              //Ĭ�ϵ���ɫ��ɫ
         SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7);*/
}
//һЩϵͳ���ܺ���

void debug() {
    static int x = 0, y = 0;
    static char flag = '#';
    //�������־ͻ��л�����Ӧ�Ķ�����ӡ���㻭��
    char print[11] = " !@#$%^&*(";
    int nextgo[4][2] = {-1, 0, 0, 1, 1, 0, 0, -1};  //˳ʱ��wdsa��һ�����ƶ�
    
    int bookMap[128] = {0};
    bookMap['w'] = 0, bookMap['d'] = 1;
    bookMap['s'] = 2, bookMap['a'] = 3;
    gotoxy(y, x), printf("%c", flag);
    char turn = _getch();
    if (turn == ' ')
        system("cls");
    else if (turn >= '0' && turn <= '9')
        flag = print[turn - '0'];
    else {
        x += nextgo[bookMap[turn]][0];
        y += nextgo[bookMap[turn]][1];
        gotoxy(y, x), printf("_");
        gotoxy(0, 20), printf("x=%d y=%d nowturn='%c'", x, y, flag);
    }
}
void printCuttingLine(int deep, char wall = '-') {  ///��ӡ�ָ���
    for (int i = 1; i < WIDTH - 1; i++) {
        gotoxy(i, deep);
        printf("%c", wall);
    }
}
//��ӡ�����������
void printBoder() {  //��ӡ�߿�
    setColor(Boder_Color);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            gotoxy(j, i);
            if (i == 0 || i == HEIGHT - 1)
                printf("#");
            else if (j == 0 || j == WIDTH - 1)
                printf("#");
        }
    }
    setColor(Normal_Color);
        //��ӡpage���÷ָ���
    int mapHeight = appMap[nowPage][0][0];  //��ȡ��ͼ�߶�
    int cuttingLine =
        appMap[nowPage][mapHeight][0];  //�����һ��0λ��ȡ�ָ��߸���
    for (int i = 0; i < cuttingLine; i++) {
        printCuttingLine(appMap[nowPage][mapHeight][i + 1]);
    }
}

//�����ַ���ӡģ��
void printDemo(int index, bool on = false) {
    int x = appMapMessage[index][0];
    int y = appMapMessage[index][1];
    int h = appMapMessage[index][2];
    int w = appMapMessage[index][3];
    int turn = appMapMessage[index][4];  //��ȡ��дͼ��״̬
    char prePrint[h + 1][WIDTH];
    int cnt = 0;
    if (turn) {
        for (int i = 0; i < h; i++) {
            strncpy(prePrint[i], appMapIcon[index] + w * i, w);
            prePrint[i][w] = '\0';
            gotoxy(y, x + i);
            printf("%s", prePrint[i]);
        }
    }
   
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            gotoxy(y + j, x + i);
            if (!on) {  //δ��ѡ��״̬
                if (!turn) {
                    printf("#");
                }  //ʲô��û�е�����ҳ
                else if (turn == 1 &&
                         (activeIndexTop == index ||
                          activeIndexBottom == index) &&
                         j == 0) {
                    gotoxy(y + j, x + i);
                    setColor(Is_Active_Color);
                    printf("%s", prePrint[i]);
                    setColor(Normal_Color);
                }
                // printf("%c",str[index++]);
            } else {
                if (i == 0 || i == h - 1 || j == 0 || j == w - 1) {
                    //�����ǰ����ҳ�Ƽ���ǩ����ô�Ƽ���ǩӦ���ǳ�����active״̬
                    //��������ѡ��״̬
                    if (turn == 1 && j == 0) {
                        gotoxy(y + j, x + i);
                        if (index >= 9 && index <= 14 ||
                            index >= 21 && index <= 24) {
                            setColor(Video_Color);
                        }  //��Ƶ����ʽ
                        /*else if (index >= 15 && index <= 19 ||
                                 index >= 25 && index <= 28 ||
                                 index >= 37 && index <= 41 || index == 35) {
                            setColor(Is_Selected3_Color);
                        }  //һЩ��λ�ַ�����ʽ*/ 
                        else
                            setColor(Is_Selected1_Color);  //������ʽ
                        printf("%s", prePrint[i]);
                        // printf("%c",str[index++]);
                    }else if (turn == 0) {
	                    setColor(Is_Selected2_Color);
	                    printf("O");
                	}
                } 
	            else {
	                if (!turn)
	                    printf("#");
	            }
	            
	            setColor(Normal_Color);
            }
            if(i == h - 1 && j == w - 1){
            	if (on) { 
	            	if (index >= 9 && index <= 14 ||
                        index >= 21 && index <= 24) {
                        setColor(Video_Color);
                    }else{
						setColor(Is_Selected1_Color);                    	
					}
				}
            	if(index == 1){
					gotoxy(11,1);
					cout << nowUser.getName() << "    ";
				}else if(index == 47){
					gotoxy(11,1);
					cout << toUser.getName() << "    ";
				} 
				else if(index == 32){
					gotoxy(22, 15);
					if(username != "")
				    cout << "��" << username << "��"; 
				}else if(index == 33){
					gotoxy(22, 18);
					for(int i = 0; i < pwd.size(); i++){
						cout << '*';
					}
				}else if(index >= 9 && index <= 14){
        			// �жϳ��� 
		        	if(index-9 < indexCnt){
		        		//��ӡ����
			        	gotoxy(y + 9, x + 1);
			        	cout << chatHis[index-9][0] << "           "; 
						//��ӡ����
						gotoxy(y + 9, x + 2);
						cout << "                                  ";
						gotoxy(y + 9, x + 2);
			        	cout << chatHis[index-9][1]; 
						//��ӡʱ��
						gotoxy(y + 40, x + 1);
			        	cout << chatHis[index-9][2].substr(0,10); 
						//��ӡ��Ϣ�� 
			        	gotoxy(y + 44, x + 2);
			        	cout << "(" << chatHis[index-9][3] << ")"; 
					}
				}else if(index >= 38 && index <= 45){
					int itemIndex = index - 38 + startItem;
					if(itemIndex > endItem){
						//�ƿ�
						for(int i = 0; i < 4; i++){
							gotoxy(y, x + i);                                       
							printf("                                                   ");
						}
					}else{
						Chat chat = chatHistory[toUser.username][itemIndex];
						//��ӡʱ�� 
						gotoxy(y + 9,x+1);
						cout << chat.time.substr(0,10);
						gotoxy(y + 35,x+1);
						cout << chat.time.substr(11,8);
						//��ӡ�Ի�
						gotoxy(y + 9,x+2);
						cout << "                      ";
						gotoxy(y + 9,x+2);
						cout << chat.message;
						//ɾ���߿� 
						int overFlow = 0;
						if(chat.toUser == nowUser.username){
							overFlow = 44;
						}
						gotoxy(y + overFlow,x+0);
						cout << "       ";
						gotoxy(y + overFlow,x+1);
						cout << "       ";
						gotoxy(y + overFlow,x+2);
						cout << "       ";
						gotoxy(y + overFlow,x+3);
						cout << "       ";
					}
					
				}
				setColor(Normal_Color); 
			}
        }
    }
    
    if (!on && !turn) {
        gotoxy(20, 20);
        printf("�� �� �� Ǹ ");
        gotoxy(14, 22);
        printf("�� Ҫ �� �� ҳ �� �� �� ��");
        gotoxy(17, 24);
        printf("�� �� �� �� �� �� ��");
        gotoxy(17, 26);
        printf("�� q �� �� �� һ ҳ");
    }
}
void clsPage() {
    //�˴�д������߿����
    system("cls");
}
void loadChat(){ //Ҫ�ȵ�¼ 
	//LRU�㷨��ȡ����6����¼ 
	stack<string> sta;
	chatHistory.clear();
	for(Chat chat:Chat::chatList){
    	if(chat.fromUser == nowUser.getName()){ //��ȡ���͵� 
    		if(!chatHistory.count(chat.toUser)){
    			sta.push(chat.toUser);
    			vector<Chat> temp;
    			temp.push_back(chat);
    			chatHistory[chat.toUser] = temp;
			}else{
				//ɾ��ջ��Ԫ�� 
				stack<string> temp;
				while(sta.top() != chat.toUser){
					temp.push(sta.top());
					sta.pop();
				}
				sta.pop();
				while(temp.size()){
					sta.push(temp.top());
					temp.pop();
				}
				sta.push(chat.toUser);
				chatHistory[chat.toUser].push_back(chat);
			}
		}else if(chat.toUser == nowUser.getName()){ //��ȡ���յ� 
    		if(!chatHistory.count(chat.fromUser)){
    			sta.push(chat.fromUser);
    			vector<Chat> temp;
    			temp.push_back(chat);
    			chatHistory[chat.fromUser] = temp;
			}else{
				//ɾ��ջ��Ԫ�� 
				stack<string> temp;
				while(sta.top() != chat.fromUser){
					temp.push(sta.top());
					sta.pop();
				}
				sta.pop();
				while(temp.size()){
					sta.push(temp.top());
					temp.pop();
				}
				sta.push(chat.fromUser); 
				chatHistory[chat.fromUser].push_back(chat);
			}
		}
	}
	int now = 0;
	while(now < 6 && sta.size()){
		string uname = sta.top();
		sta.pop();
		chatHis[now][0] = uname;
		chatHis[now][1] = chatHistory[uname][chatHistory[uname].size()-1].message;
		chatHis[now][2] = chatHistory[uname][chatHistory[uname].size()-1].time;
		chatHis[now][3] = to_string(chatHistory[uname].size());
		now++;
	}
	indexCnt = now; 
	while(now < 6){
		chatHis[now][0] = "";
		chatHis[now][1] = "";
		chatHis[now][2] = "";
		chatHis[now][3] = "";
		now++;
	}
}
void loadPage() {         //������ҳ�˵�
    int book[100] = {0};  //αԤ���أ����ع��Ľ���Ͳ��ټ���
    printBoder();
    int h = appMap[nowPage][0][0] - 2;  //��ȥ����ǽ��
    int w = 5;
    for (int i = 1; i <= h; i++) {
        for (int j = 1; j <= w; j++) {
            int index = appMap[nowPage][i][j];  //��ȡ��Ӧģ��Ķ�Ӧֵ
            if (!book[index])
                printDemo(index, false);
            book[index] = 1;
        }
    }
}
void error(string str,int x){
	gotoxy(18,x-1);
	printf("+------------------+"); 
	gotoxy(18,x); 
	cout <<"+------------------+"; 
	gotoxy(19,x); 
	cout << str; 
	gotoxy(18,x+1); 
	printf("+------------------+"); 
}
void checkAction(int action) {  //����·����ת
	int prePage = nowPage;
	if (action == 1){ //�˳����ص�½���� 
		isLogin = false;
		while(pageStack.size()) pageStack.pop();
		nowPage = 4;
		pageStack.push(nowPage);
	}
	else if (action == 2 || action == 3) {  //������
        gotoxy(22, 4);
        setColor(Is_Selected1_Color);
        printf("                         ");
        gotoxy(22, 5);
        printf("_________________");
        char searchMsg[100];
        gotoxy(22, 4);
        scanf("%s", searchMsg);
        setColor(Normal_Color);
		//����������Ϣ 
		int flag = 0;
		for(User user:User::userList){
			if(user.getName() == searchMsg){
				toUser = user;
				flag = 1;
				break;
			}
		}
		if(flag == 0){
			error("δ�ҵ��û�",4); 
		}else{
			int size = chatHistory[toUser.username].size();
			endItem = size - 1;
			if(endItem < 0) endItem = -1;
			startItem = endItem - 7;
			if(startItem < 0) startItem = 0;
			nowPage = 5;
		}
		//���û� ��ת������� 
        
    }else if(action == 32){ //�˺� 
    	gotoxy(22, 15);
        setColor(Is_Selected1_Color);
        printf("|                         ");
        char searchMsg[100];
        username = "";
        gotoxy(22, 15);
        cin >> username;
        gotoxy(22, 15);
        cout << "��" << username << "��"; 
	}else if(action == 33){ //���� 
    	gotoxy(22, 18);
        setColor(Is_Selected1_Color);
        printf("                          ");
        int len = 0;
        char searchMsg[100];
        gotoxy(22, 18);
		printf("|");
        pwd = "";
        while(1){
        	char ch = getch();
        	if(ch == 13){
        		break;
			}else if(ch == 8){
				if(len >= 1){
					gotoxy(22+len, 18);
					printf("|%c",' ');
					len--;
					pwd.pop_back();
				}
			}else{
				pwd += ch;
				if(len > 0){
					gotoxy(22+len, 18);
					printf("%c",'*');
				}
				len++;
				gotoxy(22+len, 18);
				printf("%c|",ch);
			}
		
		}
        
	}
	else if(action == 46){ //������Ϣ 
    	gotoxy(10, 43);
        setColor(Is_Selected1_Color);
        printf("|                        ");
        char msg[100];
        gotoxy(10, 43);
        cin >> msg;
        Chat::addChat(nowUser.getName(),toUser.getName(),msg);
        loadChat();
		int size = chatHistory[toUser.username].size();
		endItem = size - 1;
		startItem = endItem - 7;
		if(startItem < 0) startItem = 0;
        loadPage(); 
	}
    else if (action == 3) {          //��Ϣ��
        nowPage = 0;
    } else if (action == 5) {  //��ҳ
        nowPage = 1;
    } else if (action == 6) {  //����
        nowPage = 2;
    } else if ((action >= 9 && action <= 14)) {
    	int index = action - 9; 
    	//�ҵ�Ŀ�������û� 
    	if(chatHis[index][0] == ""){
    		error("�Ҳ������û�",4);
		} else{
			toUser.username = chatHis[index][0];
			int size = chatHistory[toUser.username].size();
			endItem = size - 1;
			startItem = endItem - 7;
			if(startItem < 0) startItem = 0;
        	nowPage = 5; 
		}
    	                           //����������ϸҳ 
    } else if (action >= 17 && action <= 19) {  //�ײ��ǩ��
        if (action == 17) {
            activeIndexBottom = 17;  //��Ӧ�����ҳ
            activeIndexTop = 5;
            nowPage = 1;  //ûд�����ȷ���ҳ �����nowPageҪ��Ӧ����
        } else if (action == 18) {
            activeIndexBottom = 18;  //��Ӧ�����ϵ�� 
            activeIndexTop = 5;
            nowPage = 2;  //ûд�����ȷ���ҳ �����nowPageҪ��Ӧ����
        } else if (action == 19) {
            activeIndexBottom = 19;  //��Ӧ��ɶ�̬ 
            activeIndexTop = 5;
            nowPage = 3;  //ûд�����ȷ���ҳ �����nowPageҪ��Ӧ����
        }else
            nowPage = 0;
    }  else if (action == 29) {  //�����Ƶģ��  ���Ӧֹͣ��Ƶ
        if (!stopPlay)
            stopPlay = true;
        else
            stopPlay = false;
    } else if(action == 34){ //��¼ģ�� 
    	if(username == ""){
    		error("�˺Ų���Ϊ��",26); 
		}else if(pwd == ""){
    		error("���벻��Ϊ��",26); 
		}else if(User::loginOrRegister(username,pwd)){
    		isLogin = true; 
    		pageStack.pop(); 
	    	nowPage = 1;
	    	pageStack.push(nowPage);
	    	for(User user:User::userList){
	    		if(user.getName() == username){
	    			nowUser = user;
				}
			}
			loadChat();
		}else{
			//��¼ʧ������ƥ�����
			error("�˺������������",26); 
		} 
    	
	}else if(action == 47){
		pageStack.pop();
    	nowPage = pageStack.top(); 
	}else {
        nowPage = 0;  // ûд�Ľ���404
    }
    activeIndexTop = action;  //�ж������л���
    if(nowPage != prePage){
    	pageStack.push(nowPage);
	}
}

void move() {
    //�����ʼ����
    static int x = 1, y = 1;
    static int oldPage = nowPage;
    int oldX = x, oldY = y;
    int nextgo[5][2] = {0, 0, -1, 0, 0,
                        1, 1, 0,  0, -1};  //˳ʱ��wdsa��һ�����ƶ�
    int bookMap[128] = {0};
    bookMap['w'] = 1, bookMap['d'] = 2;
    bookMap['s'] = 3, bookMap['a'] = 4;
    //ȡ����ǰλ��
    //һ��index�±�λ�þʹ洢��һ��action
    int index = appMap[nowPage][x][y];
    int oldIndex = index;
    printDemo(index, false);  //�����ϴε�λ��
    char turn = _getch();
    x += nextgo[bookMap[turn]][0];
    y += nextgo[bookMap[turn]][1];
    index = appMap[nowPage][x][y];
    if (index == 0) {  //����Ǳ߽������
        x = oldX, y = oldY;
    } else {  //������ƶ�����һ���µ�λ��
        while (oldIndex == index && index != 0 && bookMap[turn] != 0) {
            x += nextgo[bookMap[turn]][0];
            y += nextgo[bookMap[turn]][1];
            index = appMap[nowPage][x][y];
        }  // while��������ռ��������״̬
        if (index == 0)
            x = oldX, y = oldY;
    }
    index = appMap[nowPage][x][y];
    printDemo(index, true);  //��ӡ�µ�״̬
    if (turn == ' ' || turn == 'q') {
        if (turn == ' ') {
            checkAction(index);
        } else {  //���� 
            pageStack.pop();
			if(pageStack.size() == 0){ //������ 
				if(isLogin){ //�����Ƿ��¼ 
					pageStack.push(1); //��ҳ 
				}else{
					pageStack.push(4); //��¼ҳ 
				}
			}
            nowPage = pageStack.top();  //�ص���ҳ
        }

        if (nowPage != oldPage) {  //����仯
            oldPage = nowPage;
            x = 1;
            y = 1;
            clsPage();   //���˱߿�ȫ������
            loadPage();  //�����½���
        }
    }
    
    if(nowPage == 5 && (turn == 'a' || turn == 'd')){
    	//0-11 
		//0-7 ��һҳ�� 4-11  ��һҳ��0-7 
    	int size = chatHistory[toUser.username].size(); 
    	//���们����ҳ�㷨  
    	if(turn == 'a'){
    		startItem -= 8;
    		endItem -= 8;
    		if(startItem < 0){
    			startItem = 0;
    			endItem = startItem + 7;
			}
			if(endItem >= size){
				endItem = size - 1;
			} 
		}else{
			startItem += 8;
    		endItem += 8;
    		if(endItem >= size){
    			endItem = size - 1;
    			startItem = endItem - 7;
			}
			if(startItem < 0){
				startItem = 0;
			}
		}
		x = 1;
        y = 1;
        clsPage();   //���˱߿�ȫ������
        loadPage();  //�����½���
	}
}

void init() {
	User::load();
	Chat::load();
	getchar();
	getchar();
	/*User::load();
	Chat::load();*/
    char cmd[50];
    pageStack.push(nowPage);
    sprintf(cmd, "mode con cols=%d lines=%d", WIDTH, HEIGHT);
    printf("%s", cmd);
    system(cmd);         //���ý����С
    system("color 07");  //���ý�����ɫ��

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            gotoxy(j, i);
            printf("%c", bacground[i][j]);
        }
    }
    printBoder();
    getchar();
    system("cls");
    loadPage();
    HideCursor();
}

int main() {
    // ctrl+���ֿ��Ե������
    init();
    while (1) {
        if (kbhit()) {
            // debug(); //����debug�ͻ���ʾ������Ϣ������ע��move()
            move();
        }
    }
}
// 7  - 41
