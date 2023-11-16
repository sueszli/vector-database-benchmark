#include <windows.h>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

bool skipFrame = true; //�����س����� 

/**
 *  ��ǿ����ڲ�Դ��й¶
 *
 *	����--����N
 */
// ��������ing
class Sence {
   public:
    //�����߶�
    const static int WIDTH = 201;
    const static int HEIGHT = 76;
    //��ǰ����
    static string nowSceneFrame[HEIGHT];
    //��Ա������־
    static string roleName;
    //��Ա������־
    static string emotion;
    //��Ա�ж���־
    static string action;
    // ��Ա̨����־
    static string dialogue;
    // �ӳ�ʱ��
	static int delayTime; 
    //���س���
    static void loadFrame(string sName) {
        std::ifstream readFile("./sence/" + sName + ".txt");
        if (readFile)  //��ȡ�ɹ�
        {
            string str;
            int fileHeight = 0;
            int firstLine = 0; 
            while (getline(readFile, str)) {
            	firstLine++;
            	if(firstLine == 1){ // ��һ�л�ȡ����ʱ�� 
            		delayTime = atoi(str.c_str());
            		continue;
				}
                nowSceneFrame[fileHeight++] = str;
            }
        } else  //��ȡʧ��
        {
            cout << "Open file faild." << endl;
            for (int i = 0; i < HEIGHT; i++) {
                nowSceneFrame[i] = sName;
            }
        }
    }
    //���ų���
    static void play(string sName) {
        loadFrame(sName);
        system("cls");
        for (int i = 0; i < HEIGHT; i++) {
            cout << nowSceneFrame[i] << endl;
        }
        cout << "��sence��" + sName << endl;
        log();
        if(skipFrame){
        	getchar();
		} else{
        	Sleep(delayTime);			
		}

    }
    // log��ӡ��ǰ������־
    static void log() {
        cout << "��log��" << roleName << emotion << action << endl;
        //��ӡ�����
        roleName = "";
        emotion = "";
        action = "";
    }
};
string Sence::nowSceneFrame[Sence::HEIGHT] = {"��ʼ��"};
string Sence::action = "��ʼ��";
string Sence::emotion = "��ʼ��";
string Sence::roleName = "��ʼ��";
int Sence::delayTime = 1000;
// ������Աing
class Actor {
   public:
    //��Ա����
    string actorName;
    //��ɫ����
    string roleName;
    // �����巽��-��ʽ��̡�
    //����ʲô [Ԥ�����봦��action��д��Sence��־]
    Actor& to(string action) { Sence::action = action; }
    //�����ı� [Ԥ�����봦��emotion��д��Sence��־]
    Actor& be(string emotion) {
        // Sence::roleName = this->roleName;
        Sence::emotion = emotion;
    }
    //����
    Actor& say(string dialogue) { Sence::play(dialogue); }
    // ������
    Actor& hear(string dialogue) { Sence::play(dialogue); }
    // ��ʼ����
    Actor& start() { Sence::roleName = this->roleName; }
    Actor(string roleName) { this->roleName = roleName; }
};

// ����ģʽ ����
// ָ������ C����
// sence��ע Ϊ�籾
// �������� Ϊ��Ա�������
int main() {
	getchar();
    //��Ա��
    Actor HuaQiang("����ǿ");
    Actor XianHui("�ͻݵ�ˮ��̯�ϰ�");
    Actor SaRiLa("�ϰ�С��-������");
    Actor ZhangYuBrother("С�ܶ���-�����");
    Actor EiWhatsUp("�G��ǿһ��");

    // sence01 �԰׵�����ǿ��
    Sence::play("��һ����ǰ�����");
	getchar();
	getchar();
    // sence02 ��ǿ�ﳵ������
    HuaQiang.start()
        .be("���Ÿ���")
        .to("���ų���");

    // sence03 ͣ���ŵ�������
    HuaQiang.start()
        .to("ͣ�ó�")
        .hear("��������");

    // sence04 ��ǿѯ�ʸ��Ǽ�
    HuaQiang.start()
        .be("����")
        .to("�����ϰ�")
        .say("�϶���Ǯһ��");

    // sence05 �ϰ��Ի����Ǯ
    XianHui.start()
        .be("��Ȼ��")
        .to("�ش��")
        .say("����Ǯһ��");

    // sence06 ��ǿֱ��WhatsUp
    HuaQiang.start()
        .be("��ƤЦ��")
        .to("��Ӧ��")
        .say("�Ұ���")
        .say("��Ƥ���ǽ������Ļ���");

    // sence07 �ϰ����̴����
    XianHui.start()
        .be("��м��")
        .to("���˿���")
        .say("���������������й�ѽ")
        .say("�ⶼ�Ǵ���Ĺ�");

    // sence08 �ϰ����ǿ�ͻ�
    //         С�ܴ�����ԭ��
    XianHui.start()
        .to("��ָָ���")
        .say("���ͻ��һ��ͻ���");

    ZhangYuBrother.start()
        .be("������")
        .to("����ԭ��");

    // sence09 ��ǿЦԻ��һ��
    HuaQiang.start()
        .be("������")
        .to("̧ͷ���")
        .say("������һ��");

    // sence10 �ͻ��ϰ�ȥ����
    XianHui.start()
        .be("С��һЦ")
        .say("��")
        .to("ת�������ϲ���������")
        .say("�����ô��");

    // sence11 ��ǿѯ�ʹϱ���
    HuaQiang.start()
        .to("˫��һ��")
        .say("��ϱ�����");

    // sence12 �ϰ���Ц������
    XianHui.start()
        .be("��Ц��")
        .to("�ش��")
        .say("�ҿ�ˮ��̯�������������ϵ���");

    // sence13 ÷�����ȹϱ���
    HuaQiang.start()
        .be("��ǿ������")
        .to("�ظ���")
        .say("��������ϱ�����");

    // sence14 �ϰ循��������
    Sence::play("����");
    XianHui.start()
        .be("�����")
        .to("�㶵Ŀ���" + HuaQiang.roleName);

    // sence15 �ϰ弱��Ҫ��Ҫ
    XianHui.start()
        .be("�����")
        .to("�ʵ�")
        .say("���ǹ����Ҳ��ǲ���")
        .say("��Ҫ��Ҫ��");

    // sence16 ��ƤЦ�����Ҫ
    HuaQiang.start()
        .be("��ƤЦ����")
        .to("�����ϰ�")
        .say("�����Ҫ�����ҿ϶�Ҫ��")
        .be("��˵���ߵ�")
        .to("��������������������")
        .say("����Ҫ�ǲ�����ô��ѽ");

    // sence17 Ҫ�ǲ����ҳ���
    XianHui.start()
        .to("��ָ������")
        .say("Ҫ�ǲ������Լ�������");

    // sence18 ��ʮ�����ʮ��
    XianHui.start()
        .to("��Ū�ų���")
        .say("ʮ�����ʮ��");

    // sence19 ��ǿ��Ц������
    HuaQiang.start()
        .be("קק������")
        .to("��Ц��")
        .say("�����Ķ���ʮ�����")
        .say("�����������ѽ");

    // sence20 ���˼���������
    XianHui.start()
        .say("�����ǹ����Ҳ��ǲ���")
        .be("�����ܻ���")
        .to("�ѹϷ���")
        .say("��Ҫ��Ҫ");

    // sence21 ��ǿ����������
    HuaQiang.start()
        .to("�ѳӷ���")
        .say("����ʯ")
        .say("������˵��")
        .say("���Ҫ���������Լ��̽�ȥ")
        .to("�õ�������");

    // sence22 �ϰ��ŭ����ɱ
    XianHui.start()
        .be("��ŭ��")
        .to("����" + HuaQiang.roleName)
        .say("��TM���ҹ��ǰ�");

    // sence23 �ᵶָ��������
    SaRiLa.start()
        .be("���ޱ����")
        .to("����ɱ����������" + HuaQiang.roleName)
        .say("������");

    HuaQiang.start()
        .to("С��һ˦");
    // sence24 ԭ����������
    ZhangYuBrother.start()
        .to("�������˵��ϰ�")
        .say("�����");

    HuaQiang.start()
        .to("˧�����ﳵ��ȥ")
		.say("�ݰ�"); 
		
    // sence25 ����һЦ����ǿ
    EiWhatsUp.start()
        .be("���ĵ�")
        .to("�����к�")
        .say("����ǿ");

    HuaQiang.start()
        .be("���ĵ�")
        .to("����һЦ�ﳤȥ");
        
    Sence::play("��ǿ�˳�");

    // sence26 �������

    // sence27 DLC�籾
    // ������ing
}
