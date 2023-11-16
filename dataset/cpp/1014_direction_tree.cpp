#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <algorithm>
#include <iostream>
/*
���ʲ�������Trie��
�ֵ������� 
СHo�����ܲ��ܶ���ÿһ���Ҹ������ַ�������������ʵ������ҵ�������ַ�����ͷ�����е����أ���
����ս��СHo���������ô�᲻���أ���ÿ����һ���ַ������Ҿ����α����ʵ�������е��ʣ�
�������ҵ��ַ����ǲ���������ʵ�ǰ׺�������ˣ���
СHiЦ�������㰡������̫�����ˣ�~�����Ȿ�ʵ�����10������ʣ���ѯ����һ��Σ����Ҫ�㵽��������ȥ����
Hihocoder #1014 : Trie��
ʱ������:10000ms
����ʱ��:1000ms
�ڴ�����:256MB

root_node
| 
a(0)    c(2)
|       |
P()     a(0)��l()��l() 
|       |
p()     t()
|
l()
|
e()

apple   cat   call 
*/ 

//AC G++ 409ms 76MB

using namespace std ;
// �� �ڵ� 
typedef struct TrieNode{
    int count ;
    struct TrieNode *next[26] ;//ÿһ���ڵ㶼����ָ��26����ĸ�е�һ������ �ڵ� 
    bool exit_word ;
}TrieNode ;

// ���� ����  ���ظ��ڵ� 
TrieNode* createKeyNode(){
    TrieNode *insert_node = (TrieNode *)malloc(sizeof(TrieNode)) ;//�����ڴ� 
    insert_node->count = 0 ;//�����ýڵ�ĵ������� 
    insert_node->exit_word = false ;// �����ַ��ڵ� ��ʶ 
    for(int i = 0 ; i < 26 ; i++)insert_node->next[i] = NULL ;//ָ��ڵ�ָ��Ϊ �� next[0] Ϊa�ַ��ڵ�  next[25]Ϊz �ַ��ڵ� 
    return insert_node ;
}

// ����������   Ӣ���ַ��� Ӣ�ĵ��� 
void keyTreeInsert(TrieNode *root_node , const char* insert_str){
    char current_char ;//���������� ���뵥�ʲ������� 
    int i = 0 ;
    int j = 0 ;
    TrieNode *current_node = root_node ;//ÿ�β����µ��� �����ͷ��㿪ʼ���� 
    while(insert_str[i]!='\0'){//�����е��ַ�ȫ������ 
        current_char = insert_str[i++] ;//��ǰ���ַ� 
        j = current_char - 'a' ;//0~25  ��Ӧa~z26����ĸ 
        if(current_node->next[j]==NULL)current_node->next[j] = createKeyNode() ;//�µ����ж�Ӧ���ַ�δ������ ����һ���½ڵ� 
        current_node = current_node->next[j] ;//node���� 
        current_node->count++ ;//�����ýڵ� ���ʵ����� ���� 
    }
    current_node->exit_word = true ;//�ýڵ�Ϊ���ʵĽ����ַ� 
}

// ������ Ԫ��  ������ָ��Ԫ�ؿ�ͷ���ַ��� 
int keyTreeSearch(TrieNode *root_node,const char *search_str){
   if(!root_node)return 0 ;//���� 
   int i = 0 ;
   int j = 0 ;
   char current_char ;//��ǰ �ַ� 
   TrieNode *current_node = root_node ;//��ͷ�ڵ㿪ʼ���� 
   while(search_str[i]!='\0'){
       current_char = search_str[i++] ;//��ǰ���ַ� 
       j = current_char-'a' ;//0~25  ��Ӧa~z26����ĸ 
       if(!current_node->next[j])return 0 ;//δ�ҵ� 
       current_node = current_node->next[j] ;//���α��������һ�� �����ַ��ڵ� 
   }
   return current_node->count ;//���ؾ��� ָ���ַ������һ���ַ� �ĵ������� 
}

int main(){
    int m,n ;
    int i = 0 ;
    char source_str[11] ;//���ʳ�������Ϊ 11 
    char target_str[11] ;
    TrieNode *root_node = createKeyNode() ;//���ڵ� 
   // printf("�����뵥������:\r\n"); 
    scanf("%d" , &n) ;//�ܵ������� 
    //printf("����������ÿ������:\r\n");
    while(i++<n){
        scanf("%s" , source_str) ;
        keyTreeInsert(root_node , source_str);//ÿ�β��붼���ͷ������α��� �����ڵ� �ڵ����+1  δ���ڵĽڵ���д��� 
    }
    i= 0 ;
    // printf("������Ҫ��ѯ��������:\r\n"); 
    scanf("%d" , &m) ;
    // printf("������ÿһ��Ҫ��ѯ��:\r\n"); 
    while(i++<m){
        scanf("%s" , target_str) ;
        printf("%d\n",keyTreeSearch(root_node,target_str)) ;
    }
    return 0 ;
}
