#include<iostream>
using namespace std;
struct Node
{
    int data;
    struct Node* flink;
    struct Node* blink;
}*head;

void deleteNode(struct Node* pPre)
{
    struct Node* pLoc;;
    struct Node* pNext;
    if (pPre == NULL)    //Deleting first node
    {
        pLoc = head;
        if (pLoc->flink == NULL)    // Deleting the only node
        head = NULL;
        else      // Deleting first node
        {
            pNext = pLoc->flink;
            pNext->blink = NULL;
            head = pLoc->flink;
        }
    }
    else      //Deleting other nodes
    {
        pLoc = pPre->flink;
        if (pLoc->flink == NULL)     // Deleting last node
        pPre->flink = pLoc->flink;
        else     // Deleting middle node
        {
            pNext = pLoc->flink;
            pNext->blink = pLoc->blink;
            pPre->flink = pLoc->flink;
        }
    }
}

void insertNode(struct Node* pPre, int dataIn)
{
    struct Node* pNew = new Node;
    struct Node* pLoc;
    pNew->data = dataIn;
    if (pPre == NULL)      // Adding as first node
    {
        pNew->blink = NULL;
        if (head == NULL)     //Adding to empty list
        {
            pNew->flink = NULL;
            head = pNew;
        }
        else     // Adding before first node
        {
            pNew->flink = head;
            pLoc = head;
            pLoc->blink = pNew;
            head = pNew;
        }
    }
    else      // Adding at middle or end
    {
        pNew->flink = pPre->flink;
        pNew->blink = pPre;
        if (pPre->flink == NULL)      //Adding in end
        {
            pPre->flink = pNew;
        }
        else     // Adding at middle
        {
            pLoc = pPre->flink;
            pLoc->blink = pNew;
            pPre->flink = pNew;
        }
    }
}

struct Node* searchList(int target)     // 0->1->2->3->4
{
    struct Node* pLoc, * pPre = NULL;
    pLoc = head;
    if (head == NULL)      // Empty list
    {
        return NULL;
    }
    while (pLoc != NULL && target > pLoc->data)
    {
    pPre = pLoc;
    pLoc = pLoc->flink;
    }
    return pPre;
}

struct Node* locateList(int target)
{
    struct Node* pLoc, * pPre = NULL;
    pLoc = head;
    while (pLoc != NULL && target != pLoc->data)
    {
        pPre = pLoc;
        pLoc = pLoc->flink;
    }
    return pPre;
}

void displayList()
{
    struct Node* pLoc = head;
    struct Node* pPre = NULL;
    cout << " Nodes in original order:";
    while (pLoc != NULL)
    {
        pPre = pLoc;
        cout << pLoc->data << "\t";
        pLoc = pLoc->flink;
    }
    cout << " Nodes in reverse order:";
    while (pPre != NULL)
    {
        cout << pPre->data << "\t";
        pPre = pPre->blink;
    }
}

int main()
{
    struct Node* pPre = NULL;
    head = NULL;
    int data, choice = 1;
    do
    {
    cout << endl;
    cout << "1. Insert Node" << endl;
    cout << "2. Delete Node" << endl;
    cout << "3. Display List" << endl;
    cout << "4. Exit" << endl;
    cin >> choice;
    switch (choice)
    {
        case 1:
            cout << "Enter the value to be inserted to new node: ";
            cin >> data;
            pPre = searchList(data);
            insertNode(pPre, data);
            break;
        case 2:
            cout << "Enter the value to be deleted: ";
            cin >> data;
            pPre = locateList(data);
            deleteNode(pPre);
            cout << data << " deleted" << endl;
            break;
        case 3:
            displayList();
            break;
    }
    } while (choice != 4);
}
