
// PS_ServAppDoc.cpp : CPS_ServAppDoc Ŭ������ ����
//

#include "stdafx.h"
#include "PS_ServApp.h"

#include "PS_ServAppDoc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CPS_ServAppDoc

IMPLEMENT_DYNCREATE(CPS_ServAppDoc, CDocument)

BEGIN_MESSAGE_MAP(CPS_ServAppDoc, CDocument)
END_MESSAGE_MAP()


// CPS_ServAppDoc ����/�Ҹ�

CPS_ServAppDoc::CPS_ServAppDoc()
{
	// TODO: ���⿡ ��ȸ�� ���� �ڵ带 �߰��մϴ�.

}

CPS_ServAppDoc::~CPS_ServAppDoc()
{
}

BOOL CPS_ServAppDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: ���⿡ ���ʱ�ȭ �ڵ带 �߰��մϴ�.
	// SDI ������ �� ������ �ٽ� ����մϴ�.

	return TRUE;
}




// CPS_ServAppDoc serialization

void CPS_ServAppDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: ���⿡ ���� �ڵ带 �߰��մϴ�.
	}
	else
	{
		// TODO: ���⿡ �ε� �ڵ带 �߰��մϴ�.
	}
}


// CPS_ServAppDoc ����

#ifdef _DEBUG
void CPS_ServAppDoc::AssertValid() const
{
	CDocument::AssertValid();
}

void CPS_ServAppDoc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG


// CPS_ServAppDoc ���
