
// PS_ServAppDoc.cpp : CPS_ServAppDoc 클래스의 구현
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


// CPS_ServAppDoc 생성/소멸

CPS_ServAppDoc::CPS_ServAppDoc()
{
	// TODO: 여기에 일회성 생성 코드를 추가합니다.

}

CPS_ServAppDoc::~CPS_ServAppDoc()
{
}

BOOL CPS_ServAppDoc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: 여기에 재초기화 코드를 추가합니다.
	// SDI 문서는 이 문서를 다시 사용합니다.

	return TRUE;
}




// CPS_ServAppDoc serialization

void CPS_ServAppDoc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: 여기에 저장 코드를 추가합니다.
	}
	else
	{
		// TODO: 여기에 로딩 코드를 추가합니다.
	}
}


// CPS_ServAppDoc 진단

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


// CPS_ServAppDoc 명령
