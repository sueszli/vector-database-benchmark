#include "MisakaHookFinder.h"

MisakaHookFinder* s_this = nullptr;

static bool isOpenClipboard = false;
static bool isCustomAttach = false;
static QString customHookcode = "";
static uint64_t currentFun = -1;

MisakaHookFinder::MisakaHookFinder(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

	s_this = this;

	isHooking = false;
	
	GetProcessesList();

	fh = FindHooksHandle;

	pe = ProcessEventHandle;
	oct = OnCreateThreadHandle;
	ort = OnRemoveThreadHandle;
	opt = OutputTextHandle;

	connect(ui.AttachBtn, SIGNAL(clicked()), this, SLOT(AttachProcessBtn_Click()));
	connect(ui.CustomHookCodeBtn, SIGNAL(clicked()), this, SLOT(CustomHookCodeBtn_Click()));
	connect(ui.SearchForTextBtn, SIGNAL(clicked()), this, SLOT(SearchForTextBtn_Click()));
	connect(ui.SearchForHookBtn, SIGNAL(clicked()), this, SLOT(SearchForHookBtn_Click()));
	connect(ui.CopyHookCodeBtn, SIGNAL(clicked()), this, SLOT(CopyHookCodeBtn_Click()));
	connect(ui.ClipbordFlushBtn, SIGNAL(clicked()), this, SLOT(ClipbordFlushBtn_Click()));
	connect(ui.RemoveHookBtn, SIGNAL(clicked()), this, SLOT(RemoveHookBtn_Click()));
	connect(ui.HookFuncCombox, SIGNAL(currentIndexChanged(int)), this, SLOT(HookFunCombox_currentIndexChanged(int)));
	
	connect(s_this, SIGNAL(onConsoleBoxContextChange(QString)), this, SLOT(ConsoleBox_Change(QString)));
	connect(s_this, SIGNAL(onGameTextBoxContextChange(QString)), this, SLOT(GameTextBox_Change(QString)));
	connect(s_this, SIGNAL(onHookFunComboxChange(QString,int)), this, SLOT(HookFunCombox_Change(QString,int)));
	connect(s_this, SIGNAL(onOpenResWin()), this, SLOT(Reswin_Open()));
	connect(s_this, SIGNAL(onClipboardChange(QString)), this, SLOT(Clipboard_Change(QString)));
	connect(s_this, SIGNAL(onRemoveHookFun(uint64_t)), this, SLOT(HookFun_Remove(uint64_t)));
}

/*************************
	�˳��¼�
**************************/
void MisakaHookFinder::closeEvent(QCloseEvent* e) {
	
	if (QMessageBox::question(NULL, QStringLiteral("ȷ���˳�"), QStringLiteral("��ȷ��Ҫ�˳�MisakaHookFinder��"), QMessageBox::Yes, QMessageBox::No) == QMessageBox::Yes) {
		if (isHooking == true) {
			QVariant var = ui.ProcessesCombox->currentData();
			DWORD pid = (DWORD)var.toInt();
			TextHost::DetachProcess(pid);
		}
	}
	else {
		e->ignore();
	}
}

/*********************************
	��ȡϵͳ�����б���ʾ����Ͽ���
**********************************/
void MisakaHookFinder::GetProcessesList() {
	ui.ProcessesCombox->clear();

	PROCESSENTRY32 pe32;
	// ��ʹ������ṹ֮ǰ�����������Ĵ�С
	pe32.dwSize = sizeof(pe32);

	// ��ϵͳ�ڵ����н�����һ������
	HANDLE hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	if (hProcessSnap == INVALID_HANDLE_VALUE)
	{
		QMessageBox::information(NULL, QStringLiteral("����"), QStringLiteral("�޷�����ϵͳ���̣��볢��ʹ�ù���Ա���������!"));
		return;
	}

	// �������̿��գ�������ʾÿ�����̵���Ϣ
	BOOL bMore = ::Process32First(hProcessSnap, &pe32);
	while (bMore)
	{
		QString str1 = QString::fromWCharArray(pe32.szExeFile) + "-" + QString::number((int)pe32.th32ProcessID);
		ui.ProcessesCombox->addItem(str1, (int)pe32.th32ProcessID);

		bMore = ::Process32Next(hProcessSnap, &pe32);
	}

}

void MisakaHookFinder::AttachProcessBtn_Click() {

	QVariant var = ui.ProcessesCombox->currentData();
	DWORD pid = (DWORD)var.toInt();

	if (isHooking == false) {
		ui.ProcessesCombox->setEnabled(false);

		ui.ConsoleTextBox->appendPlainText(QStringLiteral("ע�����PID:") + var.toString());
		ui.AttachBtn->setText(QStringLiteral("����ע��"));

		TextHost::TextHostInit(pe, pe, oct, ort, opt);
		TextHost::InjectProcess(pid);

		isHooking = true;
		isCustomAttach = false;
	}
	else {
		TextHost::DetachProcess(pid);

		ui.ConsoleTextBox->appendPlainText(QStringLiteral("ȡ��ע�����PID:") + var.toString());
		ui.AttachBtn->setText(QStringLiteral("ע�����"));

		ui.ProcessesCombox->setEnabled(true);
		isHooking = false;
		isCustomAttach = false;
	}
}


void MisakaHookFinder::CustomHookCodeBtn_Click() {
	if (isHooking == false) {
		QMessageBox::information(NULL, QStringLiteral("��ʾ"), QStringLiteral("��ʹ��ָ��������ֱ��ע����Ϸ����ʱ��������������ʽ�����仯����ο�˵����̳̣�ȷ��֪Ϥ����ʹ�ã�"));
	}

	bool isOK;
	QString text = QInputDialog::getText(NULL, QStringLiteral("����������"),
		QStringLiteral("�������Զ���������:"),
		QLineEdit::Normal,
		"",
		&isOK);

	if (isOK) {
		QVariant var = ui.ProcessesCombox->currentData();
		DWORD pid = (DWORD)var.toInt();

		if (isHooking == false) {
			//��ûע����̵�����£��ȳ�ʼ������ע�룬Ȼ����жԱȣ�������Ҫ���ֱ���Ƴ�
			TextHost::TextHostInit(pe, pe, oct, ort, opt);
			TextHost::InjectProcess(pid);
			ui.ConsoleTextBox->appendPlainText(QStringLiteral("ע�����PID:") + var.toString());
			ui.AttachBtn->setText(QStringLiteral("����ע��"));
			customHookcode = text;
			isCustomAttach = true;
			isHooking = true;
		}
		TextHost::InsertHook(pid, text.toStdWString().c_str());
	}
}

void MisakaHookFinder::SearchForTextBtn_Click() {
	QMessageBox::information(NULL, QStringLiteral("ʹ����֪"), QStringLiteral("����һ�����ȶ��Ĺ��ܣ�ʹ��ǰ������Ϸ�д浵��ֹ��Ϸ���������\nʹ��ǰ��鿴������ܵ����˵���ͽ̳̣�������ڱ���Ŀ��Githubҳ�����ҵ���"));

	QDialog dialog(this);
	QFormLayout form(&dialog);
	form.addRow(new QLabel(QStringLiteral("���������������ı����ַ�����ҳ:")));
	
	QString value1 = QString(QStringLiteral("�������ı�: "));
	QLineEdit* linebox1 = new QLineEdit(&dialog);
	form.addRow(value1, linebox1);
	
	QString value2 = QString(QStringLiteral("����ҳ: "));
	QSpinBox* spinbox2 = new QSpinBox(&dialog);
	spinbox2->setMaximum(100000);
	spinbox2->setMinimum(0);
	spinbox2->setValue(932);
	form.addRow(value2, spinbox2);

	//ȷ��ȡ��Ť
	QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel,
		Qt::Horizontal, &dialog);
	form.addRow(&buttonBox);
	QObject::connect(&buttonBox, SIGNAL(accepted()), &dialog, SLOT(accept()));
	QObject::connect(&buttonBox, SIGNAL(rejected()), &dialog, SLOT(reject()));

	//����ȷ��Ť
	if (dialog.exec() == QDialog::Accepted) {
		QVariant var = ui.ProcessesCombox->currentData();
		DWORD pid = (DWORD)var.toInt();

		TextHost::SearchForText(pid, linebox1->text().toStdWString().c_str(), spinbox2->value());
	}

}

void MisakaHookFinder::SearchForHookBtn_Click() {
	
	QMessageBox::information(NULL, QStringLiteral("ʹ����֪"), QStringLiteral("����һ�����ȶ��Ĺ��ܣ�ʹ��ǰ������Ϸ�д浵��ֹ��Ϸ���������\nʹ��ǰ��鿴������ܵ����˵���ͽ̳̣�������ڱ���Ŀ��Githubҳ�����ҵ���"));

	SearchParam sp;
	sp.length = 0;//Ĭ������������÷��� https://github.com/Artikash/Textractor/blob/master/GUI/mainwindow.cpp 344��
	QVariant var = ui.ProcessesCombox->currentData();
	DWORD pid = (DWORD)var.toInt();

	TextHost::SearchForHooks(pid,&sp,fh);
	PrintToUI(0, QStringLiteral("��ʼ�������ڽ�������20s����ˢ�¼�����Ϸ���ı��Թ��ڴ���ҡ�"));
}

void MisakaHookFinder::CopyHookCodeBtn_Click() {
	QClipboard* clipboard = QApplication::clipboard();   //��ȡϵͳ������ָ��
	clipboard->setText(ui.HookFuncCombox->currentText());                          //���ü���������

	QMessageBox::information(NULL, QStringLiteral("��ʾ"), QStringLiteral("���������뵽������ɹ���"));
	
}

void MisakaHookFinder::ClipbordFlushBtn_Click() {
	if (isOpenClipboard == true) {
		isOpenClipboard = false;
		ui.ClipbordFlushBtn->setText(QStringLiteral("�������������"));
		PrintToUI(0, QStringLiteral("�ѹرռ��������"));
	}
	else {
		isOpenClipboard = true;
		ui.ClipbordFlushBtn->setText(QStringLiteral("�رռ��������"));
		PrintToUI(0, QStringLiteral("�ѿ�����������£����ڸ÷�������������ı���ʵʱˢ�µ������壡"));
	}
}

void MisakaHookFinder::RemoveHookBtn_Click() {
	QVariant var = ui.ProcessesCombox->currentData();
	DWORD pid = (DWORD)var.toInt();
	TextHost::RemoveHook(pid, GetAddressByHookComboxContent(ui.HookFuncCombox->currentText()));

	//�Ƴ���Ͽ���
	ui.HookFuncCombox->removeItem(ui.HookFuncCombox->currentIndex());
	ui.HookFuncCombox->setCurrentIndex(0);
	PrintToUI(0, QStringLiteral("���Ƴ�������ӣ�"));
}



void MisakaHookFinder::FindHooksHandle() {
	PrintToUI(0, QStringLiteral("�����ѽ��������ڼ��ؽ��..."));
	
	OpenResultWin();
}

/**************************
	ƴ����Ϣ�ַ���
**************************/
QString MisakaHookFinder::TextThreadString(int64_t thread_id, DWORD processId, uint64_t addr, uint64_t context, uint64_t subcontext, LPCWSTR name, LPCWSTR hookcode)
{
	return QString("%1:%2:%3:%4:%5:%6:%7").arg(
		QString::number(thread_id, 16),
		QString::number(processId, 16),
		QString::number(addr, 16),
		QString::number(context, 16),
		QString::number(subcontext, 16)
	).toUpper().arg(name).arg(hookcode);
}

/**************************
	����Hook������Ͽ��е����ݵõ���ַ�������Ƴ�����
***************************/
uint64_t MisakaHookFinder::GetAddressByHookComboxContent(QString str) {
	QStringList sitem = str.split(":");
	return sitem[2].toULongLong((bool*)nullptr, 16);
}


void MisakaHookFinder::HookFunCombox_currentIndexChanged(int index) {
	QStringList sitem = ui.HookFuncCombox->currentText().split(":");
	currentFun = sitem[0].toULongLong((bool*)nullptr,16);
	ui.TextOutPutBox->clear();
}

void MisakaHookFinder::ProcessEventHandle(DWORD processId) {
	PrintToUI(0,QStringLiteral("�����¼�PID:") + QString::fromStdString(std::to_string(processId)));
}

void MisakaHookFinder::OnCreateThreadHandle(int64_t thread_id, DWORD processId, uint64_t addr, uint64_t context, uint64_t subcontext, LPCWSTR name, LPCWSTR hookcode) {
	PrintToUI(0,QStringLiteral("���Hook�߳�ID") + QString::fromStdString(std::to_string(thread_id)));

	if (isCustomAttach == true) {
		//��ʹ���ض�������ע�������£����������벻һ�µ�ֱ��ɾ
		QString currentHookcode = QString::fromStdWString(hookcode);
		QStringList sitem = currentHookcode.split(":");
		if (customHookcode.compare(sitem[0], Qt::CaseInsensitive) != 0) {
			PrintToUI(0, QStringLiteral("����ɾ��Hook:") + QString::fromStdWString(hookcode));
			RemoveHookFun(addr);
		}
		else {
			QString str = TextThreadString(thread_id, processId, addr, context, subcontext, name, hookcode);
			AddHookFunc(str, thread_id);
		}
	}
	else {
		QString str = TextThreadString(thread_id, processId, addr, context, subcontext, name, hookcode);
		AddHookFunc(str, thread_id);
	}

	
}

void MisakaHookFinder::OnRemoveThreadHandle(int64_t thread_id) {
	PrintToUI(0,QStringLiteral("�Ƴ�Hook�߳�ID") + QString::fromStdString(std::to_string(thread_id)));
}

void MisakaHookFinder::OutputTextHandle(int64_t thread_id, LPCWSTR output){
	if (thread_id == 0) {
		//����̨����̣߳���Ȼ��ӡ������̨
		PrintToUI(0, QString::fromStdWString(output));
	}

	if (thread_id == currentFun) {
		//��ǰ������߳�����Ͽ�ѡ�е�����߳�
		PrintToUI(1, QString::fromStdWString(output));

		if (isOpenClipboard == true) {
			FlushClipboard(QString::fromStdWString(output));
		}
	}
}

void MisakaHookFinder::PrintToUI(int editboxID, QString str) {
	switch (editboxID)
	{
	case 0:
		s_this->emitConsoleBoxSignal(str);
		break;
	case 1:
		s_this->emitGameTextBoxSignal(str);
		break;
	default:
		break;
	}
}

void MisakaHookFinder::AddHookFunc(QString str, int data) {
	s_this->emitHookFunComboxSignal(str,data);
}

void MisakaHookFinder::emitHookFunComboxSignal(QString str,int data) {
	emit this->onHookFunComboxChange(str,data);
}

void MisakaHookFinder::emitConsoleBoxSignal(QString str)
{
	emit this->onConsoleBoxContextChange(str);
}

void MisakaHookFinder::emitGameTextBoxSignal(QString str)
{
	emit this->onGameTextBoxContextChange(str);
}

void MisakaHookFinder::ConsoleBox_Change(QString str) {
	ui.ConsoleTextBox->appendPlainText(str);
}

void MisakaHookFinder::GameTextBox_Change(QString str) {
	ui.TextOutPutBox->appendPlainText(str);
	ui.TextOutPutBox->appendPlainText("==================");
}

void MisakaHookFinder::HookFunCombox_Change(QString str, int data) {
	ui.HookFuncCombox->addItem(str,data);
}

void MisakaHookFinder::OpenResultWin() {
	s_this->emitResultWinSignal();
}

void MisakaHookFinder::emitResultWinSignal() {
	emit this->onOpenResWin();
}

void MisakaHookFinder::Reswin_Open() {
	QVariant var = ui.ProcessesCombox->currentData();
	DWORD pid = (DWORD)var.toInt();
	HookResultWindow* hrw = new HookResultWindow();
	hrw->processID = pid;
	hrw->show();
}

void  MisakaHookFinder::FlushClipboard(QString str) {
	s_this->emitClipboardSignal(str);
}


void MisakaHookFinder::emitClipboardSignal(QString str) {
	emit this->onClipboardChange(str);
}

void MisakaHookFinder::Clipboard_Change(QString str) {
	QClipboard* clipboard = QApplication::clipboard();   //��ȡϵͳ������ָ��
	clipboard->setText(str);
}

void MisakaHookFinder::RemoveHookFun(uint64_t thread) {
	s_this->emitRemoveHookFunSignal(thread);
}

void MisakaHookFinder::emitRemoveHookFunSignal(uint64_t thread) {
	emit this->onRemoveHookFun(thread);
}

void MisakaHookFinder::HookFun_Remove(uint64_t thread) {
	QVariant var = ui.ProcessesCombox->currentData();
	DWORD pid = (DWORD)var.toInt();
	TextHost::RemoveHook(pid,thread);
}
