#include "HookResultWindow.h"

HookResultWindow::HookResultWindow(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	

	QFile file("result.txt");
	if (!file.open(QFile::ReadOnly|QFile::Text)) {
		QMessageBox::warning(this, QStringLiteral("�ļ���ȡ����"), QStringLiteral("��ȡ�������:%1").arg(file.errorString()));
	}
	else 
	{
		QTextStream stream(&file);
		stream.setCodec("UTF-8");
		QStringList sitem = stream.readAll().split("<=====>\n");

		QStandardItemModel* res = new QStandardItemModel(this);

		res->setColumnCount(2);
		res->setHeaderData(0, Qt::Horizontal, QStringLiteral("������"));
		res->setHeaderData(1, Qt::Horizontal, QStringLiteral("�ı�"));

		int length = sitem.size();

		for (int i = 0;i < length;i++) {
			QStringList it = sitem[i].split(" => ");

			if (it.size() == 2) {
				QList<QStandardItem*>* row = new QList<QStandardItem*>;
				*row << new QStandardItem(it[0]) << new QStandardItem(it[1]);
				res->appendRow(*row);
			}
		}

		//ui.resTableView->setEditTriggers(QAbstractItemView::NoEditTriggers);//���ɱ༭
		ui.resTableView->setSelectionBehavior(QAbstractItemView::SelectRows);//����ѡ��ģʽΪѡ����
		ui.resTableView->setSelectionMode(QAbstractItemView::SingleSelection);//����ѡ�е���
		ui.resTableView->resizeColumnsToContents();//������
		ui.resTableView->setModel(res);
	}

	connect(ui.FindNextBtn, SIGNAL(clicked()), this, SLOT(FindNextBtn_Click()));
	connect(ui.AddCustomBtn, SIGNAL(clicked()), this, SLOT(AddCustomBtn_Click()));
}

HookResultWindow::~HookResultWindow()
{
}

void HookResultWindow::FindNextBtn_Click() {
	int row = ui.resTableView->currentIndex().row();
	QAbstractItemModel* model = ui.resTableView->model();
	QString findStr = ui.SearchBox->text();
	int rowsCount = model->rowCount();

	if (findStr == "") {
		QMessageBox::warning(this, QStringLiteral("��ʾ"), QStringLiteral("������Ҫ���ҵĹؼ��֣�"));
		return;
	}

	if (row == -1) {
		//û��ѡ�����ʱ��Ĭ�ϵ�һ����ʼ
		row == 0;
	}
	else {
		row++;//Ҫ����һ�п�ʼ
	}

	for (int i = row;i < rowsCount;i++) {
		QModelIndex index = model->index(i, 1);//ѡ���еڶ��е�����
		QVariant data = model->data(index);

		if (data.toString().contains(findStr, Qt::CaseInsensitive) == true) {
			ui.resTableView->setCurrentIndex(index);
			break;
		}
	}

	if (row >= rowsCount - 1) {
		QMessageBox::warning(this, QStringLiteral("�޽��"), QStringLiteral("������һ���޽�����Զ��������У�"));
		QModelIndex index = model->index(0, 1);
		ui.resTableView->setCurrentIndex(index);
	}

}

void HookResultWindow::AddCustomBtn_Click() {
	int row = ui.resTableView->currentIndex().row();
	if (row == -1) {
		QMessageBox::warning(this, QStringLiteral("��ʾ"), QStringLiteral("��ѡ��һ��Ҫע����������У�"));
		return;
	}

	QAbstractItemModel* model = ui.resTableView->model();
	QModelIndex index = model->index(row, 0);//ѡ���е�һ�е�����
	QVariant data = model->data(index);
	
	TextHost::InsertHook(processID,data.toString().toStdWString().c_str());
	QMessageBox::information(this, QStringLiteral("��ʾ"), QStringLiteral("������Զ���Hook�����룬������ҳ��ȷ�ϣ�"));
}
