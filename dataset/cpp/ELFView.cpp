#include "ELFView.h"
#include "string_format.h"
#include "lexical_cast_ex.h"

#include <QLabel>
#include <QFontMetrics>
#include <QSplitter>

template <typename ElfType>
CELFView<ElfType>::CELFView(QMdiArea* parent)
    : QMdiSubWindow(parent)
{
	resize(568, 457);
	parent->addSubWindow(this);
	setWindowTitle("ELF File Viewer");

	m_centralwidget = new QWidget(this);
	m_layout = new QHBoxLayout(m_centralwidget);
	m_treeWidget = new QTreeWidget(m_centralwidget);

	m_groupBox = new QGroupBox(m_centralwidget);
	auto groupBoxLayout = new QHBoxLayout();
	m_groupBox->setLayout(groupBoxLayout);

	{
		QSplitter* splitter = new QSplitter(m_centralwidget);
		splitter->addWidget(m_treeWidget);
		splitter->addWidget(m_groupBox);
		splitter->setSizes({200, 368});
		splitter->setStretchFactor(0, 0);
		splitter->setStretchFactor(1, 1);
		m_layout->addWidget(splitter);
	}

	setWidget(m_centralwidget);

	QTreeWidgetItem* colHeader = m_treeWidget->headerItem();
	colHeader->setText(0, "ELF");

	m_pHeaderView = new CELFHeaderView<ElfType>(this, groupBoxLayout);
	m_pSymbolView = new CELFSymbolView<ElfType>(this, groupBoxLayout);
	m_pSectionView = new CELFSectionView<ElfType>(this, groupBoxLayout);
	m_pProgramView = new CELFProgramView<ElfType>(this, groupBoxLayout);

	connect(m_treeWidget, &QTreeWidget::itemSelectionChanged, this, &CELFView::itemSelectionChanged);
}

template <typename ElfType>
void CELFView<ElfType>::Reset()
{
	m_pHeaderView->Reset();
	m_pSymbolView->Reset();
	m_pSectionView->Reset();
	m_pProgramView->Reset();
}

template <typename ElfType>
void CELFView<ElfType>::SetELF(ElfType* pELF)
{
	Reset();

	m_pELF = pELF;
	if(m_pELF == NULL) return;

	m_pHeaderView->SetELF(m_pELF);
	m_pSymbolView->SetELF(m_pELF);
	m_pSectionView->SetELF(m_pELF);
	m_pProgramView->SetELF(m_pELF);

	PopulateList();
}

template <typename ElfType>
void CELFView<ElfType>::resizeEvent(QResizeEvent* evt)
{
	QMdiSubWindow::resizeEvent(evt);
	m_pSectionView->ResizeEvent();
}

template <typename ElfType>
void CELFView<ElfType>::showEvent(QShowEvent* evt)
{
	QMdiSubWindow::showEvent(evt);
	widget()->show();
}

template <typename ElfType>
void CELFView<ElfType>::PopulateList()
{
	m_treeWidget->clear();

	QTreeWidgetItem* headRootItem = new QTreeWidgetItem(m_treeWidget, {"Header"});
	m_treeWidget->addTopLevelItem(headRootItem);

	QTreeWidgetItem* sectionsRootItem = new QTreeWidgetItem(m_treeWidget, {"Sections"});
	m_treeWidget->addTopLevelItem(sectionsRootItem);
	const auto& header = m_pELF->GetHeader();

	const char* sStrTab = (const char*)m_pELF->GetSectionData(header.nSectHeaderStringTableIndex);
	for(unsigned int i = 0; i < m_pELF->GetSectionHeaderCount(); i++)
	{
		std::string sDisplay;
		const char* sName(NULL);

		auto pSect = m_pELF->GetSection(i);

		if(sStrTab != NULL)
		{
			sName = sStrTab + pSect->nStringTableIndex;
		}
		else
		{
			sName = "";
		}

		if(strlen(sName))
		{
			sDisplay = sName;
		}
		else
		{
			sDisplay = ("Section ") + lexical_cast_uint<std::string>(i);
		}
		sectionsRootItem->addChild(new QTreeWidgetItem(sectionsRootItem, {sDisplay.c_str()}));
	}
	sectionsRootItem->setExpanded(true);

	m_hasPrograms = header.nProgHeaderCount != 0;
	if(m_hasPrograms)
	{
		QTreeWidgetItem* segmentsRootItem = new QTreeWidgetItem(m_treeWidget, {"Segments"});
		m_treeWidget->addTopLevelItem(segmentsRootItem);

		for(unsigned int i = 0; i < header.nProgHeaderCount; i++)
		{
			std::string sDisplay(("Segment ") + lexical_cast_uint<std::string>(i));
			segmentsRootItem->addChild(new QTreeWidgetItem(segmentsRootItem, {sDisplay.c_str()}));
		}
		segmentsRootItem->setExpanded(true);
	}

	m_hasSymbols = m_pELF->FindSection(".strtab") && m_pELF->FindSection(".symtab");
	if(m_hasSymbols)
	{
		QTreeWidgetItem* symbolsRootItem = new QTreeWidgetItem(m_treeWidget, {"Symbols"});
		m_treeWidget->addTopLevelItem(symbolsRootItem);
	}
}

template <typename ElfType>
void CELFView<ElfType>::itemSelectionChanged()
{
	m_pHeaderView->hide();
	m_pSectionView->hide();
	m_pProgramView->hide();
	m_pSymbolView->hide();

	auto selectedItems = m_treeWidget->selectedItems();
	auto item = !selectedItems.empty() ? selectedItems.at(0) : nullptr;
	if(!item)
		return;

	bool isRoot = item->parent() == nullptr;
	int rootIndex = -1;
	auto index = -1;
	if(!isRoot)
	{
		index = item->parent()->indexOfChild(item);
		rootIndex = m_treeWidget->indexOfTopLevelItem(item->parent());
	}
	else
	{
		rootIndex = m_treeWidget->indexOfTopLevelItem(item);
	}
	if(rootIndex == 0)
	{
		m_pHeaderView->show();
	}
	else if(rootIndex == 1)
	{
		if(index > -1)
		{
			m_pSectionView->SetSection(index);
			m_pSectionView->show();
		}
	}
	else if(rootIndex != -1)
	{
		if(rootIndex == 2 && m_hasPrograms)
		{
			if(index > -1)
			{
				m_pProgramView->SetProgram(index);
				m_pProgramView->show();
			}
		}
		else
		{
			m_pSymbolView->show();
		}
	}
}

template class CELFView<CELF32>;
template class CELFView<CELF64>;
