/*  PCSX2 - PS2 Emulator for PCs
 *  Copyright (C) 2002-2022  PCSX2 Dev Team
 *
 *  PCSX2 is free software: you can redistribute it and/or modify it under the terms
 *  of the GNU Lesser General Public License as published by the Free Software Found-
 *  ation, either version 3 of the License, or (at your option) any later version.
 *
 *  PCSX2 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 *  PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with PCSX2.
 *  If not, see <http://www.gnu.org/licenses/>.
 */

#include "PrecompiledHeader.h"

#include "DisassemblyWidget.h"

#include "DebugTools/DebugInterface.h"
#include "DebugTools/DisassemblyManager.h"
#include "DebugTools/Breakpoints.h"
#include "DebugTools/MipsAssembler.h"
#include "demangler/demangler.h"

#include "QtUtils.h"
#include "QtHost.h"
#include <QtGui/QMouseEvent>
#include <QtWidgets/QMenu>
#include <QtGui/QClipboard>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QMessageBox>

using namespace QtUtils;

DisassemblyWidget::DisassemblyWidget(QWidget* parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	connect(this, &DisassemblyWidget::customContextMenuRequested, this, &DisassemblyWidget::customMenuRequested);
}

DisassemblyWidget::~DisassemblyWidget() = default;

void DisassemblyWidget::contextCopyAddress()
{
	QGuiApplication::clipboard()->setText(FetchSelectionInfo(SelectionInfo::ADDRESS));
}

void DisassemblyWidget::contextCopyInstructionHex()
{
	QGuiApplication::clipboard()->setText(FetchSelectionInfo(SelectionInfo::INSTRUCTIONHEX));
}

void DisassemblyWidget::contextCopyInstructionText()
{
	QGuiApplication::clipboard()->setText(FetchSelectionInfo(SelectionInfo::INSTRUCTIONTEXT));
}

void DisassemblyWidget::contextAssembleInstruction()
{
	if (!m_cpu->isCpuPaused())
	{
		QMessageBox::warning(this, tr("Assemble Error"), tr("Unable to change assembly while core is running"));
		return;
	}

	DisassemblyLineInfo line;
	bool ok;
	m_disassemblyManager.getLine(m_selectedAddressStart, false, line);
	QString instruction = QInputDialog::getText(this, tr("Assemble Instruction"), "",
		QLineEdit::Normal, QString("%1 %2").arg(line.name.c_str()).arg(line.params.c_str()), &ok);

	if (!ok)
		return;

	u32 encodedInstruction;
	std::string errorText;
	bool valid = MipsAssembleOpcode(instruction.toLocal8Bit().constData(), m_cpu, m_selectedAddressStart, encodedInstruction, errorText);

	if (!valid)
	{
		QMessageBox::warning(this, tr("Assemble Error"), QString::fromStdString(errorText));
		return;
	}
	else
	{
		Host::RunOnCPUThread([this, start = m_selectedAddressStart, end = m_selectedAddressEnd, cpu = m_cpu, val = encodedInstruction] {
			for (u32 i = start; i <= end; i += 4)
			{
				this->m_nopedInstructions.insert({i, cpu->read32(i)});
				cpu->write32(i, val);
			}
			QtHost::RunOnUIThread([this] { VMUpdate(); });
		});
	}
}

void DisassemblyWidget::contextNoopInstruction()
{
	Host::RunOnCPUThread([this, start = m_selectedAddressStart, end = m_selectedAddressEnd, cpu = m_cpu] {
		for (u32 i = start; i <= end; i += 4)
		{
			this->m_nopedInstructions.insert({i, cpu->read32(i)});
			cpu->write32(i, 0x00);
		}
		QtHost::RunOnUIThread([this] { VMUpdate(); });
	});
}

void DisassemblyWidget::contextRestoreInstruction()
{
	Host::RunOnCPUThread([this, start = m_selectedAddressStart, end = m_selectedAddressEnd, cpu = m_cpu] {
		for (u32 i = start; i <= end; i += 4)
		{
			if (this->m_nopedInstructions.find(i) != this->m_nopedInstructions.end())
			{
				cpu->write32(i, this->m_nopedInstructions[i]);
				this->m_nopedInstructions.erase(i);
			}
		}
		QtHost::RunOnUIThread([this] { VMUpdate(); });
	});
}

void DisassemblyWidget::contextRunToCursor()
{
	Host::RunOnCPUThread([&] {
		CBreakPoints::AddBreakPoint(m_cpu->getCpuType(), m_selectedAddressStart, true);
		m_cpu->resumeCpu();
	});
}

void DisassemblyWidget::contextJumpToCursor()
{
	m_cpu->setPc(m_selectedAddressStart);
	this->repaint();
}

void DisassemblyWidget::contextToggleBreakpoint()
{
	if (!m_cpu->isAlive())
		return;

	if (CBreakPoints::IsAddressBreakPoint(m_cpu->getCpuType(), m_selectedAddressStart))
	{
		Host::RunOnCPUThread([&] { CBreakPoints::RemoveBreakPoint(m_cpu->getCpuType(), m_selectedAddressStart); });
	}
	else
	{
		Host::RunOnCPUThread([&] { CBreakPoints::AddBreakPoint(m_cpu->getCpuType(), m_selectedAddressStart); });
	}

	breakpointsChanged();
	this->repaint();
}

void DisassemblyWidget::contextFollowBranch()
{
	DisassemblyLineInfo line;

	m_disassemblyManager.getLine(m_selectedAddressStart, true, line);

	if (line.type == DISTYPE_OPCODE || line.type == DISTYPE_MACRO)
	{
		if (line.info.isBranch)
			gotoAddress(line.info.branchTarget);
		else if (line.info.hasRelevantAddress)
			gotoAddress(line.info.releventAddress);
	}
}

void DisassemblyWidget::contextGoToAddress()
{
	bool ok;
	const QString targetString = QInputDialog::getText(this, tr("Go to address"), "",
		QLineEdit::Normal, "", &ok);

	if (!ok)
		return;

	const u32 targetAddress = targetString.toUInt(&ok, 16) & ~3;

	if (!ok)
	{
		QMessageBox::warning(this, tr("Go to address error"), tr("Invalid address"));
		return;
	}

	gotoAddress(targetAddress);
}

void DisassemblyWidget::contextAddFunction()
{
	// Get current function
	const u32 curAddress = m_selectedAddressStart;
	const u32 curFuncAddr = m_cpu->GetSymbolMap().GetFunctionStart(m_selectedAddressStart);
	QString optionaldlgText;

	if (curFuncAddr != SymbolMap::INVALID_ADDRESS)
	{
		if (curFuncAddr == curAddress) // There is already a function here
		{
			QMessageBox::warning(this, tr("Add Function Error"), tr("A function entry point already exists here. Consider renaming instead."));
		}
		else
		{
			const u32 prevSize = m_cpu->GetSymbolMap().GetFunctionSize(curFuncAddr);
			u32 newSize = curAddress - curFuncAddr;

			bool ok;
			QString funcName = QInputDialog::getText(this, tr("Add Function"),
				tr("Function will be (0x%1) instructions long.\nEnter function name").arg(prevSize - newSize, 0, 16), QLineEdit::Normal, "", &ok);
			if (!ok)
				return;

			m_cpu->GetSymbolMap().SetFunctionSize(curFuncAddr, newSize); // End the current function to where we selected
			newSize = prevSize - newSize;
			m_cpu->GetSymbolMap().AddFunction(funcName.toLocal8Bit().constData(), curAddress, newSize);
			m_cpu->GetSymbolMap().SortSymbols();
		}
	}
	else
	{
		bool ok;
		QString funcName = QInputDialog::getText(this, "Add Function",
			tr("Function will be (0x%1) instructions long.\nEnter function name").arg(m_selectedAddressEnd + 4 - m_selectedAddressStart, 0, 16), QLineEdit::Normal, "", &ok);
		if (!ok)
			return;

		m_cpu->GetSymbolMap().AddFunction(funcName.toLocal8Bit().constData(), m_selectedAddressStart, m_selectedAddressEnd + 4 - m_selectedAddressStart);
		m_cpu->GetSymbolMap().SortSymbols();
	}
}

void DisassemblyWidget::contextRemoveFunction()
{
	u32 curFuncAddr = m_cpu->GetSymbolMap().GetFunctionStart(m_selectedAddressStart);

	if (curFuncAddr != SymbolMap::INVALID_ADDRESS)
	{
		u32 previousFuncAddr = m_cpu->GetSymbolMap().GetFunctionStart(curFuncAddr - 4);
		if (previousFuncAddr != SymbolMap::INVALID_ADDRESS)
		{
			// Extend the previous function to replace the spot of the function that is going to be removed
			u32 expandedSize = m_cpu->GetSymbolMap().GetFunctionSize(previousFuncAddr) + m_cpu->GetSymbolMap().GetFunctionSize(curFuncAddr);
			m_cpu->GetSymbolMap().SetFunctionSize(previousFuncAddr, expandedSize);
		}

		m_cpu->GetSymbolMap().RemoveFunction(curFuncAddr);
		m_cpu->GetSymbolMap().SortSymbols();
	}
}

void DisassemblyWidget::contextRenameFunction()
{
	const u32 curFuncAddress = m_cpu->GetSymbolMap().GetFunctionStart(m_selectedAddressStart);
	if (curFuncAddress != SymbolMap::INVALID_ADDRESS)
	{
		bool ok;
		QString funcName = QInputDialog::getText(this, tr("Rename Function"), tr("Function name"), QLineEdit::Normal, m_cpu->GetSymbolMap().GetLabelName(curFuncAddress).c_str(), &ok);
		if (!ok)
			return;

		if (funcName.isEmpty())
		{
			QMessageBox::warning(this, tr("Rename Function Error"), tr("Function name cannot be nothing."));
		}
		else
		{
			m_cpu->GetSymbolMap().SetLabelName(funcName.toLocal8Bit().constData(), curFuncAddress);
			m_cpu->GetSymbolMap().SortSymbols();
			this->repaint();
		}
	}
	else
	{
		QMessageBox::warning(this, tr("Rename Function Error"), tr("No function / symbol is currently selected."));
	}
}

void DisassemblyWidget::contextStubFunction()
{
	const u32 curFuncAddress = m_cpu->GetSymbolMap().GetFunctionStart(m_selectedAddressStart);
	if (curFuncAddress != SymbolMap::INVALID_ADDRESS)
	{
		Host::RunOnCPUThread([this, curFuncAddress, cpu = m_cpu] {
			this->m_stubbedFunctions.insert({curFuncAddress, {cpu->read32(curFuncAddress), cpu->read32(curFuncAddress + 4)}});
			cpu->write32(curFuncAddress, 0x03E00008); // jr $ra
			cpu->write32(curFuncAddress + 4, 0x00000000); // nop
			QtHost::RunOnUIThread([this] { VMUpdate(); });
		});
	}
	else // Stub the current opcode instead
	{
		Host::RunOnCPUThread([this, cpu = m_cpu] {
			this->m_stubbedFunctions.insert({m_selectedAddressStart, {cpu->read32(m_selectedAddressStart), cpu->read32(m_selectedAddressStart + 4)}});
			cpu->write32(m_selectedAddressStart, 0x03E00008); // jr $ra
			cpu->write32(m_selectedAddressStart + 4, 0x00000000); // nop
			QtHost::RunOnUIThread([this] { VMUpdate(); });
		});
	}
}

void DisassemblyWidget::contextRestoreFunction()
{
	const u32 curFuncAddress = m_cpu->GetSymbolMap().GetFunctionStart(m_selectedAddressStart);
	if (curFuncAddress != SymbolMap::INVALID_ADDRESS && m_stubbedFunctions.find(curFuncAddress) != m_stubbedFunctions.end())
	{
		Host::RunOnCPUThread([this, curFuncAddress, cpu = m_cpu] {
			cpu->write32(curFuncAddress, std::get<0>(this->m_stubbedFunctions[curFuncAddress]));
			cpu->write32(curFuncAddress + 4, std::get<1>(this->m_stubbedFunctions[curFuncAddress]));
			this->m_stubbedFunctions.erase(curFuncAddress);
			QtHost::RunOnUIThread([this] { VMUpdate(); });
		});
	}
	else if (m_stubbedFunctions.find(m_selectedAddressStart) != m_stubbedFunctions.end())
	{
		Host::RunOnCPUThread([this, cpu = m_cpu] {
			cpu->write32(m_selectedAddressStart, std::get<0>(this->m_stubbedFunctions[m_selectedAddressStart]));
			cpu->write32(m_selectedAddressStart + 4, std::get<1>(this->m_stubbedFunctions[m_selectedAddressStart]));
			this->m_stubbedFunctions.erase(m_selectedAddressStart);
			QtHost::RunOnUIThread([this] { VMUpdate(); });
		});
	}
	else
	{
		QMessageBox::warning(this, tr("Restore Function Error"), tr("Unable to stub selected address."));
	}
}
void DisassemblyWidget::SetCpu(DebugInterface* cpu)
{
	m_cpu = cpu;
	m_disassemblyManager.setCpu(cpu);
}

QString DisassemblyWidget::GetLineDisasm(u32 address)
{
	DisassemblyLineInfo lineInfo;
	m_disassemblyManager.getLine(address, true, lineInfo);
	return QString("%1 %2").arg(lineInfo.name.c_str()).arg(lineInfo.params.c_str());
};

// Here we go!
void DisassemblyWidget::paintEvent(QPaintEvent* event)
{
	QPainter painter(this);

	const u32 w = painter.device()->width() - 1;
	const u32 h = painter.device()->height() - 1;

	// Get the current font size
	const QFontMetrics fm = painter.fontMetrics();

	// Get the row height
	m_rowHeight = fm.height() + 2;

	// Find the amount of visible rows
	m_visibleRows = h / m_rowHeight;

	m_disassemblyManager.analyze(m_visibleStart, m_disassemblyManager.getNthNextAddress(m_visibleStart, m_visibleRows) - m_visibleStart);

	// Draw the rows
	bool inSelectionBlock = false;
	bool alternate = m_visibleStart % 8;

	const u32 curPC = m_cpu->getPC(); // Get the PC here, because it'll change when we are drawing and make it seem like there are two PCs

	for (u32 i = 0; i <= m_visibleRows; i++)
	{
		const u32 rowAddress = (i * 4) + m_visibleStart;
		// Row backgrounds

		if (inSelectionBlock || (m_selectedAddressStart <= rowAddress && rowAddress <= m_selectedAddressEnd))
		{
			painter.fillRect(0, i * m_rowHeight, w, m_rowHeight, this->palette().highlight());
			inSelectionBlock = m_selectedAddressEnd != rowAddress;
		}
		else
		{
			painter.fillRect(0, i * m_rowHeight, w, m_rowHeight, alternate ? this->palette().base() : this->palette().alternateBase());
		}

		// Row text
		painter.setPen(GetAddressFunctionColor(rowAddress));
		QString lineString = DisassemblyStringFromAddress(rowAddress, painter.font(), curPC, rowAddress == m_selectedAddressStart);

		painter.drawText(2, i * m_rowHeight, w, m_rowHeight, Qt::AlignLeft, lineString);

		// Breakpoint marker
		bool enabled;
		if (CBreakPoints::IsAddressBreakPoint(m_cpu->getCpuType(), rowAddress, &enabled) && !CBreakPoints::IsTempBreakPoint(m_cpu->getCpuType(), rowAddress))
		{
			if (enabled)
			{
				painter.setPen(Qt::green);
				painter.drawText(2, i * m_rowHeight, w, m_rowHeight, Qt::AlignLeft, "\u25A0");
			}
			else
			{
				painter.drawText(2, i * m_rowHeight, w, m_rowHeight, Qt::AlignLeft, "\u2612");
			}
		}
		alternate = !alternate;
	}
	// Draw the branch lines
	// This is where it gets a little scary
	// It's been mostly copied from the wx implementation

	u32 visibleEnd = m_disassemblyManager.getNthNextAddress(m_visibleStart, m_visibleRows);
	std::vector<BranchLine> branchLines = m_disassemblyManager.getBranchLines(m_visibleStart, visibleEnd - m_visibleStart);

	s32 branchCount = 0;
	s32 skippedBranches = 0;
	for (const auto& branchLine : branchLines)
	{
		if (branchCount == 5)
			break;
		const int winBottom = this->height();

		const int x = this->width() - 10 - ((std::max(0, branchLine.laneIndex - skippedBranches)) * 10);

		int top, bottom;
		// If the start is technically 'above' our address view
		if (branchLine.first < m_visibleStart)
		{
			top = -1;
		}
		// If the start is technically 'below' our address view
		else if (branchLine.first >= visibleEnd)
		{
			top = winBottom + 1;
		}
		else
		{
			// Explaination
			// ((branchLine.first - m_visibleStart) -> Find the amount of bytes from the top of the view
			// / 4 -> Convert that into rowss in instructions
			// * m_rowHeight -> convert that into rows in pixels
			// + (m_rowHeight / 2) -> Add half a row in pixels to center the arrow
			top = (((branchLine.first - m_visibleStart) / 4) * m_rowHeight) + (m_rowHeight / 2);
		}

		if (branchLine.second < m_visibleStart)
		{
			bottom = -1;
		}
		else if (branchLine.second >= visibleEnd)
		{
			bottom = winBottom + 1;
		}
		else
		{
			bottom = (((branchLine.second - m_visibleStart) / 4) * m_rowHeight) + (m_rowHeight / 2);
		}

		if ((top < 0 && bottom < 0) || (top > winBottom && bottom > winBottom) || (top < 0 && bottom > winBottom) || (top > winBottom && bottom < 0))
		{
			skippedBranches++;
			continue;
		}

		branchCount++;

		if (branchLine.first == m_selectedAddressStart || branchLine.second == m_selectedAddressStart)
		{
			painter.setPen(QColor(0xFF257AFA));
		}
		else
		{
			painter.setPen(QColor(0xFFFF3020));
		}

		if (top < 0) // first is not visible, but second is
		{
			painter.drawLine(x - 2, bottom, x + 2, bottom);
			painter.drawLine(x + 2, bottom, x + 2, 0);

			if (branchLine.type == LINE_DOWN)
			{
				painter.drawLine(x, bottom - 4, x - 4, bottom);
				painter.drawLine(x - 4, bottom, x + 1, bottom + 5);
			}
		}
		else if (bottom > winBottom) // second is not visible, but first is
		{
			painter.drawLine(x - 2, top, x + 2, top);
			painter.drawLine(x + 2, top, x + 2, winBottom);

			if (branchLine.type == LINE_UP)
			{
				painter.drawLine(x, top - 4, x - 4, top);
				painter.drawLine(x - 4, top, x + 1, top + 5);
			}
		}
		else
		{ // both are visible
			if (branchLine.type == LINE_UP)
			{
				painter.drawLine(x - 2, bottom, x + 2, bottom);
				painter.drawLine(x + 2, bottom, x + 2, top);
				painter.drawLine(x + 2, top, x - 4, top);

				painter.drawLine(x, top - 4, x - 4, top);
				painter.drawLine(x - 4, top, x + 1, top + 5);
			}
			else
			{
				painter.drawLine(x - 2, top, x + 2, top);
				painter.drawLine(x + 2, top, x + 2, bottom);
				painter.drawLine(x + 2, bottom, x - 4, bottom);

				painter.drawLine(x, bottom - 4, x - 4, bottom);
				painter.drawLine(x - 4, bottom, x + 1, bottom + 5);
			}
		}
	}
	// Draw a border
	painter.setPen(this->palette().shadow().color());
	painter.drawRect(0, 0, w, h);
}

void DisassemblyWidget::mousePressEvent(QMouseEvent* event)
{
	const u32 selectedAddress = (static_cast<int>(event->position().y()) / m_rowHeight * 4) + m_visibleStart;
	if (event->buttons() & Qt::LeftButton)
	{
		if (event->modifiers() & Qt::ShiftModifier)
		{
			if (selectedAddress < m_selectedAddressStart)
			{
				m_selectedAddressStart = selectedAddress;
			}
			else if (selectedAddress > m_visibleStart)
			{
				m_selectedAddressEnd = selectedAddress;
			}
		}
		else
		{
			m_selectedAddressStart = selectedAddress;
			m_selectedAddressEnd = selectedAddress;
		}
	}
	else if (event->buttons() & Qt::RightButton)
	{
		if (m_selectedAddressStart == m_selectedAddressEnd)
		{
			m_selectedAddressStart = selectedAddress;
			m_selectedAddressEnd = selectedAddress;
		}
	}
	this->repaint();
}

void DisassemblyWidget::mouseDoubleClickEvent(QMouseEvent* event)
{
	if (!m_cpu->isAlive())
		return;

	const u32 selectedAddress = (static_cast<int>(event->position().y()) / m_rowHeight * 4) + m_visibleStart;
	if (CBreakPoints::IsAddressBreakPoint(m_cpu->getCpuType(), selectedAddress))
	{
		Host::RunOnCPUThread([&] { CBreakPoints::RemoveBreakPoint(m_cpu->getCpuType(), selectedAddress); });
	}
	else
	{
		Host::RunOnCPUThread([&] { CBreakPoints::AddBreakPoint(m_cpu->getCpuType(), selectedAddress); });
	}
	breakpointsChanged();
	this->repaint();
}

void DisassemblyWidget::wheelEvent(QWheelEvent* event)
{
	if (event->angleDelta().y() < 0) // todo: max address bounds check?
	{
		m_visibleStart += 4;
	}
	else if (event->angleDelta().y() && m_visibleStart > 0)
	{
		m_visibleStart -= 4;
	}
	this->repaint();
}

void DisassemblyWidget::keyPressEvent(QKeyEvent* event)
{
	switch (event->key())
	{
		case Qt::Key_Up:
		{
			m_selectedAddressStart -= 4;
			if (!(event->modifiers() & Qt::ShiftModifier))
				m_selectedAddressEnd = m_selectedAddressStart;

			// Auto scroll
			if (m_visibleStart > m_selectedAddressStart)
				m_visibleStart -= 4;
		}
		break;
		case Qt::Key_PageUp:
		{
			m_selectedAddressStart -= m_visibleRows * 4;
			m_selectedAddressEnd = m_selectedAddressStart;
			m_visibleStart -= m_visibleRows * 4;
		}
		break;
		case Qt::Key_Down:
		{
			m_selectedAddressEnd += 4;
			if (!(event->modifiers() & Qt::ShiftModifier))
				m_selectedAddressStart = m_selectedAddressEnd;

			// Purposely scroll on the second to last row. It's possible to
			// size the window so part of a row is visible and we don't want to have half a row selected and cut off!
			if (m_visibleStart + ((m_visibleRows - 1) * 4) < m_selectedAddressEnd)
				m_visibleStart += 4;

			break;
		}
		case Qt::Key_PageDown:
		{
			m_selectedAddressStart += m_visibleRows * 4;
			m_selectedAddressEnd = m_selectedAddressStart;
			m_visibleStart += m_visibleRows * 4;
			break;
		}
		case Qt::Key_G:
			contextGoToAddress();
			break;
		case Qt::Key_C:
			contextCopyInstructionText();
			break;
		case Qt::Key_B:
		case Qt::Key_Space:
			contextToggleBreakpoint();
			break;
		case Qt::Key_M:
			contextAssembleInstruction();
			break;
		case Qt::Key_Right:
			contextFollowBranch();
			break;
		case Qt::Key_Left:
			gotoAddress(m_cpu->getPC());
			break;
	}

	this->repaint();
}

void DisassemblyWidget::customMenuRequested(QPoint pos)
{
	if (!m_cpu->isAlive())
		return;

	QMenu* contextMenu = new QMenu(this);

	QAction* action = 0;
	contextMenu->addAction(action = new QAction(tr("Copy Address"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextCopyAddress);
	contextMenu->addAction(action = new QAction(tr("Copy Instruction Hex"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextCopyInstructionHex);
	contextMenu->addAction(action = new QAction(tr("Copy Instruction Text"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextCopyInstructionText);
	contextMenu->addSeparator();
	if (AddressCanRestore(m_selectedAddressStart, m_selectedAddressEnd))
	{
		contextMenu->addAction(action = new QAction(tr("Restore Instruction(s)"), this));
		connect(action, &QAction::triggered, this, &DisassemblyWidget::contextRestoreInstruction);
	}
	contextMenu->addAction(action = new QAction(tr("Assemble new Instruction(s)"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextAssembleInstruction);
	contextMenu->addAction(action = new QAction(tr("NOP Instruction(s)"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextNoopInstruction);
	contextMenu->addSeparator();
	contextMenu->addAction(action = new QAction(tr("Run to Cursor"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextRunToCursor);
	contextMenu->addAction(action = new QAction(tr("Jump to Cursor"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextJumpToCursor);
	contextMenu->addAction(action = new QAction(tr("Toggle Breakpoint"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextToggleBreakpoint);
	contextMenu->addAction(action = new QAction(tr("Follow Branch"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextFollowBranch);
	contextMenu->addSeparator();
	contextMenu->addAction(action = new QAction(tr("Go to Address"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextGoToAddress);
	contextMenu->addAction(action = new QAction(tr("Go to in Memory View"), this));
	connect(action, &QAction::triggered, this, [this]() { gotoInMemory(m_selectedAddressStart); });
	contextMenu->addSeparator();
	contextMenu->addAction(action = new QAction(tr("Add Function"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextAddFunction);
	contextMenu->addAction(action = new QAction(tr("Rename Function"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextRenameFunction);
	contextMenu->addAction(action = new QAction(tr("Remove Function"), this));
	connect(action, &QAction::triggered, this, &DisassemblyWidget::contextRemoveFunction);
	if (FunctionCanRestore(m_selectedAddressStart))
	{
		contextMenu->addAction(action = new QAction(tr("Restore Function"), this));
		connect(action, &QAction::triggered, this, &DisassemblyWidget::contextRestoreFunction);
	}
	else
	{
		contextMenu->addAction(action = new QAction(tr("Stub (NOP) Function"), this));
		connect(action, &QAction::triggered, this, &DisassemblyWidget::contextStubFunction);
	}
	contextMenu->setAttribute(Qt::WA_DeleteOnClose);
	contextMenu->popup(this->mapToGlobal(pos));
}

inline QString DisassemblyWidget::DisassemblyStringFromAddress(u32 address, QFont font, u32 pc, bool selected)
{
	DisassemblyLineInfo line;

	if (!m_cpu->isValidAddress(address))
		return tr("%1 NOT VALID ADDRESS").arg(address, 8, 16, QChar('0')).toUpper();
	// Todo? support non symbol view?
	m_disassemblyManager.getLine(address, true, line);

	const bool isConditional = line.info.isConditional && m_cpu->getPC() == address;
	const bool isConditionalMet = line.info.conditionMet;
	const bool isCurrentPC = m_cpu->getPC() == address;

	const std::string addressSymbol = m_cpu->GetSymbolMap().GetLabelName(address);

	const auto demangler = demangler::CDemangler::createGcc();

	QString lineString("  %1  %2 %3  %4 %5");

	if (addressSymbol.empty()) // The address wont have symbol text if it's the start of a function for example
		lineString = lineString.arg(address, 8, 16, QChar('0')).toUpper();
	else
	{
		// We want this text elided
		QFontMetrics metric(font);
		QString symbolString;
		if (m_demangleFunctions)
		{
			symbolString = QString::fromStdString(demangler->demangleToString(addressSymbol));
			if (symbolString.isEmpty())
			{
				symbolString = QString::fromStdString(addressSymbol);
			}
		}
		else
		{
			symbolString = QString::fromStdString(addressSymbol);
		}

		lineString = lineString.arg(metric.elidedText(symbolString, Qt::ElideRight, (selected ? 32 : 8) * font.pointSize()));
	}

	lineString = lineString.leftJustified(4, ' ') // Address / symbol
					 .arg(line.name.c_str())
					 .arg(line.params.c_str()) // opcode + arguments
					 .arg(isConditional ? (isConditionalMet ? "# true" : "# false") : "")
					 .arg(isCurrentPC ? "<--" : "");

	return lineString;
}

QColor DisassemblyWidget::GetAddressFunctionColor(u32 address)
{
	// This is an attempt to figure out if the current palette is dark or light
	// We calculate the luminescence of the alternateBase colour
	// and swap between our darker and lighter function colours

	std::array<QColor, 6> colors;
	const QColor base = this->palette().alternateBase().color();

	const auto Y = (base.redF() * 0.33) + (0.5 * base.greenF()) + (0.16 * base.blueF());

	if (Y > 0.5)
	{
		colors = {
			QColor::fromRgba(0xFFFA3434),
			QColor::fromRgba(0xFF206b6b),
			QColor::fromRgba(0xFF858534),
			QColor::fromRgba(0xFF378c37),
			QColor::fromRgba(0xFF783278),
			QColor::fromRgba(0xFF21214a),
		};
	}
	else
	{
		colors = {
			QColor::fromRgba(0xFFe05555),
			QColor::fromRgba(0xFF55e0e0),
			QColor::fromRgba(0xFFe8e855),
			QColor::fromRgba(0xFF55e055),
			QColor::fromRgba(0xFFe055e0),
			QColor::fromRgba(0xFFC2C2F5),
		};
	}

	const auto funNum = m_cpu->GetSymbolMap().GetFunctionNum(address);
	if (funNum == -1)
		return this->palette().text().color();

	return colors[funNum % 6];
}

QString DisassemblyWidget::FetchSelectionInfo(SelectionInfo selInfo)
{
	QString infoBlock;
	for (u32 i = m_selectedAddressStart; i <= m_selectedAddressEnd; i += 4)
	{
		if (i != m_selectedAddressStart)
			infoBlock += '\n';
		if (selInfo == SelectionInfo::ADDRESS)
		{
			infoBlock += FilledQStringFromValue(i, 16);
		}
		else if (selInfo == SelectionInfo::INSTRUCTIONTEXT)
		{
			DisassemblyLineInfo line;
			m_disassemblyManager.getLine(i, true, line);
			infoBlock += QString("%1 %2").arg(line.name.c_str()).arg(line.params.c_str());
		}
		else // INSTRUCTIONHEX
		{
			infoBlock += FilledQStringFromValue(m_cpu->read32(i), 16);
		}
	}
	return infoBlock;
}

void DisassemblyWidget::gotoAddress(u32 address)
{
	const u32 destAddress = address & ~3;
	// Center the address
	m_visibleStart = (destAddress - (m_visibleRows * 4 / 2)) & ~3;
	m_selectedAddressStart = destAddress;
	m_selectedAddressEnd = destAddress;

	this->repaint();
	this->setFocus();
}

bool DisassemblyWidget::AddressCanRestore(u32 start, u32 end)
{
	for (u32 i = start; i <= end; i += 4)
	{
		if (this->m_nopedInstructions.find(i) != this->m_nopedInstructions.end())
		{
			return true;
		}
	}
	return false;
}

bool DisassemblyWidget::FunctionCanRestore(u32 address)
{
	u32 funcStartAddress = m_cpu->GetSymbolMap().GetFunctionStart(address);

	if (funcStartAddress != SymbolMap::INVALID_ADDRESS)
	{
		if (m_stubbedFunctions.find(funcStartAddress) != this->m_stubbedFunctions.end())
		{
			return true;
		}
	}
	else
	{
		if (m_stubbedFunctions.find(address) != this->m_stubbedFunctions.end())
		{
			return true;
		}
	}
	return false;
}
