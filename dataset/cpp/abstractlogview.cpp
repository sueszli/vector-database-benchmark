/*
 * Copyright (C) 2009, 2010, 2011, 2012, 2013, 2015 Nicolas Bonnefon
 * and other contributors
 *
 * This file is part of glogg.
 *
 * glogg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * glogg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with glogg.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Copyright (C) 2016 -- 2019 Anton Filimonov and other contributors
 *
 * This file is part of klogg.
 *
 * klogg is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * klogg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with klogg.  If not, see <http://www.gnu.org/licenses/>.
 */

// This file implements the AbstractLogView base class.
// Most of the actual drawing and event management common to the two views
// is implemented in this class.  The class only calls protected virtual
// functions when view specific behaviour is desired, using the template
// pattern.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <qcolor.h>
#include <qscreen.h>
#include <utility>
#include <vector>

#include <QApplication>
#include <QClipboard>
#include <QFileDialog>
#include <QFontMetrics>
#include <QGestureEvent>
#include <QInputDialog>
#include <QMenu>
#include <QPaintEvent>
#include <QPainter>
#include <QPalette>
#include <QProgressDialog>
#include <QRect>
#include <QScrollBar>
#include <QShortcut>
#if QT_VERSION >= QT_VERSION_CHECK( 5, 10, 0 )
#include <QStringView>
#else
#include <QStringRef>
#endif
#include <QtCore>

#include <tbb/flow_graph.h>

#include "abstractlogview.h"
#include "containers.h"
#include "linetypes.h"

#include "active_screen.h"
#include "clipboard.h"
#include "configuration.h"
#include "highlighterset.h"
#include "highlightersmenu.h"
#include "log.h"
#include "overview.h"
#include "quickfind.h"
#include "quickfindpattern.h"
#include "regularexpressionpattern.h"
#include "shortcuts.h"

#ifdef Q_OS_WIN

#pragma warning( disable : 4244 )

#include <intrin.h>

#if _WIN64
inline int countLeadingZeroes( uint64_t value )
{
    unsigned long leading_zero = 0;

    if ( _BitScanReverse64( &leading_zero, value ) ) {
        return 63ul - leading_zero;
    }
    else {
        return 64;
    }
}
#else
inline int countLeadingZeroes( uint64_t value )
{
    unsigned long leading_zero = 0;

    if ( _BitScanReverse( &leading_zero, static_cast<uint32_t>( value ) ) ) {
        return 63ul - leading_zero;
    }
    else {
        return 64;
    }
}
#endif

#else
inline int countLeadingZeroes( uint64_t value )
{
    return __builtin_clzll( value );
}
#endif

namespace {

int mapPullToFollowLength( int length );

int intLog2( uint64_t x )
{
    return 63 - countLeadingZeroes( x | 1 );
}

// see https://lemire.me/blog/2021/05/28/computing-the-number-of-digits-of-an-integer-quickly/
int countDigits( uint64_t x )
{
    int l2 = intLog2( x );
    int ans = ( ( 77 * l2 ) >> 8 );
    static uint64_t table[] = { 9,
                                99,
                                999,
                                9999,
                                99999,
                                999999,
                                9999999,
                                99999999,
                                999999999,
                                9999999999,
                                99999999999,
                                999999999999,
                                9999999999999,
                                99999999999999,
                                999999999999999,
                                9999999999999999,
                                99999999999999999,
                                999999999999999999,
                                9999999999999999999u,
                                0xFFFFFFFFFFFFFF };
    if ( x > table[ ans ] ) {
        ans += 1;
    }
    return ans + 1;
}

int textWidth( const QFontMetrics& fm, const QString& text )
{
#if ( QT_VERSION >= QT_VERSION_CHECK( 5, 11, 0 ) )
    return fm.horizontalAdvance( text );
#else
    return fm.width( text );
#endif
}

#if QT_VERSION >= QT_VERSION_CHECK( 5, 10, 0 )
int textWidth( const QFontMetrics& fm, const QStringView& text )
{
    if ( text.isEmpty() ) {
        return 0;
    }
    return textWidth( fm, QString::fromRawData( text.data(), klogg::isize( text ) ) );
}
#else
int textWidth( const QFontMetrics& fm, const QStringRef& text )
{
    if ( text.isEmpty() ) {
        return 0;
    }
    return textWidth( fm, QString::fromRawData( text.data(), klogg::isize( text ) ) );
}
#endif

std::unique_ptr<QPainter> pixmapPainter( QPaintDevice* paintDevice, const QFont& font )
{
    auto painter = std::make_unique<QPainter>( paintDevice );
    // LOG_DEBUG << "font: " << viewport()->font().family().toStdString();
    // LOG_DEBUG << "font painter: " << painter->font().family().toStdString();

    painter->setFont( font );
    painter->setRenderHints( QPainter::Antialiasing | QPainter::TextAntialiasing );

    return painter;
}

QFontMetrics pixmapFontMetrics( const QFont& font )
{
    QPixmap pm{ 1, 1 };
    auto devicePainter = pixmapPainter( &pm, font );
    return devicePainter->fontMetrics();
}

class WrappedLinesView {
  public:
#if QT_VERSION >= QT_VERSION_CHECK( 5, 10, 0 )
    using WrappedString = QStringView;
#else
    using WrappedString = QStringRef;
#endif

    explicit WrappedLinesView( const QString& longLine, LineLength visibleColumns )
    {
        if ( longLine.isEmpty() ) {
            wrappedLines_.push_back( WrappedString{} );
        }
        else {
#if QT_VERSION >= QT_VERSION_CHECK( 5, 10, 0 )
            WrappedString lineToWrap( longLine );
#else
            WrappedString lineToWrap( &longLine );
#endif
            while ( lineToWrap.size() > visibleColumns.get() ) {
                wrappedLines_.push_back( lineToWrap.left( visibleColumns.get() ) );
                lineToWrap = lineToWrap.mid( visibleColumns.get() );
            }
            if ( lineToWrap.size() > 0 ) {
                wrappedLines_.push_back( lineToWrap );
            }
        }
    }

    size_t wrappedLinesCount() const
    {
        return wrappedLines_.size();
    }

    klogg::vector<WrappedString> mid( LineColumn start, LineLength length ) const
    {
        auto getLength = []( const auto& view ) -> LineLength::UnderlyingType {
            return type_safe::narrow_cast<LineLength::UnderlyingType>( view.size() );
        };

        klogg::vector<WrappedString> resultChunks;
        if ( wrappedLines_.size() == 1 ) {
            auto& wrappedLine = wrappedLines_.front();
            auto len = std::min( length.get(), getLength( wrappedLine ) - start.get() );
            resultChunks.push_back( wrappedLine.mid( start.get(), ( len > 0 ? len : 0 ) ) );
            return resultChunks;
        }

        size_t wrappedLineIndex = 0;
        auto positionInWrappedLine = start.get();
        while ( positionInWrappedLine > getLength( wrappedLines_[ wrappedLineIndex ] ) ) {
            positionInWrappedLine -= getLength( wrappedLines_[ wrappedLineIndex ] );
            wrappedLineIndex++;
            if ( wrappedLineIndex >= wrappedLines_.size() ) {
                return resultChunks;
            }
        }

        auto chunkLength = length.get();
        while ( positionInWrappedLine + chunkLength
                > getLength( wrappedLines_[ wrappedLineIndex ] ) ) {
            resultChunks.push_back(
                wrappedLines_[ wrappedLineIndex ].mid( positionInWrappedLine ) );
            wrappedLineIndex++;
            positionInWrappedLine = 0;
            chunkLength -= getLength( resultChunks.back() );
            if ( wrappedLineIndex >= wrappedLines_.size() ) {
                return resultChunks;
            }
        }

        if ( chunkLength > 0 ) {
            auto& wrappedLine = wrappedLines_[ wrappedLineIndex ];
            auto len = std::min( chunkLength, getLength( wrappedLine ) - positionInWrappedLine );
            resultChunks.push_back(
                wrappedLine.mid( positionInWrappedLine, ( len > 0 ? len : 0 ) ) );
        }

        return resultChunks;
    }

    bool isEmpty() const
    {
        return wrappedLines_.empty() || wrappedLines_.front().isEmpty();
    }

    klogg::vector<WrappedString> wrappedLines_;
};

class LineChunk {
  public:
    LineChunk( LineColumn firstCol, LineColumn endCol, QColor foreColor, QColor backColor )
        : start_{ firstCol }
        , end_{ endCol }
        , foreColor_{ foreColor }
        , backColor_{ backColor }
    {
    }

    LineColumn start() const
    {
        return start_;
    }
    LineColumn end() const
    {
        return end_;
    }

    LineLength size() const
    {
        auto length = end_.get() - start_.get() + 1;
        return length >= 0 ? LineLength{ length } : 0_length;
    }

    QColor foreColor() const
    {
        return foreColor_;
    }

    QColor backColor() const
    {
        return backColor_;
    }

  private:
    LineColumn start_ = {};
    LineColumn end_ = {};

    QColor foreColor_;
    QColor backColor_;
};

// Utility class for syntax colouring.
// It stores the chunks of line to draw
// each chunk having a different colour
class LineDrawer {
  public:
    explicit LineDrawer( const QColor& backColor )
        : backColor_( backColor )
    {
    }

    // Add a chunk of line using the given colours.
    // Both first_col and last_col are included
    // An empty chunk will be ignored.
    // the first column will be set to 0 if negative
    // The column are relative to the screen
    void addChunk( LineColumn firstCol, LineColumn lastCol, QColor fore, QColor back )
    {
        // LOG_INFO << "addChunk " << firstCol.get() << " " << lastCol.get();
        const auto length = lastCol.get() - firstCol.get() + 1;

        if ( length > 0 ) {
            chunks_.emplace_back( firstCol, lastCol, fore, back );
        }

        // LOG_INFO << "added Chunk of  " << length;
    }

    // Draw the current line of text using the given painter,
    // in the passed block (in pixels)
    // The line must be cut to fit on the screen.
    // leftExtraBackgroundPx is the an extra margin to start drawing
    // the coloured // background, going all the way to the element
    // left of the line looks better.
    void draw( QPainter* painter, int initialXPos, int initialYPos, int lineWidth,
               const WrappedLinesView& wrappedLines, int leftExtraBackgroundPx )
    {
        QFontMetrics fm = painter->fontMetrics();
        const int fontHeight = fm.height();
        const int fontAscent = fm.ascent();

        int xPos = initialXPos;
        int yPos = initialYPos;
        // LOG_INFO << "drawing chunks " << chunks_.size();
        for ( const auto& chunk : chunks_ ) {
            // Draw each chunk
            // LOG_INFO << "draw chunk: " << chunk.start().get() << " " << chunk.size().get()
            //         << " empty w: " << wrappedLines.isEmpty();
            const auto& wrappedChunks = wrappedLines.mid( chunk.start(), chunk.size() );
            bool isFirstLine = true;
            for ( const auto& chunkText : wrappedChunks ) {
                if ( !isFirstLine ) {
                    xPos = initialXPos;
                    yPos += fontHeight;
                }
                isFirstLine = false;

                if ( chunkText.isEmpty() ) {
                    continue;
                }

                auto chunkWidth = textWidth( fm, chunkText );
                if ( xPos == initialXPos ) {
                    // First chunk, we extend the left background a bit,
                    // it looks prettier.
                    painter->fillRect( xPos - leftExtraBackgroundPx, yPos,
                                       chunkWidth + leftExtraBackgroundPx, fontHeight,
                                       chunk.backColor() );
                }
                else {
                    // other chunks...
                    painter->fillRect( xPos, yPos, chunkWidth, fontHeight, chunk.backColor() );
                }

                painter->setPen( chunk.foreColor() );
                painter->drawText(
                    xPos, yPos + fontAscent,
                    QString::fromRawData( chunkText.data(), klogg::isize( chunkText ) ) );

                xPos += chunkWidth;
            }
        }

        // Draw the empty block at the end of the line
        int blankWidth = lineWidth - xPos;

        if ( blankWidth > 0 )
            painter->fillRect( xPos, yPos, blankWidth, fontHeight, backColor_ );
    }

  private:
    klogg::vector<LineChunk> chunks_;
    QColor backColor_;
};

} // namespace

void DigitsBuffer::reset()
{
    LOG_DEBUG << "DigitsBuffer::reset()";

    timer_.stop();
    digits_.clear();
}

void DigitsBuffer::add( char character )
{
    LOG_DEBUG << "DigitsBuffer::add()";

    digits_.append( QChar( character ) );
    timer_.start( DigitsTimeout, this );
}

LineNumber::UnderlyingType DigitsBuffer::content()
{
    const auto result = digits_.toULongLong();
    reset();

    return result;
}

bool DigitsBuffer::isEmpty() const
{
    return digits_.isEmpty();
}

void DigitsBuffer::timerEvent( QTimerEvent* event )
{
    if ( event->timerId() == timer_.timerId() ) {
        reset();
    }
    else {
        QObject::timerEvent( event );
    }
}

AbstractLogView::AbstractLogView( const AbstractLogData* newLogData,
                                  const QuickFindPattern* const quickFindPattern, QWidget* parent )
    : QAbstractScrollArea( parent )
    , followElasticHook_( HookThreshold )
    , logData_( newLogData )
    , searchEnd_( newLogData->getNbLine().get() )
    , quickFindPattern_( quickFindPattern )
    , quickFind_( new QuickFind( *newLogData ) )
    , pixmapFontMetrics_( this->font() )
{
    setViewport( nullptr );

    useTextWrap_ = Configuration::get().useTextWrap();

    // Hovering
    setMouseTracking( true );

    createMenu();

    connect( quickFindPattern_, SIGNAL( patternUpdated() ), this, SLOT( handlePatternUpdated() ) );
    connect( quickFind_, SIGNAL( notify( const QFNotification& ) ), this,
             SIGNAL( notifyQuickFind( const QFNotification& ) ) );
    connect( quickFind_, SIGNAL( clearNotification() ), this,
             SIGNAL( clearQuickFindNotification() ) );

    connect( quickFind_, &QuickFind::searchDone, this, &AbstractLogView::setQuickFindResult,
             Qt::QueuedConnection );

    connect( &followElasticHook_, SIGNAL( lengthChanged() ), this, SLOT( repaint() ) );
    connect( &followElasticHook_, SIGNAL( hooked( bool ) ), this,
             SIGNAL( followModeChanged( bool ) ) );

    connect( verticalScrollBar(), &QAbstractSlider::actionTriggered, this, [ this ]( int action ) {
        if ( !followMode_ ) {
            return;
        }

        if ( action == QAbstractSlider::SliderPageStepSub
             || action == QAbstractSlider::SliderSingleStepSub ) {
            disableFollow();
        }
    } );
}

AbstractLogView::~AbstractLogView()
{
    try {
        if ( quickFind_ ) {
            quickFind_->stopSearch();
        }
    } catch ( const std::exception& e ) {
        LOG_ERROR << "Failed to stop search: " << e.what();
    }
}

//
// Received events
//

void AbstractLogView::changeEvent( QEvent* changeEvent )
{
    QAbstractScrollArea::changeEvent( changeEvent );

    // Stop the timer if the widget becomes inactive
    if ( changeEvent->type() == QEvent::ActivationChange ) {
        if ( !isActiveWindow() )
            autoScrollTimer_.stop();
    }
    viewport()->update();
}

void AbstractLogView::mousePressEvent( QMouseEvent* mouseEvent )
{
    auto line = convertCoordToLine( mouseEvent->pos().y() );

    if ( mouseEvent->button() == Qt::LeftButton ) {
        // Invalidate our cache
        textAreaCache_.invalid_ = true;

        if ( line.has_value() && mouseEvent->modifiers() & Qt::ShiftModifier ) {
            selection_.selectRangeFromPrevious( *line );
            selectionCurrentEndPos_ = convertCoordToFilePos( mouseEvent->pos() );
            Q_EMIT newSelection( *line, 1_lcount, 0_lcol, 0_length );
            update();
        }
        else if ( line.has_value() ) {
            if ( mouseEvent->pos().x() < bulletZoneWidthPx_ ) {
                // Mark a line if it is clicked in the left margin
                // (only if click and release in the same area)
                markingClickInitiated_ = true;
                markingClickLine_ = line;
            }
            else {
                // Select the line, and start a selection
                if ( *line < logData_->getNbLine() ) {
                    selection_.selectLine( *line );
                    Q_EMIT newSelection( *line, 1_lcount, 0_lcol, 0_length );
                }

                // Remember the click in case we're starting a selection
                selectionStarted_ = true;
                selectionStartPos_ = convertCoordToFilePos( mouseEvent->pos() );
                selectionCurrentEndPos_ = selectionStartPos_;
            }
        }
    }
    else if ( mouseEvent->button() == Qt::RightButton ) {
        if ( line.has_value() && line >= logData_->getNbLine() ) {
            line = {};
        }

        const auto filePos = convertCoordToFilePos( mouseEvent->pos() );

        if ( line.has_value()
             && !selection_.isPortionSelected( *line, filePos.column(), filePos.column() ) ) {
            selection_.selectLine( *line );
            Q_EMIT newSelection( *line, 1_lcount, 0_lcol, 0_length );
            textAreaCache_.invalid_ = true;
        }

        if ( selection_.isSingleLine() ) {
            copyAction_->setText( tr( "&Copy this line" ) );
            copyWithLineNumbersAction_->setText( tr( "Copy this line with line number" ) );

            setSearchStartAction_->setEnabled( true );
            setSearchEndAction_->setEnabled( true );

            setSelectionStartAction_->setEnabled( true );
            setSelectionEndAction_->setEnabled( !!selectionStart_ );
        }
        else {
            copyAction_->setText( tr( "&Copy" ) );
            copyAction_->setStatusTip( tr( "Copy the selection" ) );

            copyWithLineNumbersAction_->setText( tr( "Copy with line numbers" ) );

            setSearchStartAction_->setEnabled( false );
            setSearchEndAction_->setEnabled( false );

            setSelectionStartAction_->setEnabled( false );
            setSelectionEndAction_->setEnabled( false );
        }

        bool hasUnmarkedLines = false;
        auto lines = selection_.getLines();
        for ( auto i = 0u; i < lines.size(); ++i ) {
            using LineTypeFlags = AbstractLogData::LineTypeFlags;
            const auto currentLineType = lineType( lines[ i ] );
            if ( !currentLineType.testFlag( LineTypeFlags::Mark ) ) {
                hasUnmarkedLines = true;
                break;
            }
        }
        markAction_->setText( hasUnmarkedLines ? tr( "&Mark" ) : tr( "Unmark" ) );

        if ( selection_.isPortion() ) {
            findNextAction_->setEnabled( true );
            findPreviousAction_->setEnabled( true );
            addToSearchAction_->setEnabled( true );
            replaceSearchAction_->setEnabled( true );
        }
        else {
            findNextAction_->setEnabled( false );
            findPreviousAction_->setEnabled( false );
            addToSearchAction_->setEnabled( false );
            replaceSearchAction_->setEnabled( false );
        }

        highlightersMenu_->createHighlightersMenu();
        highlightersMenu_->populateHighlightersMenu();
        highlightersMenu_->setApplyChange( [ this ]() { Q_EMIT highlightersChange(); } );

        auto colorLablesActionGroup = new QActionGroup( this );
        connect( colorLablesActionGroup, &QActionGroup::triggered, this,
                 &AbstractLogView::setColorLabel );
        colorLabelsMenu_->clear();
        colorLabelsMenu_->setEnabled( selection_.isPortion() || selection_.isSingleLine() );
        if ( colorLabelsMenu_->isEnabled() ) {
            auto selectedText = selection_.getSelectedText( logData_ );
            std::optional<size_t> currentLabel;
            for ( auto i = 0u; i < quickHighlighters_.size(); ++i ) {
                if ( quickHighlighters_[ i ].contains( selectedText ) ) {
                    currentLabel = i;
                    break;
                }
            }

            auto noneAction = colorLabelsMenu_->addAction( tr( "None" ) );
            noneAction->setActionGroup( colorLablesActionGroup );
            noneAction->setCheckable( true );
            noneAction->setChecked( !currentLabel.has_value() );
            if ( currentLabel ) {
                noneAction->setData( static_cast<unsigned>( *currentLabel ) );
            }

            const auto& quickHighlightersConfiguration
                = HighlighterSetCollection::get().quickHighlighters();

            colorLabelsMenu_->addSeparator();
            const auto maxLabel
                = std::min( quickHighlighters_.size(),
                            static_cast<size_t>( quickHighlightersConfiguration.size() ) );
            for ( auto i = 0u; i < maxLabel; ++i ) {

                const auto& currentLabelConfiguration
                    = quickHighlightersConfiguration.at( static_cast<int>( i ) );
                auto colorLabelAction
                    = colorLabelsMenu_->addAction( currentLabelConfiguration.name );
                colorLabelAction->setActionGroup( colorLablesActionGroup );
                colorLabelAction->setCheckable( true );
                colorLabelAction->setChecked( currentLabel == i );
                colorLabelAction->setData( i );

                QPixmap pixmap( 20, 10 );
                auto fillColor = currentLabelConfiguration.color.backColor;
                fillColor.setAlphaF( 1.0 );
                pixmap.fill( fillColor );
                colorLabelAction->setIcon( QIcon( pixmap ) );
                colorLabelAction->setIconVisibleInMenu( true );
            }
            colorLabelsMenu_->addSeparator();
            auto clearAllAction = colorLabelsMenu_->addAction( tr( "Clear all" ) );
            connect( clearAllAction, &QAction::triggered, this,
                     &AbstractLogView::clearColorLabels );
        }
        // Display the popup (blocking)
        popupMenu_->exec( QCursor::pos( activeScreen( this ) ) );

        highlightersMenu_->clearHighlightersMenu();
        colorLablesActionGroup->deleteLater();
    }

    Q_EMIT activity();
}

void AbstractLogView::mouseMoveEvent( QMouseEvent* mouseEvent )
{
    // Selection implementation
    if ( selectionStarted_ ) {
        // Invalidate our cache
        textAreaCache_.invalid_ = true;

        const auto thisEndPos = convertCoordToFilePos( mouseEvent->pos() );

        if ( thisEndPos != selectionCurrentEndPos_ ) {
            const auto lineNumber = thisEndPos.line();
            // Are we on a different line?
            if ( selectionStartPos_.line() != thisEndPos.line() ) {
                if ( thisEndPos.line() != selectionCurrentEndPos_.line() ) {
                    // This is a 'range' selection
                    selection_.selectRange( selectionStartPos_.line(), lineNumber );

                    Q_EMIT newSelection(
                        lineNumber, selection_.getSelectedLinesCount(),
                        0_lcol, // portion selection always starts from the first column
                        LineLength{ getSelectedText().size() } );

                    update();
                }
            }
            // So we are on the same line. Are we moving horizontaly?
            else if ( thisEndPos.column() != selectionCurrentEndPos_.column() ) {
                // This is a 'portion' selection
                selection_.selectPortion( lineNumber, selectionStartPos_.column(),
                                          thisEndPos.column() );
                auto selectionStr = getSelectedText();
                Q_EMIT newSelection( lineNumber, 1_lcount, selectionStartPos_.column(),
                                     LineLength( selectionStr.size() ) );
                update();
            }
            // On the same line, and moving vertically then
            else {
                // This is a 'line' selection
                selection_.selectLine( lineNumber );
                Q_EMIT newSelection( lineNumber, 1_lcount, 0_lcol, 0_length );
                update();
            }
            selectionCurrentEndPos_ = thisEndPos;
        }

        // Do we need to scroll while extending the selection?
        QRect visible = viewport()->rect();
        visible.setLeft( leftMarginPx_ );
        if ( visible.contains( mouseEvent->pos() ) )
            autoScrollTimer_.stop();
        else if ( !autoScrollTimer_.isActive() )
            autoScrollTimer_.start( 100, this );
    }
    else {
        considerMouseHovering( mouseEvent->pos().x(), mouseEvent->pos().y() );
    }
}

void AbstractLogView::mouseReleaseEvent( QMouseEvent* mouseEvent )
{
    if ( markingClickInitiated_ ) {
        markingClickInitiated_ = false;
        const auto line = convertCoordToLine( mouseEvent->pos().y() );
        if ( line.has_value() && line == markingClickLine_ ) {
            // Invalidate our cache
            textAreaCache_.invalid_ = true;

            Q_EMIT markLines( { *line } );
        }
    }
    else {
        selectionStarted_ = false;
        if ( autoScrollTimer_.isActive() )
            autoScrollTimer_.stop();
        updateGlobalSelection();
    }
}

void AbstractLogView::mouseDoubleClickEvent( QMouseEvent* mouseEvent )
{
    if ( mouseEvent->button() == Qt::LeftButton ) {
        // Invalidate our cache
        textAreaCache_.invalid_ = true;

        const auto pos = convertCoordToFilePos( mouseEvent->pos() );
        selectWordAtPosition( pos );
    }

    Q_EMIT activity();
}

void AbstractLogView::timerEvent( QTimerEvent* timerEvent )
{
    if ( timerEvent->timerId() == autoScrollTimer_.timerId() ) {
        QRect visible = viewport()->rect();
        visible.setLeft( leftMarginPx_ );
        const QPoint globalPos = QCursor::pos( activeScreen( this ) );
        const QPoint pos = viewport()->mapFromGlobal( globalPos );
        QMouseEvent ev( QEvent::MouseMove, pos, globalPos, Qt::LeftButton, Qt::LeftButton,
                        Qt::NoModifier );
        mouseMoveEvent( &ev );
        int deltaX = qMax( pos.x() - visible.left(), visible.right() - pos.x() ) - visible.width();
        int deltaY = qMax( pos.y() - visible.top(), visible.bottom() - pos.y() ) - visible.height();
        int delta = qMax( deltaX, deltaY );

        if ( delta >= 0 ) {
            if ( delta < 7 )
                delta = 7;
            int timeout = 4900 / ( delta * delta );
            autoScrollTimer_.start( timeout, this );

            if ( deltaX > 0 )
                horizontalScrollBar()->triggerAction( pos.x() < visible.center().x()
                                                          ? QAbstractSlider::SliderSingleStepSub
                                                          : QAbstractSlider::SliderSingleStepAdd );

            if ( deltaY > 0 )
                verticalScrollBar()->triggerAction( pos.y() < visible.center().y()
                                                        ? QAbstractSlider::SliderSingleStepSub
                                                        : QAbstractSlider::SliderSingleStepAdd );
        }
    }
    QAbstractScrollArea::timerEvent( timerEvent );
}

void AbstractLogView::moveSelectionUp()
{
    const auto delta = qMax( LineNumber::UnderlyingType{ 1 }, digitsBuffer_.content() );
    disableFollow();
    moveSelection( LinesCount( delta ), true );
}

void AbstractLogView::moveSelectionDown()
{
    const auto delta = qMax( LineNumber::UnderlyingType{ 1 }, digitsBuffer_.content() );
    disableFollow();
    moveSelection( LinesCount( delta ), false );
}

void AbstractLogView::registerShortcut( const std::string& action, std::function<void()> func )
{
    const auto& config = Configuration::get();
    const auto& configuredShortcuts = config.shortcuts();

    ShortcutAction::registerShortcut( configuredShortcuts, shortcuts_, this, Qt::WidgetShortcut,
                                      action, func );
}

void AbstractLogView::registerShortcuts()
{
    doRegisterShortcuts();
}

void AbstractLogView::doRegisterShortcuts()
{
    LOG_INFO << "Reloading shortcuts";

    for ( auto& shortcut : shortcuts_ ) {
        shortcut.second->deleteLater();
    }

    shortcuts_.clear();

    registerShortcut( ShortcutAction::LogViewSelectionUp, [ this ]() { moveSelectionUp(); } );
    registerShortcut( ShortcutAction::LogViewSelectionDown, [ this ]() { moveSelectionDown(); } );

    registerShortcut( ShortcutAction::LogViewScrollUp, [ this ]() {
        verticalScrollBar()->triggerAction( QScrollBar::SliderPageStepSub );
    } );
    registerShortcut( ShortcutAction::LogViewScrollDown, [ this ]() {
        verticalScrollBar()->triggerAction( QScrollBar::SliderPageStepAdd );
    } );
    registerShortcut( ShortcutAction::LogViewScrollLeft, [ this ]() {
        horizontalScrollBar()->triggerAction( QScrollBar::SliderPageStepSub );
    } );
    registerShortcut( ShortcutAction::LogViewScrollRight, [ this ]() {
        horizontalScrollBar()->triggerAction( QScrollBar::SliderPageStepAdd );
    } );

    registerShortcut( ShortcutAction::LogViewJumpToTop,
                      [ this ]() { selectAndDisplayLine( 0_lnum ); } );
    registerShortcut( ShortcutAction::LogViewJumpToBottom, [ this ]() {
        const bool wasAtBottom = verticalScrollBar()->value() == verticalScrollBar()->maximum();
        if ( !wasAtBottom ) {
            selectAndDisplayLine( maxDisplayLineNumber() - 1_lcount );
            jumpToBottom();
        }
        else {
            Q_EMIT followModeChanged( true );
            followElasticHook_.hook( true );
        }
    } );

    registerShortcut( ShortcutAction::LogViewJumpToStartOfLine,
                      [ this ]() { jumpToStartOfLine(); } );
    registerShortcut( ShortcutAction::LogViewJumpToEndOfLine, [ this ]() { jumpToEndOfLine(); } );
    registerShortcut( ShortcutAction::LogViewJumpToRightOfScreen,
                      [ this ]() { jumpToRightOfScreen(); } );

    registerShortcut( ShortcutAction::LogViewQfForward, [ this ]() { Q_EMIT searchNext(); } );
    registerShortcut( ShortcutAction::LogViewQfBackward, [ this ]() { Q_EMIT searchPrevious(); } );
    registerShortcut( ShortcutAction::LogViewQfSelectedForward,
                      [ this ]() { findNextSelected(); } );
    registerShortcut( ShortcutAction::LogViewQfSelectedBackward,
                      [ this ]() { findPreviousSelected(); } );

    registerShortcut( ShortcutAction::LogViewMark, [ this ]() { markSelected(); } );

    registerShortcut( ShortcutAction::LogViewJumpToLineNumber, [ this ]() {
        const auto newLine = qMax( 0ull, digitsBuffer_.content() - 1ull );
        trySelectLine( LineNumber( newLine ) );
    } );

    registerShortcut( ShortcutAction::LogViewExitView, [ this ]() { Q_EMIT exitView(); } );

    registerShortcut( ShortcutAction::LogViewSendSelectionToScratchpad,
                      [ this ]() { Q_EMIT sendSelectionToScratchpad(); } );

    registerShortcut( ShortcutAction::LogViewReplaceScratchpadWithSelection,
                      [ this ]() { Q_EMIT replaceScratchpadWithSelection(); } );

    registerShortcut( ShortcutAction::LogViewAddToSearch, [ this ]() { addToSearch(); } );
    registerShortcut( ShortcutAction::LogViewExcludeFromSearch,
                      [ this ]() { excludeFromSearch(); } );
    registerShortcut( ShortcutAction::LogViewReplaceSearch, [ this ]() { replaceSearch(); } );

    registerShortcut( ShortcutAction::LogViewSelectLinesUp, [ this ]() {
        auto newPosition = selectionCurrentEndPos_;
        if ( newPosition.line() == 0_lnum ) {
            // Reached the begin
            return;
        }
        newPosition = FilePosition( selectionCurrentEndPos_.line() - 1_lcount,
                                    selectionCurrentEndPos_.column() );
        selectAndDisplayRange( newPosition );
    } );

    registerShortcut( ShortcutAction::LogViewSelectLinesDown, [ this ]() {
        auto newPosition = selectionCurrentEndPos_;
        if ( newPosition.line() >= maxDisplayLineNumber() - 1_lcount ) {
            // Reached the end
            return;
        }
        newPosition = FilePosition( selectionCurrentEndPos_.line() + 1_lcount,
                                    selectionCurrentEndPos_.column() );
        selectAndDisplayRange( newPosition );
    } );
}

void AbstractLogView::keyPressEvent( QKeyEvent* keyEvent )
{
    LOG_DEBUG << "keyPressEvent received " << keyEvent->text();

    const auto text = keyEvent->text();

    if ( keyEvent->modifiers() == Qt::NoModifier && text.size() == 1 ) {
        const auto character = text.at( 0 ).toLatin1();
        if ( ( ( character > '0' ) && ( character <= '9' ) )
             || ( !digitsBuffer_.isEmpty() && character == '0' ) ) {
            // Adds the digit to the timed buffer
            digitsBuffer_.add( character );
            keyEvent->accept();
        }
        else if ( digitsBuffer_.isEmpty() && character == '0' ) {
            jumpToStartOfLine();
            keyEvent->accept();
        }
    }
    else {
        keyEvent->ignore();
    }

    if ( keyEvent->isAccepted() ) {
        Q_EMIT activity();
    }
    else {
        // Only pass bare keys to the superclass this is so that
        // shortcuts such as Ctrl+Alt+Arrow are handled by the parent.
        if ( keyEvent->modifiers() == Qt::NoModifier
             || keyEvent->modifiers() == Qt::KeypadModifier ) {
            QAbstractScrollArea::keyPressEvent( keyEvent );
        }
    }
}

void AbstractLogView::wheelEvent( QWheelEvent* wheelEvent )
{
    Q_EMIT activity();

    int yDelta = 0;
    const auto pixelDelta = wheelEvent->pixelDelta();

    if ( pixelDelta.isNull() ) {
        yDelta = static_cast<int>(
            std::floor( static_cast<float>( wheelEvent->angleDelta().y() ) / 0.7f ) );
    }
    else {
        yDelta = pixelDelta.y();
    }

    if ( yDelta == 0 ) {
        QAbstractScrollArea::wheelEvent( wheelEvent );
        return;
    }

    if ( wheelEvent->modifiers().testFlag( Qt::ControlModifier ) ) {
        Q_EMIT changeFontSize( yDelta > 0 );
        return;
    }

    // LOG_DEBUG << "wheelEvent";

    // This is to handle the case where follow mode is on, but the user
    // has moved using the scroll bar. We take them back to the bottom.
    if ( followMode_ )
        jumpToBottom();

    const auto allowFollowOnScroll = Configuration::get().allowFollowOnScroll();
    if ( verticalScrollBar()->value() == verticalScrollBar()->maximum() ) {
        if ( allowFollowOnScroll || yDelta > 0 ) {
            // First see if we need to block the elastic (on Mac)
            if ( wheelEvent->phase() == Qt::ScrollBegin ) {
                followElasticHook_.hold();
            }
            else if ( wheelEvent->phase() == Qt::ScrollEnd
#if QT_VERSION >= QT_VERSION_CHECK( 5, 12, 0 )
                      || wheelEvent->phase() == Qt::ScrollMomentum
#endif
            ) {
                followElasticHook_.release();
            }

            followElasticHook_.move( -yDelta );
        }

        // LOG_DEBUG << "Elastic " << y_delta;
    }

    // LOG_DEBUG << "Length = " << followElasticHook_.size();
    if ( !allowFollowOnScroll
         || ( followElasticHook_.size() == 0 && !followElasticHook_.isHooked() ) ) {
        QAbstractScrollArea::wheelEvent( wheelEvent );
    }
}

void AbstractLogView::resizeEvent( QResizeEvent* )
{
    if ( logData_ == nullptr )
        return;

    LOG_DEBUG << "resizeEvent received";

    updateDisplaySize();
}

bool AbstractLogView::event( QEvent* e )
{
    LOG_DEBUG << "Event! Type: " << e->type();

    // Make sure we ignore the gesture events as
    // they seem to be accepted by default.
    if ( e->type() == QEvent::Gesture ) {
        const auto gestureEvent = static_cast<QGestureEvent*>( e );
        if ( gestureEvent ) {
            const auto gestures = gestureEvent->gestures();
            for ( QGesture* gesture : gestures ) {
                LOG_DEBUG << "Gesture: " << gesture->gestureType();
                gestureEvent->ignore( gesture );
            }

            // Ensure the event is sent up to parents who might care
            return false;
        }
    }

    return QAbstractScrollArea::event( e );
}

int AbstractLogView::lineNumberToVerticalScroll( LineNumber line ) const
{
    return static_cast<int>(
        std::round( static_cast<double>( line.get() ) * verticalScrollMultiplicator() ) );
}

LineNumber AbstractLogView::verticalScrollToLineNumber( int scrollPosition ) const
{
    return LineNumber( static_cast<LineNumber::UnderlyingType>(
        std::round( static_cast<double>( scrollPosition ) / verticalScrollMultiplicator() ) ) );
}

double AbstractLogView::verticalScrollMultiplicator() const
{
    return verticalScrollBar()->maximum() < std::numeric_limits<int>::max()
               ? 1.0
               : static_cast<double>( std::numeric_limits<int>::max() )
                     / static_cast<double>( logData_->getNbLine().get() );
}

void AbstractLogView::scrollContentsBy( int dx, int dy )
{
    LOG_DEBUG << "scrollContentsBy received " << dy << "position " << verticalScrollBar()->value();

    const auto lastTopLine = ( logData_->getNbLine() - getNbVisibleLines() );

    const auto scrollPosition = verticalScrollToLineNumber( verticalScrollBar()->value() );

    if ( ( lastTopLine.get() > 0 ) && scrollPosition.get() > lastTopLine.get() ) {
        // The user is going further than the last line, we need to lock the last line at the bottom
        LOG_DEBUG << "scrollContentsBy beyond!";
        firstLine_ = scrollPosition;
        lastLineAligned_ = true;
    }
    else {
        firstLine_ = scrollPosition;
        lastLineAligned_ = false;
    }

    firstCol_ = ( firstCol_.get() - dx ) >= 0 ? LineColumn{ firstCol_.get() - dx } : 0_lcol;

    // Update the overview if we have one
    if ( overview_ != nullptr ) {
        const auto lastLine = firstLine_ + getNbVisibleLines();
        overview_->updateCurrentPosition( firstLine_, lastLine );
    }

    // Are we hovering over a new line?
    const auto mousePos = mapFromGlobal( QCursor::pos( activeScreen( this ) ) );
    considerMouseHovering( mousePos.x(), mousePos.y() );

    // Redraw
    update();
}

void AbstractLogView::paintEvent( QPaintEvent* paintEvent )
{
    const QRect invalidRect = paintEvent->rect();
    if ( ( invalidRect.isEmpty() ) || ( logData_ == nullptr ) )
        return;

    LOG_DEBUG << "paintEvent received, firstLine_=" << firstLine_
              << " lastLineAligned_=" << lastLineAligned_ << " rect: " << invalidRect.topLeft().x()
              << ", " << invalidRect.topLeft().y() << ", " << invalidRect.bottomRight().x() << ", "
              << invalidRect.bottomRight().y();

#ifdef GLOGG_PERF_MEASURE_FPS
    static uint32_t maxline = logData_->getNbLine();
    if ( !perfCounter_.addEvent() && logData_->getNbLine() > maxline ) {
        LOG_WARNING << "Redraw per second: " << perfCounter_.readAndReset()
                    << " lines: " << logData_->getNbLine();
        perfCounter_.addEvent();
        maxline = logData_->getNbLine();
    }
#endif

    auto start = std::chrono::system_clock::now();

    // Can we use our cache?
    auto deltaY = textAreaCache_.first_line_.get() - firstLine_.get();

    if ( textAreaCache_.invalid_ || ( textAreaCache_.first_column_ != firstCol_ ) ) {
        // Force a full redraw
        deltaY = std::numeric_limits<decltype( deltaY )>::max();
    }

    if ( deltaY != 0 ) {
        // Full or partial redraw
        drawTextArea( &textAreaCache_.pixmap_ );

        textAreaCache_.invalid_ = false;
        textAreaCache_.first_line_ = firstLine_;
        textAreaCache_.first_column_ = firstCol_;

        LOG_DEBUG << "End of writing "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now() - start )
                         .count();
    }
    else {
        // Use the cache as is: nothing to do!
    }

    // Height including the potentially invisible last line
    const auto wholeHeight = static_cast<int>( getNbVisibleLines().get() ) * charHeight_;
    // Height in pixels of the "pull to follow" bottom bar.
    const auto pullToFollowHeight
        = mapPullToFollowLength( followElasticHook_.size() )
          + ( followElasticHook_.isHooked()
                  ? ( wholeHeight - viewport()->height() ) + PullToFollowHookedHeight
                  : 0 );

    if ( pullToFollowHeight && ( pullToFollowCache_.nb_columns_ != getNbVisibleCols() ) ) {
        LOG_DEBUG << "Drawing pull to follow bar";
        pullToFollowCache_.pixmap_
            = drawPullToFollowBar( viewport()->width(), viewport()->devicePixelRatio() );
        pullToFollowCache_.nb_columns_ = getNbVisibleCols();
    }

    QPainter devicePainter( viewport() );
    int drawingTopPosition = -pullToFollowHeight;
    int drawingPullToFollowTopPosition = drawingTopPosition + wholeHeight;

    // This is to cover the special case where there is less than a screenful
    // worth of data, we want to see the document from the top, rather than
    // pushing the first couple of lines above the viewport.
    if ( followElasticHook_.isHooked()
         && ( logData_->getNbLine() + LinesCount( 1 ) < getNbVisibleLines() ) ) {
        drawingTopOffset_ = 0;
        drawingTopPosition += ( wholeHeight - viewport()->height() ) + PullToFollowHookedHeight;
        drawingPullToFollowTopPosition
            = drawingTopPosition + viewport()->height() - PullToFollowHookedHeight;
    }
    // This is the case where the user is on the 'extra' slot at the end
    // and is aligned on the last line (but no elastic shown)
    else if ( lastLineAligned_ && !followElasticHook_.isHooked() ) {
        drawingTopOffset_ = -( wholeHeight - viewport()->height() );
        drawingTopPosition += drawingTopOffset_;
        drawingPullToFollowTopPosition = drawingTopPosition + wholeHeight;
    }
    else {
        drawingTopOffset_ = -pullToFollowHeight;
    }

    devicePainter.drawPixmap( 0, drawingTopPosition, textAreaCache_.pixmap_ );

    // Draw the "pull to follow" zone if needed
    if ( pullToFollowHeight ) {
        devicePainter.drawPixmap( 0, drawingPullToFollowTopPosition, pullToFollowCache_.pixmap_ );
    }

    LOG_DEBUG << "End of repaint "
              << std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::system_clock::now() - start )
                     .count();
}

// These two functions are virtual and this implementation is clearly
// only valid for a non-filtered display.
// We count on the 'filtered' derived classes to override them.
LineNumber AbstractLogView::displayLineNumber( LineNumber lineNumber ) const
{
    return lineNumber + 1_lcount; // show a 1-based index
}

LineNumber AbstractLogView::lineIndex( LineNumber lineNumber ) const
{
    return lineNumber;
}

LineNumber AbstractLogView::maxDisplayLineNumber() const
{
    return LineNumber( logData_->getNbLine().get() );
}

void AbstractLogView::setOverview( Overview* overview, OverviewWidget* overviewWidget )
{
    overview_ = overview;
    overviewWidget_ = overviewWidget;

    if ( overviewWidget_ ) {
        connect( overviewWidget_, &OverviewWidget::lineClicked, this,
                 &AbstractLogView::jumpToLine );
    }
    refreshOverview();
}

LineNumber AbstractLogView::getViewPosition() const
{
    LineNumber line;

    const auto selectedLine = selection_.selectedLine();
    if ( selectedLine.has_value() ) {
        line = *selectedLine;
    }
    else {
        // Middle of the view
        line = firstLine_ + LinesCount( getNbVisibleLines().get() / 2 );
    }

    return line;
}

void AbstractLogView::searchUsingFunction( QuickFindSearchFn searchFunction )
{
    disableFollow();
    ( quickFind_->*searchFunction )( selection_, quickFindPattern_->getMatcher() );
}

void AbstractLogView::setQuickFindResult( bool hasMatch, const Portion& portion )
{
    if ( portion.isValid() ) {
        LOG_DEBUG << "search " << portion.line();
        displayLine( portion.line() );
        selection_.selectPortion( portion );
        Q_EMIT newSelection( portion.line(), 1_lcount, 0_lcol, 0_length );
    }
    else if ( !hasMatch ) {
        selection_.clear();
    }
}

void AbstractLogView::searchForward()
{
    searchUsingFunction( &QuickFind::searchForward );
}

void AbstractLogView::searchBackward()
{
    searchUsingFunction( &QuickFind::searchBackward );
}

void AbstractLogView::incrementallySearchForward()
{
    searchUsingFunction( &QuickFind::incrementallySearchForward );
}

void AbstractLogView::incrementallySearchBackward()
{
    searchUsingFunction( &QuickFind::incrementallySearchBackward );
}

void AbstractLogView::incrementalSearchAbort()
{
    selection_ = quickFind_->incrementalSearchAbort();
    Q_EMIT changeQuickFind( "", QuickFindMux::Forward );
}

void AbstractLogView::incrementalSearchStop()
{
    auto oldSelection = quickFind_->incrementalSearchStop();
    if ( selection_.isEmpty() ) {
        selection_ = oldSelection;
    }
}

void AbstractLogView::allowFollowMode( bool allow )
{
    followElasticHook_.allowHook( allow );
}

void AbstractLogView::setSearchPattern( const RegularExpressionPattern& pattern )
{
    searchPattern_ = pattern;
    forceRefresh();
}

void AbstractLogView::setQuickHighlighters(
    const std::vector<QuickHighlighters>& quickHighlighters )
{
    quickHighlighters_ = quickHighlighters;
    forceRefresh();
}

void AbstractLogView::followSet( bool checked )
{
    followMode_ = checked;
    followElasticHook_.hook( checked );
    forceRefresh();

    if ( checked )
        jumpToBottom();
}

void AbstractLogView::textWrapSet( bool checked )
{
    useTextWrap_ = checked;
    updateScrollBars();
    forceRefresh();
}

void AbstractLogView::refreshOverview()
{
    assert( overviewWidget_ );

    // Create space for the Overview if needed
    if ( ( getOverview() != nullptr ) && getOverview()->isVisible() ) {
        setViewportMargins( 0, 0, OverviewWidth, 0 );
        overviewWidget_->show();
    }
    else {
        setViewportMargins( 0, 0, 0, 0 );
        overviewWidget_->hide();
    }
}

// Reset the QuickFind when the pattern is changed.
void AbstractLogView::handlePatternUpdated()
{
    LOG_DEBUG << "AbstractLogView::handlePatternUpdated()";

    quickFind_->resetLimits();
    forceRefresh();
}

// OR the current selection with the current search expression
void AbstractLogView::addToSearch()
{
    if ( selection_.isPortion() ) {
        LOG_DEBUG << "AbstractLogView::addToSearch()";
        Q_EMIT addToSearch( selection_.getSelectedText( logData_ ) );
    }
    else {
        LOG_ERROR << "AbstractLogView::addToSearch called for a wrong type of selection";
    }
}

// Replace the current search expression with the current selection
void AbstractLogView::replaceSearch()
{
    if ( selection_.isPortion() ) {
        LOG_DEBUG << "AbstractLogView::replaceSearch()";
        Q_EMIT replaceSearch( selection_.getSelectedText( logData_ ) );
    }
    else {
        LOG_ERROR << "AbstractLogView::replaceSearch called for a wrong type of selection";
    }
}

void AbstractLogView::excludeFromSearch()
{
    if ( selection_.isPortion() ) {
        LOG_DEBUG << "AbstractLogView::excludeFromSearch()";
        Q_EMIT excludeFromSearch( selection_.getSelectedText( logData_ ) );
    }
    else {
        LOG_ERROR << "AbstractLogView::excludeFromSearch called for a wrong type of selection";
    }
}

// Find next occurrence of the selected text (*)
void AbstractLogView::findNextSelected()
{
    // Use the selected 'word' and search forward
    if ( selection_.isPortion() ) {
        Q_EMIT changeQuickFind( selection_.getSelectedText( logData_ ), QuickFindMux::Forward );
        Q_EMIT searchNext();
    }
}

// Find next previous of the selected text (#)
void AbstractLogView::findPreviousSelected()
{
    if ( selection_.isPortion() ) {
        Q_EMIT changeQuickFind( selection_.getSelectedText( logData_ ), QuickFindMux::Backward );
        Q_EMIT searchNext();
    }
}

// Copy the selection to the clipboard
void AbstractLogView::copy()
{

    try {
        auto text = selection_.getSelectedText( logData_ );
        text.replace( QChar::Null, QChar::Space );
        sendTextToClipboard( text );
    } catch ( std::exception& err ) {
        LOG_ERROR << "failed to copy data to clipboard " << err.what();
    }
}

// Copy the selection with line numbers to the clipboard
void AbstractLogView::copyWithLineNumbers()
{
    try {
        auto text = selection_.getSelectedText( logData_, true );
        text.replace( QChar::Null, QChar::Space );
        sendTextToClipboard( text );
    } catch ( std::exception& err ) {
        LOG_ERROR << "failed to copy data to clipboard " << err.what();
    }
}

void AbstractLogView::markSelected()
{
    auto lines = selection_.getLines();
    if ( !lines.empty() ) {
        Q_EMIT markLines( lines );
    }
}

void AbstractLogView::saveToFile()
{
    auto start = 0_lnum;
    const auto totalLines = logData_->getNbLine();
    auto end = LineNumber{ totalLines.get() };

    saveLinesToFile( start, end );
}

void AbstractLogView::saveSelectedToFile()
{
    const auto selectedLines = selection_.getLines();
    if ( selectedLines.empty() ) {
        return;
    }

    const auto start = selectedLines.front();
    const auto lastLine = selectedLines.back();
    const auto end = lastLine + 1_lcount;
    saveLinesToFile( start, end );
}

void AbstractLogView::saveLinesToFile( LineNumber begin, LineNumber end )
{
    auto filename = QFileDialog::getSaveFileName( this, "Save content" );
    if ( filename.isEmpty() ) {
        return;
    }

    QSaveFile saveFile{ filename };
    saveFile.open( QIODevice::WriteOnly | QIODevice::Truncate );
    if ( !saveFile.isOpen() ) {
        LOG_ERROR << "Failed to open file to save";
        return;
    }

    QProgressDialog progressDialog( this );
    progressDialog.setLabelText( tr( "Saving content to %1" ).arg( filename ) );
    klogg::vector<std::pair<LineNumber, LinesCount>> offsets;
    auto lineOffset = begin;
    const auto chunkSize = 5000_lcount;
    offsets.reserve( ( end - ( lineOffset + chunkSize ) ).get() );

    for ( ; lineOffset + chunkSize < end; lineOffset += LinesCount( chunkSize.get() ) ) {
        offsets.emplace_back( lineOffset, chunkSize );
    }
    offsets.emplace_back( lineOffset, LinesCount( ( end - lineOffset ).get() % chunkSize.get() ) );

    const QTextCodec* codec = logData_->getDisplayEncoding();
    if ( !codec ) {
        codec = QTextCodec::codecForName( "utf-8" );
    }

    AtomicFlag interruptRequest;

    progressDialog.setRange( 0, 1000 );
    connect( &progressDialog, &QProgressDialog::canceled,
             [ &interruptRequest ]() { interruptRequest.set(); } );

    tbb::flow::graph saveFileGraph;
    using LinesData = std::pair<klogg::vector<QString>, bool>;
    auto lineReader = tbb::flow::input_node<LinesData>(
        saveFileGraph,
        [ this, &offsets, &interruptRequest, &progressDialog, offsetIndex = 0u,
          finalLine = false ]( tbb::flow_control& fc ) mutable -> LinesData {
            if ( !interruptRequest && offsetIndex < offsets.size() ) {
                const auto& offset = offsets.at( offsetIndex );
                LinesData lines{ logData_->getLines( offset.first, offset.second ), true };
                for ( auto& l : lines.first ) {
#if !defined( Q_OS_WIN )
                    l.append( QChar::CarriageReturn );
#endif
                    l.append( QChar::LineFeed );
                }

                offsetIndex++;
                progressDialog.setValue( static_cast<int>(
                    std::floor( static_cast<float>( offsetIndex )
                                / static_cast<float>( offsets.size() + 1 ) * 1000.f ) ) );
                return lines;
            }
            else if ( !finalLine ) {
                finalLine = true;
            }
            else {
                fc.stop();
            }

            return {};
        } );

    auto lineWriter = tbb::flow::function_node<LinesData, tbb::flow::continue_msg>(
        saveFileGraph, 1,
        [ &interruptRequest, &codec, &saveFile,
          &progressDialog ]( const LinesData& lines ) mutable {
            if ( !lines.second ) {
                if ( !interruptRequest ) {
                    saveFile.commit();
                }

                progressDialog.finished( 0 );
                return tbb::flow::continue_msg{};
            }

            for ( const auto& l : lines.first ) {

                const auto encodedLine = codec->fromUnicode( l );
                const auto written = saveFile.write( encodedLine );

                if ( written != encodedLine.size() ) {
                    LOG_ERROR << "Saving file write failed";
                    interruptRequest.set();
                    return tbb::flow::continue_msg{};
                }
            }
            return tbb::flow::continue_msg{};
        } );

    tbb::flow::make_edge( lineReader, lineWriter );

    progressDialog.setWindowModality( Qt::ApplicationModal );
    progressDialog.open();

    lineReader.activate();
    saveFileGraph.wait_for_all();
}

void AbstractLogView::updateSearchLimits()
{
    forceRefresh();

    Q_EMIT changeSearchLimits( searchStart_, searchEnd_ );
}

void AbstractLogView::setSearchStart()
{
    const auto selectedLine = selection_.selectedLine();
    searchStart_
        = selectedLine.has_value() ? displayLineNumber( *selectedLine ) - 1_lcount : 0_lnum;
    updateSearchLimits();
}

void AbstractLogView::setSearchEnd()
{
    const auto selectedLine = selection_.selectedLine();
    searchEnd_ = selectedLine.has_value() ? displayLineNumber( *selectedLine )
                                          : LineNumber( logData_->getNbLine().get() );
    updateSearchLimits();
}

void AbstractLogView::setSelectionStart()
{
    selectionStart_ = selection_.selectedLine();
}

void AbstractLogView::setSelectionEnd()
{
    const auto selectionEnd = selection_.selectedLine();

    if ( selectionStart_ && selectionEnd ) {
        selection_.selectRange( *selectionStart_, *selectionEnd );
        selectionStart_ = {};

        forceRefresh();
    }
}

//
// Public functions
//

void AbstractLogView::updateData()
{
    LOG_DEBUG << "AbstractLogView::updateData";

    const auto lastLineNumber = LineNumber( logData_->getNbLine().get() );

    // Check the top Line is within range
    if ( firstLine_ >= lastLineNumber ) {
        firstLine_ = 0_lnum;
        firstCol_ = 0_lcol;
        verticalScrollBar()->setValue( 0 );
        horizontalScrollBar()->setValue( 0 );
    }

    // Crop selection if it become out of range
    selection_.crop( lastLineNumber - 1_lcount );

    // Adapt the scroll bars to the new content
    updateScrollBars();

    // Reset the QuickFind in case we have new stuff to search into
    quickFind_->resetLimits();

    if ( followMode_ )
        jumpToBottom();

    // Update the overview if we have one
    if ( overview_ != nullptr ) {
        // Calculate the index of the last line shown
        const LineNumber lastLine = qMin( lastLineNumber, firstLine_ + getNbVisibleLines() );
        overview_->updateCurrentPosition( firstLine_, lastLine );
    }

    forceRefresh();
}

void AbstractLogView::updateFont( const QFont& font )
{
    setFont( font );
    pixmapFontMetrics_ = pixmapFontMetrics( font );
    updateDisplaySize();
    update();
}

void AbstractLogView::updateDisplaySize()
{
    // Font is assumed to be mono-space (is restricted by options dialog)
    charHeight_ = std::max( pixmapFontMetrics_.height(), 1 );
    charWidth_ = textWidth( pixmapFontMetrics_, QString( "m" ) );

    // Update the scroll bars
    updateScrollBars();
    verticalScrollBar()->setPageStep( static_cast<int>( getNbVisibleLines().get() ) );

    if ( followMode_ )
        jumpToBottom();

    LOG_DEBUG << "viewport.width()=" << viewport()->width();
    LOG_DEBUG << "viewport.height()=" << viewport()->height();
    LOG_DEBUG << "width()=" << width();
    LOG_DEBUG << "height()=" << height();

    if ( overviewWidget_ )
        overviewWidget_->setGeometry( viewport()->width() + 2, 1, OverviewWidth - 1,
                                      viewport()->height() );

    // Our text area cache is now invalid
    textAreaCache_.invalid_ = true;
    textAreaCache_.pixmap_ = QPixmap{
        static_cast<int>( std::ceil( viewport()->width() * viewport()->devicePixelRatio() ) ),
        static_cast<int>( std::ceil( static_cast<int>( getNbVisibleLines().get() ) * charHeight_
                                     * viewport()->devicePixelRatio() ) )
    };
    textAreaCache_.pixmap_.setDevicePixelRatio( viewport()->devicePixelRatio() );
}

LineNumber AbstractLogView::getTopLine() const
{
    return firstLine_;
}

QString AbstractLogView::getSelectedText() const
{
    return selection_.getSelectedText( logData_ );
}

bool AbstractLogView::isPartialSelection() const
{
    return selection_.isPortion();
}

void AbstractLogView::selectAll()
{
    selection_.selectRange( 0_lnum, LineNumber( logData_->getNbLine().get() ) - 1_lcount );
    forceRefresh();
}

void AbstractLogView::trySelectLine( LineNumber lineToSelect )
{
    if ( lineToSelect >= logData_->getNbLine() ) {
        lineToSelect = lineToSelect - 1_lcount;
    }

    selectAndDisplayLine( lineToSelect );
}

void AbstractLogView::selectAndDisplayLine( LineNumber line )
{
    disableFollow();
    selection_.selectLine( line );
    selectionStartPos_ = FilePosition{ line, 0_lcol };
    selectionCurrentEndPos_ = selectionStartPos_;
    displayLine( line );
    Q_EMIT newSelection( line, 1_lcount, 0_lcol, 0_length );
}

void AbstractLogView::selectPortionAndDisplayLine( LineNumber line, LinesCount nLines,
                                                   LineColumn startCol, LineLength nSymbols )
{
    disableFollow();
    selection_.selectLine( line );
    selectionStartPos_ = FilePosition{ line, startCol };
    selectionCurrentEndPos_ = FilePosition{ line, startCol + nSymbols };
    displayLine( line );
    Q_EMIT newSelection( line, nLines, startCol, nSymbols );
}

// The difference between this function and displayLine() is quite
// subtle: this one always jump, even if the line passed is visible.
void AbstractLogView::jumpToLine( LineNumber line )
{
    // Put the selected line in the middle if possible
    const auto newTopLine = line - LinesCount( getNbVisibleLines().get() / 2 );
    // This will also trigger a scrollContents event
    verticalScrollBar()->setValue( lineNumberToVerticalScroll( newTopLine ) );
}

void AbstractLogView::setLineNumbersVisible( bool lineNumbersVisible )
{
    lineNumbersVisible_ = lineNumbersVisible;
}

void AbstractLogView::forceRefresh()
{
    // Invalidate our cache
    textAreaCache_.invalid_ = true;
    update();
}

void AbstractLogView::setSearchLimits( LineNumber startLine, LineNumber endLine )
{
    searchStart_ = startLine;
    searchEnd_ = endLine;

    forceRefresh();
}

//
// Private functions
//

// Returns the number of lines visible in the viewport
LinesCount AbstractLogView::getNbVisibleLines() const
{
    return LinesCount(
        static_cast<LinesCount::UnderlyingType>( viewport()->height() / charHeight_ + 1 ) );
}

// Returns the number of columns visible in the viewport
LineLength AbstractLogView::getNbVisibleCols() const
{
    const auto scrollBarWidth = verticalScrollBar()->isVisible() ? verticalScrollBar()->width() : 0;
    return LineLength{ ( viewport()->width() - leftMarginPx_ - scrollBarWidth ) / charWidth_ + 1 };
}

// Converts the mouse x, y coordinates to the line number in the file
OptionalLineNumber AbstractLogView::convertCoordToLine( int yPos ) const
{
    const auto offset = std::abs( ( yPos - drawingTopOffset_ ) / charHeight_ );
    const auto wrappedLineInfoIndex = static_cast<size_t>( std::floor( offset ) );
    if ( wrappedLineInfoIndex < wrappedLinesNumbers_.size() && !wrappedLinesNumbers_.empty() ) {
        return wrappedLinesNumbers_[ wrappedLineInfoIndex ].first;
    }
    else {
        return OptionalLineNumber{};
    }
}

// Converts the mouse x, y coordinates to the char coordinates (in the file)
// This function ensure the pos exists in the file.
FilePosition AbstractLogView::convertCoordToFilePos( const QPoint& pos ) const
{
    const auto offset = std::abs( ( pos.y() - drawingTopOffset_ ) / charHeight_ );

    if ( wrappedLinesNumbers_.empty() ) {
        return FilePosition{ 0_lnum, 0_lcol };
    }

    const auto wrappedLineInfoIndex
        = wrappedLinesNumbers_.size() > 1
              ? std::clamp( static_cast<size_t>( std::floor( offset ) ), size_t{ 0 },
                            wrappedLinesNumbers_.size() - 1 )
              : 0;

    auto [ line, wrappedLine ] = wrappedLinesNumbers_[ wrappedLineInfoIndex ];
    if ( line >= logData_->getNbLine() )
        line = LineNumber( logData_->getNbLine().get() ) - 1_lcount;

    const auto lineText = logData_->getExpandedLineString( line );
    if ( lineText.size() <= 1 ) {
        return FilePosition{ line, 0_lcol };
    }

    WrappedLinesView::WrappedString visibleText;
    if ( useTextWrap_ ) {
        WrappedLinesView wrapped{ lineText, getNbVisibleCols() };
        visibleText = wrapped.wrappedLines_[ wrappedLine ];
    }
    else {
#if QT_VERSION >= QT_VERSION_CHECK( 5, 10, 0 )
        visibleText = QStringView( lineText ).mid( firstCol_.get(), getNbVisibleCols().get() );
#else
        visibleText = QStringRef( &lineText ).mid( firstCol_.get(), getNbVisibleCols().get() );
#endif
    }

    klogg::vector<LineColumn> possibleColumns( static_cast<size_t>( visibleText.size() ) );
    std::iota( possibleColumns.begin(), possibleColumns.end(), 0_lcol );

    const auto columnIt = std::lower_bound(
        possibleColumns.cbegin(), possibleColumns.cend(), pos,
        [ this, &visibleText ]( LineColumn c, const QPoint& p ) {
            const auto width
                = textWidth( pixmapFontMetrics_, visibleText.left( c.get() ) ) + leftMarginPx_;

            return width < p.x();
        } );

    const auto length
        = LineColumn{ type_safe::narrow_cast<LineColumn::UnderlyingType>( visibleText.size() ) };

    auto column = ( columnIt != possibleColumns.end() ? *columnIt : length ) - 1_length;
    if ( useTextWrap_ ) {
        column += LineLength{ getNbVisibleCols().get() * static_cast<int>( wrappedLine ) };
    }
    else {
        column += LineLength{ firstCol_.get() };
    }

    column = std::clamp( column, 0_lcol, LineColumn( lineText.size() ) - 1_length );

    LOG_DEBUG << "AbstractLogView::convertCoordToFilePos col=" << column << " line=" << line;
    return FilePosition{ line, column };
}

// Makes the widget adjust itself to display the passed line.
// Doing so, it will throw itself a scrollContents event.
void AbstractLogView::displayLine( LineNumber line )
{
    // If the line is already the screen
    if ( ( line >= firstLine_ ) && ( line < ( firstLine_ + getNbVisibleLines() ) ) ) {
        // Invalidate our cache
        forceRefresh();
    }
    else {
        jumpToLine( line );
    }

    const auto portion = selection_.getPortionForLine( line );
    if ( portion.isValid() ) {
        horizontalScrollBar()->setValue( type_safe::narrow_cast<int>(
            portion.endColumn().get() - getNbVisibleCols().get() + 1 ) );
    }
}

// Move the selection up and down by the passed number of lines
void AbstractLogView::moveSelection( LinesCount delta, bool isDeltaNegative )
{
    LOG_DEBUG << "AbstractLogView::moveSelection delta=" << delta;

    auto selection = selection_.getLines();
    LineNumber newLine;

    if ( !selection.empty() ) {
        if ( isDeltaNegative )
            newLine = selection.front() - delta;
        else
            newLine = selection.back() + delta;
    }

    if ( newLine >= logData_->getNbLine() ) {
        newLine = LineNumber( logData_->getNbLine().get() ) - 1_lcount;
    }

    // Select and display the new line
    selection_.selectLine( newLine );
    displayLine( newLine );
    selectionStartPos_ = FilePosition{ newLine, 0_lcol };
    selectionCurrentEndPos_ = selectionStartPos_;
    Q_EMIT newSelection( newLine, selection_.getSelectedLinesCount(), 0_lcol,
                         LineLength{ getSelectedText().size() } );
}

// Make the start of the lines visible
void AbstractLogView::jumpToStartOfLine()
{
    horizontalScrollBar()->setValue( 0 );
}

LineLength AbstractLogView::maxLineLength( const klogg::vector<LineNumber>& lines ) const
{
    const auto longestLine = std::max_element(
        lines.cbegin(), lines.cend(), [ this ]( const auto& lhs, const auto& rhs ) {
            const auto lhsLength = logData_->getLineLength( lhs );
            const auto rhsLength = logData_->getLineLength( rhs );
            return lhsLength < rhsLength;
        } );

    return logData_->getLineLength( LineNumber( *longestLine ) );
}

// Make the end of the lines in the selection visible
void AbstractLogView::jumpToEndOfLine()
{
    const auto selection = selection_.getLines();
    horizontalScrollBar()->setValue( type_safe::narrow_cast<int>( maxLineLength( selection ).get()
                                                                  - getNbVisibleCols().get() ) );
}

// Make the end of the lines on the screen visible
void AbstractLogView::jumpToRightOfScreen()
{
    const auto nbVisibleLines = getNbVisibleLines();

    klogg::vector<LineNumber::UnderlyingType> visibleLinesNumbers( nbVisibleLines.get() );
    std::iota( visibleLinesNumbers.begin(), visibleLinesNumbers.end(), firstLine_.get() );

    klogg::vector<LineNumber> visibleLines( nbVisibleLines.get() );
    std::transform( visibleLinesNumbers.cbegin(), visibleLinesNumbers.cend(), visibleLines.begin(),
                    []( auto number ) { return LineNumber{ number }; } );
    horizontalScrollBar()->setValue( type_safe::narrow_cast<int>(
        maxLineLength( visibleLines ).get() - getNbVisibleCols().get() ) );
}

// Jump to the first line
void AbstractLogView::jumpToTop()
{
    // This will also trigger a scrollContents event
    verticalScrollBar()->setValue( 0 );
    forceRefresh(); // in case the screen hasn't moved
}

// Jump to the last line
void AbstractLogView::jumpToBottom()
{
    const auto newTopLine
        = ( logData_->getNbLine().get() < getNbVisibleLines().get() )
              ? 0
              : logData_->getNbLine().get() - getNbBottomWrappedVisibleLines().get() + 1;

    // This will also trigger a scrollContents event
    verticalScrollBar()->setValue( lineNumberToVerticalScroll( LineNumber( newTopLine ) ) );

    forceRefresh();
}

// Select the word under the given position
void AbstractLogView::selectWordAtPosition( const FilePosition& pos )
{
    const QString line = logData_->getExpandedLineString( pos.line() );

    const int clickPos = type_safe::narrow_cast<int>( pos.column().get() );

    const auto isWordSeparator = []( QChar c ) {
        return !c.isLetterOrNumber() && c.category() != QChar::Punctuation_Connector;
    };

    if ( line.isEmpty() || isWordSeparator( line[ clickPos ] ) ) {
        return;
    }

    const auto wordStart
        = std::find_if( line.rbegin() + line.size() - clickPos, line.rend(), isWordSeparator );
    const auto selectionStart = LineColumn{ type_safe::narrow_cast<LineColumn::UnderlyingType>(
        std::distance( line.begin(), wordStart.base() ) ) };

    const auto wordEnd = std::find_if( line.begin() + clickPos, line.end(), isWordSeparator );
    const auto selectionEnd = LineColumn{ type_safe::narrow_cast<LineColumn::UnderlyingType>(
        std::distance( line.begin(), wordEnd ) - 1 ) };

    selection_.selectPortion( pos.line(), selectionStart, selectionEnd );
    updateGlobalSelection();
    forceRefresh();
}

// Update the system global (middle click) selection (X11 only)
void AbstractLogView::updateGlobalSelection()
{
    try {
        auto clipboard = QApplication::clipboard();
        // Updating it only for "non-trivial" (range or portion) selections
        if ( !selection_.isSingleLine() )
            clipboard->setText( selection_.getSelectedText( logData_ ), QClipboard::Selection );
    } catch ( std::exception& err ) {
        LOG_ERROR << "failed to copy data to clipboard " << err.what();
    }
}

void AbstractLogView::selectAndDisplayRange( FilePosition pos )
{
    disableFollow();
    selection_.selectRange( selectionStartPos_.line(), pos.line() );
    selectionCurrentEndPos_ = pos;
    displayLine( pos.line() );
    Q_EMIT newSelection( pos.line(), selection_.getSelectedLinesCount(), 0_lcol,
                         LineLength{ getSelectedText().size() } );
}

// Create the pop-up menu
void AbstractLogView::createMenu()
{
    copyAction_ = new QAction( tr( "&Copy" ), this );
    // No text as this action title depends on the type of selection
    connect( copyAction_, &QAction::triggered, this, [ this ]( auto ) { this->copy(); } );

    copyWithLineNumbersAction_ = new QAction( tr( "Copy with line numbers" ), this );
    // No text as this action title depends on the type of selection
    connect( copyWithLineNumbersAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->copyWithLineNumbers(); } );

    markAction_ = new QAction( tr( "&Mark" ), this );
    connect( markAction_, &QAction::triggered, this, [ this ]( auto ) { this->markSelected(); } );

    saveToFileAction_ = new QAction( tr( "Save to file" ), this );
    connect( saveToFileAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->saveToFile(); } );

    saveSelectedToFileAction_ = new QAction( tr( "Save selected to file" ), this );
    connect( saveSelectedToFileAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->saveSelectedToFile(); } );

    // For '#' and '*', shortcuts doesn't seem to work but
    // at least it displays them in the menu, we manually handle those keys
    // as keys event anyway (in keyPressEvent).
    findNextAction_ = new QAction( tr( "Find &next" ), this );
    findNextAction_->setShortcut( Qt::Key_Asterisk );
    findNextAction_->setStatusTip( tr( "Find the next occurrence" ) );
    connect( findNextAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->findNextSelected(); } );

    findPreviousAction_ = new QAction( tr( "Find &previous" ), this );
    findPreviousAction_->setShortcut( tr( "/" ) );
    findPreviousAction_->setStatusTip( tr( "Find the previous occurrence" ) );
    connect( findPreviousAction_, &QAction::triggered,
             [ this ]( auto ) { this->findPreviousSelected(); } );

    replaceSearchAction_ = new QAction( tr( "&Replace search" ), this );
    replaceSearchAction_->setStatusTip( tr( "Replace the search expression with the selection" ) );
    connect( replaceSearchAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->replaceSearch(); } );

    addToSearchAction_ = new QAction( tr( "&Add to search" ), this );
    addToSearchAction_->setStatusTip( tr( "Add the selection to the current search" ) );
    connect( addToSearchAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->addToSearch(); } );

    excludeFromSearchAction_ = new QAction( tr( "&Exclude from search" ), this );
    excludeFromSearchAction_->setStatusTip( tr( "Excludes the selection from search" ) );
    connect( excludeFromSearchAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->excludeFromSearch(); } );

    setSearchStartAction_ = new QAction( tr( "Set search start" ), this );
    connect( setSearchStartAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->setSearchStart(); } );

    setSearchEndAction_ = new QAction( tr( "Set search end" ), this );
    connect( setSearchEndAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->setSearchEnd(); } );

    clearSearchLimitAction_ = new QAction( tr( "Clear search limits" ), this );
    connect( clearSearchLimitAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->clearSearchLimits(); } );

    setSelectionStartAction_ = new QAction( tr( "Set selection start" ), this );
    connect( setSelectionStartAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->setSelectionStart(); } );

    setSelectionEndAction_ = new QAction( tr( "Set selection end" ), this );
    connect( setSelectionEndAction_, &QAction::triggered, this,
             [ this ]( auto ) { this->setSelectionEnd(); } );

    saveDefaultSplitterSizesAction_ = new QAction( tr( "Save splitter position" ), this );
    connect( saveDefaultSplitterSizesAction_, &QAction::triggered, this,
             [ this ]( auto ) { Q_EMIT saveDefaultSplitterSizes(); } );

    sendToScratchpadAction_ = new QAction( tr( "Send to scratchpad" ), this );
    connect( sendToScratchpadAction_, &QAction::triggered, this,
             [ this ]( auto ) { Q_EMIT sendSelectionToScratchpad(); } );

    replaceInScratchpadAction_ = new QAction( tr( "Replace scratchpad" ), this );
    connect( replaceInScratchpadAction_, &QAction::triggered, this,
             [ this ]( auto ) { Q_EMIT replaceScratchpadWithSelection(); } );

    popupMenu_ = new QMenu( this );
    highlightersMenu_ = new HighlightersMenu( tr( "Highlighters" ) );
    popupMenu_->addMenu( highlightersMenu_ );
    colorLabelsMenu_ = popupMenu_->addMenu( tr( "Color labels" ) );

    popupMenu_->addSeparator();
    popupMenu_->addAction( markAction_ );
    popupMenu_->addSeparator();
    popupMenu_->addAction( copyAction_ );
    popupMenu_->addAction( copyWithLineNumbersAction_ );
    popupMenu_->addAction( sendToScratchpadAction_ );
    popupMenu_->addAction( replaceInScratchpadAction_ );
    popupMenu_->addSeparator();
    popupMenu_->addAction( findNextAction_ );
    popupMenu_->addAction( findPreviousAction_ );
    popupMenu_->addSeparator();
    popupMenu_->addAction( replaceSearchAction_ );
    popupMenu_->addAction( addToSearchAction_ );
    popupMenu_->addAction( excludeFromSearchAction_ );
    popupMenu_->addSeparator();
    popupMenu_->addAction( setSearchStartAction_ );
    popupMenu_->addAction( setSearchEndAction_ );
    popupMenu_->addAction( clearSearchLimitAction_ );
    popupMenu_->addSeparator();
    popupMenu_->addAction( setSelectionStartAction_ );
    popupMenu_->addAction( setSelectionEndAction_ );
    popupMenu_->addSeparator();
    popupMenu_->addAction( saveDefaultSplitterSizesAction_ );
    popupMenu_->addAction( saveToFileAction_ );
    popupMenu_->addAction( saveSelectedToFileAction_ );
}

void AbstractLogView::considerMouseHovering( int xPos, int yPos )
{
    const auto line = convertCoordToLine( yPos );
    if ( ( xPos < leftMarginPx_ ) && ( line.has_value() ) && ( *line < logData_->getNbLine() ) ) {
        // Mouse moved in the margin, send event up
        // (possibly to highlight the overview)
        if ( line != lastHoveredLine_ ) {
            LOG_DEBUG << "Mouse moved in margin line: " << line;
            Q_EMIT mouseHoveredOverLine( *line );
            lastHoveredLine_ = line;
        }
    }
    else {
        if ( lastHoveredLine_.has_value() ) {
            Q_EMIT mouseLeftHoveringZone();
            lastHoveredLine_ = {};
        }
    }
}

LinesCount AbstractLogView::getNbBottomWrappedVisibleLines() const
{
    const auto visibleLines = getNbVisibleLines();

    if ( useTextWrap_ ) {
        const auto totalLines = logData_->getNbLine();

        LinesCount wrappedLinesCount{ 0 };
        LinesCount offset = 0_lcount;
        for ( ; offset < visibleLines && offset < totalLines && wrappedLinesCount < visibleLines;
              ++offset ) {
            LineNumber line = LineNumber{ totalLines.get() } - offset;
            wrappedLinesCount += LinesCount{ static_cast<LinesCount::UnderlyingType>(
                logData_->getLineLength( line ).get() / getNbVisibleCols().get() + 1 ) };
        }

        return offset;
    }
    return visibleLines;
}

void AbstractLogView::updateScrollBars()
{
    if ( logData_->getNbLine() < getNbVisibleLines() ) {
        verticalScrollBar()->setRange( 0, 0 );
    }
    else {
        const auto visibleWrappedLines = getNbBottomWrappedVisibleLines();
        const auto wrappedLinesScrollAdjust = ( getNbVisibleLines() - visibleWrappedLines ).get();

        verticalScrollBar()->setRange(
            0, static_cast<int>( std::min( logData_->getNbLine().get() - getNbVisibleLines().get()
                                               + LinesCount::UnderlyingType{ 1 }
                                               + wrappedLinesScrollAdjust,
                                           maxValue<LinesCount>().get() ) ) );
    }

    int64_t hScrollMaxValue = 0;
    if ( !useTextWrap_ && logData_->getMaxLength().get() >= getNbVisibleCols().get() ) {
        hScrollMaxValue = logData_->getMaxLength().get() - getNbVisibleCols().get() + 1;
    }

    horizontalScrollBar()->setRange( 0, type_safe::narrow_cast<int>( hScrollMaxValue ) );
    horizontalScrollBar()->setPageStep(
        type_safe::narrow_cast<int>( getNbVisibleCols().get() * 7 / 8 ) );
}

void AbstractLogView::drawTextArea( QPaintDevice* paintDevice )
{
    // LOG_DEBUG << "devicePixelRatio: " << viewport()->devicePixelRatio();
    // LOG_DEBUG << "viewport size: " << viewport()->size().width();
    // LOG_DEBUG << "pixmap size: " << textPixmap.width();
    // Repaint the viewport
    auto painter = pixmapPainter( paintDevice, this->font() );
    // LOG_DEBUG << "font: " << viewport()->font().family().toStdString();
    // LOG_DEBUG << "font painter: " << painter->font().family().toStdString();

    const int fontHeight = charHeight_;
    const int fontAscent = painter->fontMetrics().ascent();
    const LineLength nbVisibleCols = getNbVisibleCols();

    const int paintDeviceHeight
        = static_cast<int>( std::floor( paintDevice->height() / viewport()->devicePixelRatio() ) );
    const int paintDeviceWidth
        = static_cast<int>( std::floor( paintDevice->width() / viewport()->devicePixelRatio() ) );

    const QPalette& palette = viewport()->palette();
    const auto& highlighterSet = HighlighterSetCollection::get().currentActiveSet();
    const auto& quickHighlighters = HighlighterSetCollection::get().quickHighlighters();
    QColor foreColor, backColor;

    static const QBrush normalBulletBrush = QBrush( Qt::white );
    static const QBrush matchBulletBrush = QBrush( Qt::red );
    static const QBrush markBrush = QBrush( "dodgerblue" );
    static const QBrush markedMatchBrush = QBrush( "violet" );

    static constexpr int SeparatorWidth = 1;
    static constexpr int BulletAreaWidth = 11;
    static constexpr int ContentMarginWidth = 1;
    static constexpr int LineNumberPadding = 3;

    // First check the lines to be drawn are within range (might not be the case if
    // the file has just changed)
    const auto linesInFile = logData_->getNbLine();

    if ( firstLine_ >= linesInFile )
        firstLine_ = LineNumber( linesInFile.get() ? linesInFile.get() - 1 : 0 );

    const auto nbLines = qMin( getNbVisibleLines(), linesInFile - LinesCount( firstLine_.get() ) );

    const int bottomOfTextPx = static_cast<int>( nbLines.get() ) * fontHeight;

    LOG_DEBUG << "drawing lines from " << firstLine_ << " (" << nbLines << " lines)";
    LOG_DEBUG << "bottomOfTextPx: " << bottomOfTextPx;
    LOG_DEBUG << "Height: " << paintDeviceHeight;

    painter->fillRect( 0, 0, paintDeviceWidth, paintDeviceHeight,
                       palette.color( QPalette::Window ) );

    // First draw the bullet left margin
    painter->setPen( palette.color( QPalette::Text ) );
    painter->fillRect( 0, 0, BulletAreaWidth, paintDeviceHeight, Qt::darkGray );

    // Column at which the content should start (pixels)
    int contentStartPosX = BulletAreaWidth + SeparatorWidth;

    // This is also the bullet zone width, used for marking clicks
    bulletZoneWidthPx_ = contentStartPosX;

    // Update the length of line numbers
    const int nbDigitsInLineNumber = countDigits( maxDisplayLineNumber().get() );

    // Draw the line numbers area
    int lineNumberAreaStartX = 0;
    if ( lineNumbersVisible_ ) {
        const auto lineNumberWidth = charWidth_ * nbDigitsInLineNumber;
        const auto lineNumberAreaWidth = 2 * LineNumberPadding + lineNumberWidth;
        lineNumberAreaStartX = contentStartPosX;

        painter->setPen( palette.color( QPalette::Text ) );
        painter->fillRect( contentStartPosX - SeparatorWidth, 0,
                           lineNumberAreaWidth + SeparatorWidth, paintDeviceHeight, Qt::darkGray );

        painter->drawLine( contentStartPosX + lineNumberAreaWidth - SeparatorWidth, 0,
                           contentStartPosX + lineNumberAreaWidth - SeparatorWidth,
                           paintDeviceHeight );

        // Update for drawing the actual text
        contentStartPosX += lineNumberAreaWidth;
    }
    else {
        painter->fillRect( contentStartPosX - SeparatorWidth, 0, SeparatorWidth + 1,
                           paintDeviceHeight, palette.color( QPalette::Disabled, QPalette::Text ) );
        // contentStartPosX += SEPARATOR_WIDTH;
    }

    painter->drawLine( BulletAreaWidth, 0, BulletAreaWidth, paintDeviceHeight - 1 );

    // This is the total width of the 'margin' (including line number if any)
    // used for mouse calculation etc...
    leftMarginPx_ = contentStartPosX + SeparatorWidth;

    const auto searchStartIndex = lineIndex( searchStart_ );
    const auto searchEndIndex = [ this ] {
        auto index = lineIndex( searchEnd_ );
        if ( searchEnd_ + 1_lcount != displayLineNumber( index ) ) {
            // in filtered view lineIndex for "past the end" returns last line
            // it should not be marked as excluded
            index = index + 1_lcount;
        }

        return index;
    }();

    // Lines to write
    const auto expandedLines = logData_->getExpandedLines( firstLine_, nbLines );

    const auto highlightPatternMatches = Configuration::get().mainSearchHighlight();
    const auto variateHighlightPatternMatches = Configuration::get().variateMainSearchHighlight();

    std::optional<Highlighter> patternHighlight;
    if ( highlightPatternMatches && !searchPattern_.isBoolean && !searchPattern_.isExclude
         && !searchPattern_.pattern.isEmpty() ) {
        const auto mainSearchBackColor = Configuration::get().mainSearchBackColor();
        patternHighlight = Highlighter{};
        patternHighlight->setHighlightOnlyMatch( true );
        patternHighlight->setVariateColors( variateHighlightPatternMatches );
        patternHighlight->setPattern( searchPattern_.pattern );
        patternHighlight->setIgnoreCase( !searchPattern_.isCaseSensitive );
        patternHighlight->setUseRegex( !searchPattern_.isPlainText );

        patternHighlight->setBackColor( mainSearchBackColor );
        patternHighlight->setForeColor( Qt::black );
    }

    klogg::vector<Highlighter> additionalHighlighters;
    for ( auto i = 0u; i < quickHighlighters_.size(); ++i ) {
        const auto quickHighlighterIndex = static_cast<int>( i );
        if ( quickHighlighterIndex >= quickHighlighters.size() ) {
            LOG_WARNING << "Not enough quickHighlighters configured";
            break;
        }

        const auto quickHighlighter = quickHighlighters.at( quickHighlighterIndex );

        std::transform( quickHighlighters_[ i ].begin(), quickHighlighters_[ i ].end(),
                        std::back_inserter( additionalHighlighters ),
                        [ quickHighlighter ]( const QString& word ) {
                            Highlighter h{ word, false, true, quickHighlighter.color.foreColor,
                                           quickHighlighter.color.backColor };
                            h.setUseRegex( false );
                            return h;
                        } );
    }

    // Position in pixel of the base line of the line to print
    int yPos = 0;
    wrappedLinesNumbers_.clear();
    for ( auto currentLine = 0_lcount; currentLine < nbLines; ++currentLine ) {
        const auto lineNumber = firstLine_ + currentLine;
        const QString logLine = logData_->getLineString( lineNumber );

        const int xPos = contentStartPosX + ContentMarginWidth;

        klogg::vector<HighlightedMatch> highlighterMatches;

        if ( selection_.isLineSelected( lineNumber ) && !selection_.isSingleLine() ) {
            // Reverse the selected line
            foreColor = palette.color( QPalette::HighlightedText );
            backColor = palette.color( QPalette::Highlight );
            painter->setPen( palette.color( QPalette::Text ) );
        }
        else {
            const auto highlightType = highlighterSet.matchLine( logLine, highlighterMatches );

            if ( highlightType == HighlighterMatchType::LineMatch ) {
                // color applies to whole line
                foreColor = highlighterMatches.front().foreColor();
                backColor = highlighterMatches.front().backColor();
            }
            else {
                // Use the default colors
                if ( lineNumber < searchStartIndex || lineNumber >= searchEndIndex ) {
                    foreColor = palette.brush( QPalette::Disabled, QPalette::Text ).color();
                }
                else {
                    foreColor = palette.color( QPalette::Text );
                }

                backColor = palette.color( QPalette::Base );
            }

            if ( patternHighlight ) {
                klogg::vector<HighlightedMatch> patternMatches;
                patternHighlight->matchLine( logLine, patternMatches );
                highlighterMatches.insert( highlighterMatches.end(), patternMatches.begin(),
                                           patternMatches.end() );
            }

            highlighterMatches.reserve( additionalHighlighters.size() );
            for ( const auto& highlighter : additionalHighlighters ) {
                klogg::vector<HighlightedMatch> patternMatches;
                highlighter.matchLine( logLine, patternMatches );
                highlighterMatches.insert( highlighterMatches.end(), patternMatches.begin(),
                                           patternMatches.end() );
            }
        }

        const auto untabifyHighlight = [ &logLine ]( const auto& match ) {
#if QT_VERSION >= QT_VERSION_CHECK( 5, 10, 0 )
            const auto prefix = QStringView{ logLine }.left( match.startColumn().get() );
            const auto matchPart
                = QStringView{ logLine }.mid( match.startColumn().get(), match.size().get() );
#else
            const auto prefix = logLine.leftRef( match.startColumn().get() );
            const auto matchPart = logLine.midRef( match.startColumn().get(), match.size().get() );
#endif
            const auto expandedPrefixLength = untabify( prefix.toString() ).size();
            const LineLength startDelta
                = LineLength{ type_safe::narrow_cast<LineLength::UnderlyingType>(
                    expandedPrefixLength - prefix.size() ) };

            const LineLength expandedMatchLength = LineLength{
                untabify( matchPart.toString(),
                          LineColumn{ type_safe::narrow_cast<LineColumn::UnderlyingType>(
                              expandedPrefixLength ) } )
                    .size()
            };

            const auto lengthDelta
                = expandedMatchLength
                  - LineLength{ type_safe::narrow_cast<LineLength::UnderlyingType>(
                      matchPart.size() ) };

            return HighlightedMatch{ match.startColumn() + startDelta, match.size() + lengthDelta,
                                     match.foreColor(), match.backColor() };
        };

        klogg::vector<HighlightedMatch> allHighlights;
        allHighlights.reserve( highlighterMatches.size() );
        std::transform( highlighterMatches.cbegin(), highlighterMatches.cend(),
                        std::back_inserter( allHighlights ), untabifyHighlight );

        // string to print, cut to fit the length and position of the view
        const QString& expandedLine = expandedLines[ currentLine.get() ];

        // Has the line got elements to be highlighted
        klogg::vector<HighlightedMatch> quickFindMatches;
        quickFindPattern_->matchLine( expandedLine, quickFindMatches );
        allHighlights.insert( allHighlights.end(),
                              std::make_move_iterator( quickFindMatches.begin() ),
                              std::make_move_iterator( quickFindMatches.end() ) );

        // Is there something selected in the line?
        const auto selectionPortion = selection_.getPortionForLine( lineNumber );
        if ( selectionPortion.isValid() ) {
            allHighlights.emplace_back( selectionPortion.startColumn(), selectionPortion.size(),
                                        palette.color( QPalette::HighlightedText ),
                                        palette.color( QPalette::Highlight ) );
        }

        const auto wrappedLineLength
            = useTextWrap_ ? nbVisibleCols : LineLength{ klogg::isize( expandedLine ) + 1 };
        const WrappedLinesView wrappedLineView{ expandedLine, wrappedLineLength };
        const auto finalLineHeight
            = fontHeight * static_cast<int>( wrappedLineView.wrappedLinesCount() );
        // LOG_INFO << "Draw line " << lineNumber << ": " << expandedLine;

        painter->fillRect( xPos - ContentMarginWidth, yPos, viewport()->width(), finalLineHeight,
                           backColor );

        LineDrawer lineDrawer( backColor );
        if ( !allHighlights.empty() && !expandedLine.isEmpty() ) {
            auto highlightColors = klogg::vector<std::pair<QColor, QColor>>(
                static_cast<size_t>( expandedLine.size() ),
                std::make_pair( foreColor, backColor ) );

            for ( const auto& match : allHighlights ) {
                auto matchEnd = match.startColumn() + match.size();
                auto matchLengthInString = match.size();
                if ( matchEnd >= LineColumn{ expandedLine.size() } ) {
                    matchLengthInString
                        = LineLength{ klogg::isize( expandedLine ) - match.startColumn().get() };
                }
                if ( matchLengthInString > 0_length ) {
                    std::fill_n( highlightColors.begin() + match.startColumn().get(),
                                 matchLengthInString.get(),
                                 std::make_pair( match.foreColor(), match.backColor() ) );
                }
            }

            klogg::vector<LineColumn> columnIndexes( highlightColors.size() );
            std::iota( columnIndexes.begin(), columnIndexes.end(), 0_lcol );

            auto columnIndexIt = columnIndexes.begin();

            const auto firstVisibleColumn
                = std::clamp( useTextWrap_ ? 0_lcol : firstCol_, 0_lcol,
                              LineColumn{ klogg::isize( expandedLine ) } );
            std::advance( columnIndexIt, firstVisibleColumn.get() );
            while ( columnIndexIt != columnIndexes.end() ) {
                auto highlightDiffColumnIt = std::adjacent_find(
                    columnIndexIt, columnIndexes.end(),
                    [ &highlightColors ]( LineColumn lhsColumn, LineColumn rhsColumn ) {
                        return highlightColors[ lhsColumn.get<size_t>() ]
                               != highlightColors[ rhsColumn.get<size_t>() ];
                    } );

                if ( highlightDiffColumnIt != columnIndexes.end() ) {
                    auto highlightChunkStart = *columnIndexIt;
                    auto highlightChunkEnd = *highlightDiffColumnIt;
                    lineDrawer.addChunk(
                        highlightChunkStart, highlightChunkEnd,
                        highlightColors[ highlightChunkStart.get<size_t>() ].first,
                        highlightColors[ highlightChunkStart.get<size_t>() ].second );

                    columnIndexIt = highlightDiffColumnIt + 1;
                }
                else {
                    break;
                }
            }
            if ( columnIndexIt != columnIndexes.end() && !columnIndexes.empty() ) {
                const auto lastHighlightChunkStart = *columnIndexIt;
                const auto lastHighlightChunkEnd = *columnIndexes.rbegin();
                if ( lastHighlightChunkEnd >= lastHighlightChunkStart ) {
                    lineDrawer.addChunk( lastHighlightChunkStart, lastHighlightChunkEnd,
                                         highlightColors.back().first,
                                         highlightColors.back().second );
                }
            }
        }
        else {
            if ( useTextWrap_ ) {
                lineDrawer.addChunk( 0_lcol, LineColumn{ expandedLine.size() }, foreColor,
                                     backColor );
            }
            else {
                lineDrawer.addChunk( firstCol_, firstCol_ + nbVisibleCols, foreColor, backColor );
            }
        }
        lineDrawer.draw( painter.get(), xPos, yPos, viewport()->width(), wrappedLineView,
                         ContentMarginWidth );

        if ( ( selection_.isLineSelected( lineNumber ) && selection_.isSingleLine() )
             || selection_.getPortionForLine( lineNumber ).isValid() ) {
            auto selectionPen = QPen( palette.color( QPalette::Highlight ) );
            selectionPen.setWidth( 1 );
            painter->setPen( selectionPen );
            painter->drawLine( xPos - ContentMarginWidth + 1, yPos, viewport()->width(), yPos );
            painter->drawLine( xPos - ContentMarginWidth + 1, yPos + finalLineHeight - 1,
                               viewport()->width(), yPos + finalLineHeight - 1 );
        }

        // Then draw the bullet
        painter->setPen( Qt::black );
        const int circleSize = 3;
        const int arrowHeight = 4;
        const int middleXLine = BulletAreaWidth / 2;
        const int middleYLine = yPos + ( fontHeight / 2 );

        using LineTypeFlags = AbstractLogData::LineTypeFlags;
        const auto currentLineType = lineType( lineNumber );
        if ( currentLineType.testFlag( LineTypeFlags::Mark ) ) {
            // A pretty arrow if the line is marked
            const QPointF points[ 7 ] = {
                QPointF( 1, middleYLine - 2 ),
                QPointF( middleXLine, middleYLine - 2 ),
                QPointF( middleXLine, middleYLine - arrowHeight ),
                QPointF( BulletAreaWidth - 1, middleYLine ),
                QPointF( middleXLine, middleYLine + arrowHeight ),
                QPointF( middleXLine, middleYLine + 2 ),
                QPointF( 1, middleYLine + 2 ),
            };

            painter->setBrush( currentLineType.testFlag( LineTypeFlags::Match ) ? markedMatchBrush
                                                                                : markBrush );
            painter->drawPolygon( points, 7 );
        }
        else {
            // For pretty circles
            painter->setRenderHint( QPainter::Antialiasing );

            QBrush brush = normalBulletBrush;
            if ( currentLineType.testFlag( LineTypeFlags::Match ) )
                brush = matchBulletBrush;
            painter->setBrush( brush );
            painter->drawEllipse( middleXLine - circleSize, middleYLine - circleSize,
                                  circleSize * 2, circleSize * 2 );
        }

        // Draw the line number
        if ( lineNumbersVisible_ ) {
            static const QString lineNumberFormat( "%1" );
            const QString& lineNumberStr = lineNumberFormat.arg(
                displayLineNumber( lineNumber ).get(), nbDigitsInLineNumber );
            painter->setPen( Qt::white );
            painter->drawText( lineNumberAreaStartX + LineNumberPadding, yPos + fontAscent,
                               lineNumberStr );
        }
        for ( auto i = 0u; i < wrappedLineView.wrappedLinesCount(); ++i ) {
            wrappedLinesNumbers_.push_back( std::make_pair( lineNumber, i ) );
        }

        yPos += finalLineHeight;
        if ( yPos > viewport()->height() ) {
            break;
        }
    } // For each line
}

// Draw the "pull to follow" bar and return a pixmap.
// The width is passed in "logic" pixels.
QPixmap AbstractLogView::drawPullToFollowBar( int width, qreal pixelRatio )
{
    static constexpr int barWidth = 40;
    QPixmap pixmap( static_cast<int>( width * pixelRatio ), static_cast<int>( barWidth * 6.0 ) );
    pixmap.setDevicePixelRatio( pixelRatio );
    pixmap.fill( this->palette().color( this->backgroundRole() ) );
    const int nbBars = width / ( barWidth * 2 ) + 1;

    QPainter painter( &pixmap );
    painter.setPen( QPen( QColor( 0, 0, 0, 0 ) ) );
    painter.setBrush( QBrush( QColor( "lightyellow" ) ) );

    for ( int i = 0; i < nbBars; ++i ) {
        QPoint points[ 4 ] = { { ( i * 2 + 1 ) * barWidth, 0 },
                               { 0, ( i * 2 + 1 ) * barWidth },
                               { 0, ( i + 1 ) * 2 * barWidth },
                               { ( i + 1 ) * 2 * barWidth, 0 } };
        painter.drawConvexPolygon( points, 4 );
    }

    return pixmap;
}

void AbstractLogView::disableFollow()
{
    Q_EMIT followModeChanged( false );
    followElasticHook_.hook( false );
}

void AbstractLogView::setColorLabel( QAction* action )
{
    if ( action->data().isValid() ) {
        Q_EMIT addColorLabel( static_cast<size_t>( action->data().toInt() ) );
    }
    else {
        Q_EMIT clearColorLabels();
    }
}

namespace {

// Convert the length of the pull to follow bar to pixels
int mapPullToFollowLength( int length )
{
    return length / 14;
}

} // namespace
