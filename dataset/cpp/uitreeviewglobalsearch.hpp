#ifndef ECODE_UITREEVIEWGLOBALSEARCH_HPP
#define ECODE_UITREEVIEWGLOBALSEARCH_HPP

#include "projectsearch.hpp"
#include <eepp/ui/doc/syntaxcolorscheme.hpp>
#include <eepp/ui/uitreeview.hpp>

using namespace EE::UI;
using namespace EE::UI::Doc;

namespace ecode {

class UITreeViewCellGlobalSearch : public UITreeViewCell {
  public:
	static UITreeViewCellGlobalSearch* New( bool selectionEnabled, bool hAsCPP ) {
		return eeNew( UITreeViewCellGlobalSearch, ( selectionEnabled, hAsCPP ) );
	}

	explicit UITreeViewCellGlobalSearch( bool selectionEnabled, bool hAsCPP );

	UIPushButton* setText( const String& text );

	virtual void draw();

	virtual void updateCell( Model* model );

	void toggleSelected();

  protected:
	std::pair<size_t, size_t> mSearchStrPos;
	String mResultStr;
	bool mHAsCpp{ false };

	std::function<UITextView*( UIPushButton* )> getCheckBoxFn();

	void* getDataPtr( const ModelIndex& modelIndex );

	UIPushButton* updateText( const std::string& text );

	ProjectSearch::ResultData::Result* getResultPtr();

	ProjectSearch::ResultData* getResultDataPtr();
};

class UITreeViewGlobalSearch : public UITreeView {
  public:
	static UITreeViewGlobalSearch* New( const SyntaxColorScheme& colorScheme, bool searchReplace ) {
		return eeNew( UITreeViewGlobalSearch, ( colorScheme, searchReplace ) );
	}

	UITreeViewGlobalSearch( const SyntaxColorScheme& colorScheme, bool searchReplace );

	UIWidget* createCell( UIWidget* rowWidget, const ModelIndex& index );

	const SyntaxColorScheme& getColorScheme() const { return mColorScheme; }

	const Color& getLineNumColor() const { return mLineNumColor; }

	void updateColorScheme( const SyntaxColorScheme& colorScheme ) { mColorScheme = colorScheme; }

	void setSearchStr( const String& searchStr ) { mSearchStr = searchStr; }

	const String& getSearchStr() const { return mSearchStr; }

	bool hAsCPP = false;
  protected:
	Color mLineNumColor;
	SyntaxColorScheme mColorScheme;
	String mSearchStr;
	bool mSearchReplace{ false };

	virtual Uint32 onKeyDown( const KeyEvent& event );
};

} // namespace ecode

#endif // ECODE_UITREEVIEWGLOBALSEARCH_HPP
