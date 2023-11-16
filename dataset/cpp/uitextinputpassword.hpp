#ifndef EE_UICUITEXTINPUTPASSWORD_HPP
#define EE_UICUITEXTINPUTPASSWORD_HPP

#include <eepp/ui/uitextinput.hpp>

namespace EE { namespace UI {

class EE_API UITextInputPassword : public UITextInput {
  public:
	static UITextInputPassword* New();

	UITextInputPassword();

	~UITextInputPassword();

	virtual void draw();

	virtual const String& getText() const;

	virtual UITextView* setText( const String& text );

	Text* getPassCache() const;

	const String& getBulletCharacter() const;

	void setBulletCharacter( const String& bulletCharacter );

  protected:
	Text* mPassCache;
	Vector2f mHintAlignOffset;
	String mBulletCharacter;

	void alignFix();

	void updateText();

	void updatePass( const String& pass );

	void updateFontStyleConfig();

	virtual void onStateChange();

	virtual void onFontChanged();
};

}} // namespace EE::UI

#endif
