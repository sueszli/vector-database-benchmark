/********************************************************************************
 * The Peacenet - bit::phoenix("software");
 * 
 * MIT License
 *
 * Copyright (c) 2018-2019 Michael VanOverbeek, Declan Hoare, Ian Clary, 
 * Trey Smith, Richard Moch, Victor Tran and Warren Harris
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * Contributors:
 *  - Michael VanOverbeek <alkaline@bitphoenixsoftware.com>
 *
 ********************************************************************************/


#include "PTerminalWidget.h"
#include "FConsoleReadLineLatentAction.h"
#include "CommonUtils.h"
#include "Rendering/DrawElements.h"
#include "FTerminalSlowTypeLatentAction.h"

FString UPTerminalWidget::Sanitize(FString InDirtyBitch)
{
	FString Ret;
	TArray<TCHAR> Chars = InDirtyBitch.GetCharArray();
	ETerminalColor CurrentColor;
	FSlateFontInfo font;
	bool invert;
	bool attention;

	for(int i = 0; i < Chars.Num(); i++)
	{
		if(Chars[i] == TEXT('\0'))
			continue;

		bool wasLiteral =false;
		if (this->ParseControlCode(InDirtyBitch, i, CurrentColor, font, invert, attention, wasLiteral))
		{
			if(!wasLiteral)
			{
				continue;
			}
		}

		Ret.AppendChar(Chars[i]);
	}
	return Ret;
}

FReply UPTerminalWidget::NativeOnMouseButtonUp( const FGeometry& InGeometry, const FPointerEvent& InMouseEvent )
{
	this->Selecting = false;
	return FReply::Handled().SetUserFocus(this->TakeWidget());
}

FReply UPTerminalWidget::NativeOnMouseMove( const FGeometry& InGeometry, const FPointerEvent& InMouseEvent )
{
	if(this->Selecting)
	{
		FVector2D MousePos = InGeometry.AbsoluteToLocal(InMouseEvent.GetScreenSpacePosition());

		int end = GetCharIndexAtPosition(InGeometry, MousePos);

		if(end == -1)
		{
			SelectionEnd = this->TextBuffer.Len();
		}
		else if(end < SelectionStart)
		{
			SelectionEnd = SelectionStart;
			SelectionStart = end;
		}
		else
		{
			SelectionEnd = end;
		}
	}
	return FReply::Handled().SetUserFocus(this->TakeWidget());
}

int UPTerminalWidget::GetCharIndexAtPosition(const FGeometry InGeometry, FVector2D InPosition)
{
	// Where are we?
	float x = 0, y = 0;

	y -= this->ScrollOffsetY;

	// What's in the text buffer?
	TArray<TCHAR> chars = this->TextBuffer.GetCharArray();

	// How many chars are there?
	int num = chars.Num();

	// Get the size in pixels of the geometry so we know how big our widget is.
	FVector2D size = InGeometry.GetLocalSize();
	FSlateFontInfo font = this->RegularTextFont;
	const UFont* FontPtr = Cast<UFont>(font.FontObject);
	//The width of a character.
	float char_w = CharacterWidth * ZoomFactor;
	//The height of a character.
	float char_h = CharacterHeight * ZoomFactor;
	
	TCHAR Last = TEXT('\0');
	
	ETerminalColor CurrentColor = ETerminalColor::White;
	ETerminalColor CurrentBackgroundColor = ETerminalColor::Black;

	bool invert = false;
	bool attention = false;

	// Go through every character in the array.
	for(int i = 0; i < num; i++)
	{
		bool wasLiteral = false;

		if(chars[i] == TEXT('\0'))
			continue;

		if (this->ParseControlCode(this->TextBuffer, i, CurrentColor, font, invert, attention, wasLiteral))
		{
			if (!wasLiteral)
			{
				continue;
			}
		}

		bool IsInLine = (InPosition.Y >= y) && (InPosition.Y <= y + char_h);

		//Get the character.
		TCHAR c = chars[i];

		float kerning = 0.f;

		if (Last != TEXT('\0'))
		{
			kerning = FontPtr->GetCharKerning(Last, c) * ZoomFactor;
		}

		// If the character is a tab ('\t'), handle it.
		if (c == TEXT('\t'))
		{
			// Get remainder of char_x divided by 8 * char_w.
			float space = FMath::Fmod(x, char_w * 8);

			if(IsInLine)
			{
				if(InPosition.X >= x && InPosition.X <= x + ((char_w * 8) - space))
					return i;
			}

			// Realistically, truncating the value won't make much of a difference.
			// But we'll store it as a float simply to make Unreal's life easier.
			// All we need to do is add this value to char_x.
			x += (char_w * 8) - space;

			Last = TEXT('\0');

			// Go to next char.
			continue;
		}

		//return to char 0 if we're a \r
		if (c == TEXT('\r'))
		{
			x = 0;
			kerning = 0;
			Last = TEXT('\0');
			continue;
		}

		if (c == TEXT('\0'))
			break; //Stop rendering at a null terminator because this is C++. I hate this.

		//Drop down to a new line.
		if (c == TEXT('\n'))
		{
			if(IsInLine && InPosition.X >= x)
				return i;
			x = 0;
			y += char_h;
			Last = TEXT('\0');
			continue;
		}

		//Skip vertical tabs.
		if (c == TEXT('\v'))
			continue;

		UCommonUtils::MeasureChar(c, font, char_w, char_h);
		char_w = char_w * ZoomFactor;
		char_h = char_h * ZoomFactor;

		x += kerning;

		//If char_x + char_w is greater than our width, drop down to a new line.
		if (x + char_w > size.X)
		{
			kerning = 0;
			Last = TEXT('\0');
			x = 0;
			y += char_h;

			IsInLine = (InPosition.Y >= y) && (InPosition.Y <= y + char_h);
		}

		if(IsInLine)
		{
			if(InPosition.X >= x && InPosition.X <= x + char_w)
				return i;
		}

		x += char_w;
	}

	// We didn't select anything...
	return -1;
}

FReply UPTerminalWidget::NativeOnMouseButtonDown( const FGeometry& InGeometry, const FPointerEvent& InMouseEvent )
{
		// Is this a right-click?
	if(InMouseEvent.GetEffectingButton().GetFName() == TEXT("RightMouseButton"))
	{
		// Do we have a selection?
		if(this->SelectionStart != -1 && this->SelectionEnd != -1)
		{
			// Get the length of the text.
			int len = this->TextBuffer.Len();

			// Get the length of the selection.
			int selectionLen = SelectionEnd - SelectionStart;

			// Get the length of the text starting from the selection start.
			int endLen = len - SelectionStart;

			// If the selection len is zsn't zero...
			if(selectionLen > 0)
			{
				// If it's les than or equal to the end len...
				if(selectionLen <= endLen)
				{
					FString substring = this->TextBuffer.RightChop(SelectionStart).LeftChop(selectionLen);

					UCommonUtils::PutClipboardText(this->Sanitize(substring));

					SelectionStart = SelectionEnd = -1;

					return FReply::Handled().SetUserFocus(this->TakeWidget());
				}
				else
				{
					// We copy the text buffer from the start of the selection.
					FString substring = this->TextBuffer.RightChop(SelectionStart).LeftChop(endLen);

					// Put it in the clipboard.
					UCommonUtils::PutClipboardText(this->Sanitize(substring));

					SelectionStart = SelectionEnd = -1;

					return FReply::Handled().SetUserFocus(this->TakeWidget());
				}
			}
		}

		FString ClipboardContent;
		if(UCommonUtils::GetClipboardText(ClipboardContent))
		{
			TextInputBuffer.Append(ClipboardContent);
			if(this->EchoInputText)
			{
				TextBuffer.Append(ClipboardContent);
				this->NewTextAdded=true;
			}
		}
		return FReply::Handled().SetUserFocus(this->TakeWidget());
	}
	else if(InMouseEvent.GetEffectingButton().GetFName() == TEXT("LeftMouseButton"))
	{
		if(SelectionStart == -1)
		{
			FVector2D MousePos = InGeometry.AbsoluteToLocal(InMouseEvent.GetScreenSpacePosition());

			SelectionStart = GetCharIndexAtPosition(InGeometry, MousePos);
			SelectionEnd = SelectionStart;

			this->Selecting = true;

			return FReply::Handled().SetUserFocus(this->TakeWidget());			
		}
		else
		{
			SelectionStart = SelectionEnd = -1;
			return FReply::Handled().SetUserFocus(this->TakeWidget());			
		}
	}
	return FReply::Handled().SetUserFocus(this->TakeWidget());
}

FSlateFontInfo UPTerminalWidget::ZoomText(FSlateFontInfo InFont) const
{
	FSlateFontInfo Zoomed = FSlateFontInfo(InFont);
	Zoomed.Size = (int)(Zoomed.Size * ZoomFactor);
	return Zoomed;
}

bool UPTerminalWidget::SkipControlCode(const FString & InBuffer, int & InIndex, bool& OutLiteral) const
{
	OutLiteral = false;

	TCHAR c = InBuffer[InIndex];

	if (c != '&')
		return false;

	if (InIndex < InBuffer.GetCharArray().Num() - 2)
	{
		if (InBuffer[InIndex + 1] == '&')
		{
			OutLiteral = true;
			return true;
		}
	}

	InIndex ++;
	return true;
}

bool UPTerminalWidget::ParseControlCode(const FString & InBuffer, int & InIndex, ETerminalColor & OutColor, FSlateFontInfo & OutFont, bool & OutInvert, bool & OutAttention, bool& OutLiteral) const
{
	OutLiteral = false;

	char ctrl = InBuffer[InIndex];

	if (ctrl != '&')
		return false;
	
	InIndex++;
	if (InIndex < InBuffer.GetCharArray().Num() - 1)
	{
		ctrl = InBuffer[InIndex];

		if (ctrl == '&')
		{
			OutLiteral = true;
			return true;
		}

		if (!UCommonUtils::IsColorCode("&" + FString::Chr(ctrl), OutColor))
		{
			if (ctrl == 'r') // Reset
			{
				OutInvert = false;
				OutAttention = false;
				OutFont = this->RegularTextFont;
			}
			else if (ctrl == '!') // invert
			{
				OutInvert = true;
			}
			else if (ctrl == '~') // attention
			{
				OutAttention = true;
			}
			else if (ctrl == '*') // bold
			{
				OutFont = this->BoldTextFont;
			}
			else if (ctrl == '_') // italic
			{
				OutFont = this->ItalicTextFont;
			}
			else if (ctrl == '-') // both
			{
				OutFont = this->BoldItalicTextFont;
			}
		}
	}

	return true;
}

void UPTerminalWidget::Exit()
{
	OnExit.Broadcast();
}

void UPTerminalWidget::ReadLine(UObject* WorldContextObject, struct FLatentActionInfo LatentInfo, FString& OutText)
{
	if (WorldContextObject) 
	{
		UWorld* world = WorldContextObject->GetWorld();
		if (world)
		{
			FLatentActionManager& LatentActionManager = world->GetLatentActionManager();
			if (LatentActionManager.FindExistingAction<FConsoleReadLineLatentAction>(LatentInfo.CallbackTarget, LatentInfo.UUID) == NULL)
			{

				//Here in a second, once I confirm the project loads, we need to see whats wrong with this
				LatentActionManager.AddNewAction(LatentInfo.CallbackTarget, LatentInfo.UUID, new FConsoleReadLineLatentAction(this, LatentInfo, OutText));
			}
		}
	}
}

UPTerminalWidget * UPTerminalWidget::SlowlyWriteText(UObject * WorldContextObject, FLatentActionInfo LatentInfo, const FString & InText, float InDelayTime)
{
	if (WorldContextObject)
	{
		UWorld* world = WorldContextObject->GetWorld();
		if (world)
		{
			FLatentActionManager& LatentActionManager = world->GetLatentActionManager();
			if (LatentActionManager.FindExistingAction<FTerminalSlowTypeLatentAction>(LatentInfo.CallbackTarget, LatentInfo.UUID) == NULL)
			{

				//Here in a second, once I confirm the project loads, we need to see whats wrong with this
				LatentActionManager.AddNewAction(LatentInfo.CallbackTarget, LatentInfo.UUID, new FTerminalSlowTypeLatentAction(this, LatentInfo, InText, InDelayTime));
			}
		}
	}
	return this;

}

UPTerminalWidget * UPTerminalWidget::SlowlyWriteLine(UObject * WorldContextObject, FLatentActionInfo LatentInfo, const FString & InText, float InDelayTime)
{
	return SlowlyWriteText(WorldContextObject, LatentInfo, InText + TEXT("\n"), InDelayTime);
}

UPTerminalWidget * UPTerminalWidget::SlowlyOverwriteLine(UObject * WorldContextObject, FLatentActionInfo LatentInfo, const FString & InText, float InDelayTime)
{
	return SlowlyWriteText(WorldContextObject, LatentInfo, InText + TEXT("\r"), InDelayTime);
}

UPTerminalWidget* UPTerminalWidget::Write(FString InText)
{
	TextBuffer.Append(InText);
	NewTextAdded = true;

	bCursorActive = true;
	cursorTime = 0;

	return this;
}

UPTerminalWidget* UPTerminalWidget::WriteLine(FString InText)
{
	return Write(InText + TEXT("\n"));
}

UPTerminalWidget* UPTerminalWidget::OverwriteLine(FString InText)
{
	return Write(InText + TEXT("\r"));
}

UPTerminalWidget* UPTerminalWidget::Clear()
{
	TextBuffer.Empty(0);
	NewTextAdded = true;
	return this;
}

FReply UPTerminalWidget::NativeOnFocusReceived(const FGeometry & InGeometry, const FFocusEvent & InFocusEvent)
{
	cursorTime = 0;
	bCursorActive = true;

	return FReply::Handled();
}

void UPTerminalWidget::NativeConstruct()
{
	UCommonUtils::MeasureChar('#', this->RegularTextFont, CharacterWidth, CharacterHeight);

	SetVisibility(ESlateVisibility::Visible);

	Super::NativeConstruct();
}

int32 UPTerminalWidget::NativePaint(const FPaintArgs& Args, const FGeometry& AllottedGeometry, const FSlateRect& MyCullingRect, FSlateWindowElementList& OutDrawElements, int32 LayerId, const FWidgetStyle& InWidgetStyle, bool bParentEnabled) const
{
	// Get the size in pixels of the geometry so we know how big our widget is.
	FVector2D size = AllottedGeometry.GetLocalSize();
	FSlateFontInfo font = this->RegularTextFont;
	const UFont* FontPtr = Cast<UFont>(font.FontObject);
	float char_x = 0.f; //Where will the character be rendered on the X coordinate?
	float char_y = 0.f - ScrollOffsetY; //The current text position on the Y axis, accounting for the scroll offset.
	//The width of a character.
	float char_w = CharacterWidth * ZoomFactor;
	//The height of a character.
	float char_h = CharacterHeight * ZoomFactor;
	TArray<TCHAR> arr = TextBuffer.GetCharArray();
	TCHAR Last = TEXT('\0');
	//How many elements are in the text array?
	int arrNum = arr.Num() - 1; //there's a \0 at the end

	ETerminalColor CurrentColor = ETerminalColor::White;
	ETerminalColor CurrentBackgroundColor = ETerminalColor::Black;

	bool invert = false;
	bool attention = false;

	if (bRenderBackground)
	{
		//Draws the background of the terminal.
		LayerId++;

		if (TerminalBrush)
		{
			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(FVector2D::ZeroVector, size),
				&TerminalBrush->Brush,
				ESlateDrawEffect::None,
				FLinearColor::Black);
		}
	}

	for (int i = 0; i < arrNum; i++)
	{
		bool wasLiteral = false;
		if (this->ParseControlCode(this->TextBuffer, i, CurrentColor, font, invert, attention, wasLiteral))
		{
			if (!wasLiteral)
			{
				continue;
			}
		}

		//Get the character.
		TCHAR c = arr[i];

		float kerning = 0.f;

		if (Last != TEXT('\0'))
		{
			kerning = FontPtr->GetCharKerning(Last, c) * ZoomFactor;
		}

		// If the character is a tab ('\t'), handle it.
		if (c == TEXT('\t'))
		{
			// Get remainder of char_x divided by 8 * char_w.
			float space = FMath::Fmod(char_x, char_w * 8);

			// Realistically, truncating the value won't make much of a difference.
			// But we'll store it as a float simply to make Unreal's life easier.
			// All we need to do is add this value to char_x.
			char_x += (char_w * 8) - space;

			Last = TEXT('\0');

			// Go to next char.
			continue;
		}

		//return to char 0 if we're a \r
		if (c == TEXT('\r'))
		{
			char_x = 0;
			kerning = 0;
			Last = TEXT('\0');
			continue;
		}

		if (c == TEXT('\0'))
			break; //Stop rendering at a null terminator because this is C++. I hate this.

		//Drop down to a new line.
		if (c == TEXT('\n'))
		{
			char_x = 0;
			char_y += char_h;
			Last = TEXT('\0');
			continue;
		}

		//Skip vertical tabs.
		if (c == TEXT('\v'))
			continue;

		UCommonUtils::MeasureChar(c, font, char_w, char_h);
		char_w = char_w * ZoomFactor;
		char_h = char_h * ZoomFactor;

		char_x += kerning;

		//If char_x + char_w is greater than our width, drop down to a new line.
		if (char_x + char_w > size.X)
		{
			kerning = 0;
			Last = TEXT('\0');
			char_x = 0;
			char_y += char_h;
		}

		//If char_y is < 0, there is no reason to do any of this except move char_x by char_w.
		if (char_y < 0.f)
		{
			char_x += char_w;
			continue;
		}

		//if char_y is >= our control's height, we have NO REASON to continue rendering.
		if (char_y >= size.Y)
			break;

		if((SelectionStart > -1 && i >= SelectionStart) && (SelectionEnd > -1 && i <= SelectionEnd))
		{
			LayerId++;

			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(FVector2D(char_x, char_y), FVector2D(char_w, char_h)),
				&TerminalBrush->Brush,
				ESlateDrawEffect::None,
				UCommonUtils::GetTerminalColor(ETerminalColor::White));



			LayerId++;

			FSlateDrawElement::MakeText(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToOffsetPaintGeometry(FVector2D(char_x, char_y)),
				FText::FromString(FString::Chr(c)),
				this->ZoomText(font),
				ESlateDrawEffect::None,
				UCommonUtils::GetTerminalColor(ETerminalColor::Black));

		}
		else
		{
			LayerId++;

			FSlateDrawElement::MakeText(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToOffsetPaintGeometry(FVector2D(char_x, char_y)),
				FText::FromString(FString::Chr(c)),
				this->ZoomText(font),
				ESlateDrawEffect::None,
				UCommonUtils::GetTerminalColor(CurrentColor));
		}

		Last = c;

		//Advance the x coordinate by a single char.
		char_x += char_w;
	}

	//One last time so the cursor isn't cut off and is an accurate representation of where new text will be placed.
	if (char_x + char_w > size.X)
	{
		char_x = 0;
		char_y += char_h;
	}


	if (TerminalBrush && bCursorActive)
	{
		LayerId++;

		FSlateDrawElement::MakeBox(
			OutDrawElements,
			LayerId,
			AllottedGeometry.ToPaintGeometry(FVector2D(char_x, char_y), FVector2D(char_w, char_h)),
			&TerminalBrush->Brush,
			ESlateDrawEffect::None,
			UCommonUtils::GetTerminalColor(CurrentColor));

		if (!HasAnyUserFocus())
		{
			LayerId++;

			FSlateDrawElement::MakeBox(
				OutDrawElements,
				LayerId,
				AllottedGeometry.ToPaintGeometry(FVector2D(char_x+2, char_y+2), FVector2D(char_w-4, char_h-4)),
				&TerminalBrush->Brush,
				ESlateDrawEffect::None,
				UCommonUtils::GetTerminalColor(CurrentBackgroundColor));

		}
	}


	return LayerId;
}

void UPTerminalWidget::InjectInput(const FString& Input)
{
	this->TextInputBuffer = Input + "\n";
	if (this->EchoInputText)
	{
		this->WriteLine(Input);
	}
}

void UPTerminalWidget::NativeTick(const FGeometry& MyGeometry, float InDeltaTime)
{
	Super::NativeTick(MyGeometry, InDeltaTime);

	if (HasAnyUserFocus())
	{
		cursorTime += InDeltaTime;
		if (cursorTime >= CursorBlinkTimeMS)
		{
			cursorTime = 0;
			bCursorActive = !bCursorActive;
		}
	}
	else {
		cursorTime = 0;
		bCursorActive = true;
	}

	//Grab the width and height in pixels of the char '#'. Ideally, all terminals use monospace fonts, so, every character should just be the same size.
	//I could program this to grab the size of every character as I render them but I don't want to do that.
	//Plus this will kinda make things faster.
	//
	//Just use a fucking monospace font, god.
	UCommonUtils::MeasureChar('#', this->RegularTextFont, CharacterWidth, CharacterHeight);

	FVector2D gSize = MyGeometry.GetLocalSize();

	if (GeometrySize != gSize || NewTextAdded)
	{
		NewTextAdded = false;
		GeometrySize = gSize;

		float h = GetLineHeight();

		MaxScrollOffset = FMath::Max(0.f, h - GeometrySize.Y);

		ScrollOffsetY = MaxScrollOffset;
	}



}

FReply UPTerminalWidget::NativeOnMouseWheel(const FGeometry & InGeometry, const FPointerEvent & InMouseEvent)
{
	float wheelDelta = InMouseEvent.GetWheelDelta();

	ScrollOffsetY = FMath::Clamp(ScrollOffsetY - (wheelDelta*CharacterHeight), 0.f, MaxScrollOffset);

	return FReply::Handled();
}

FReply UPTerminalWidget::NativeOnKeyChar(const FGeometry & InGeometry, const FCharacterEvent & InCharEvent)
{	
	TCHAR c = InCharEvent.GetCharacter();

	if (c == TEXT('\b'))
	{
		if (!TextInputBuffer.IsEmpty())
		{
			TCHAR taken = TextInputBuffer[TextInputBuffer.Len() - 1];
			TextInputBuffer.RemoveAt(TextInputBuffer.Len() - 1, 1, true);
			if (EchoInputText)
			{
				if (taken == '&')
				{
					TextBuffer.RemoveAt(TextBuffer.Len() - 2, 2, true);
				}
				else
				{
					TextBuffer.RemoveAt(TextBuffer.Len() - 1, 1, true);
				}
				NewTextAdded = true;
			}
		}
	}
	else if (c == TEXT('\r'))
	{
		TextInputBuffer = TextInputBuffer.AppendChar(TEXT('\n'));
		TextBuffer = TextBuffer.AppendChar(TEXT('\n'));
		NewTextAdded = true;
		this->IsInputLineAvailable = true;
	}
	else 
	{
		TextInputBuffer = TextInputBuffer.AppendChar(c);
		if (EchoInputText)
		{
			if (c == '&')
			{
				TextBuffer.Append("&&");
			}
			else
			{
				TextBuffer = TextBuffer.AppendChar(c);
			}
			NewTextAdded = true;
		}
	}

	bCursorActive = true;
	cursorTime = 0;

	return FReply::Handled();
}

FReply UPTerminalWidget::NativeOnKeyDown(const FGeometry & InGeometry, const FKeyEvent & InKeyEvent)
{
	TCHAR c = InKeyEvent.GetCharacter();

	if (InKeyEvent.IsControlDown() || InKeyEvent.IsCommandDown())
	{
		if (c == TEXT('-'))
		{
			if (ZoomFactor - 0.25f < 1.f)
			{
				ZoomFactor = 1.f;
			}
			else
			{
				ZoomFactor -= 0.25f;
			}
			// Force the scroll offset to be recalculated.
			NewTextAdded = true;

			// Alert BP that we zoomed.
			float NewCharWidth = this->CharacterWidth*ZoomFactor;
			float NewCharHeight = this->CharacterHeight*ZoomFactor;
			this->TerminalZoomed(NewCharWidth, NewCharHeight);
			this->TerminalZoomedEvent.Broadcast(NewCharWidth, NewCharHeight);

			return FReply::Handled();
		}
		else if (c == TEXT('='))
		{
			ZoomFactor += 0.25f;
			// Force the scroll offset to be recalculated.
			NewTextAdded = true;
			
			// Alert BP that we zoomed.
			float NewCharWidth = this->CharacterWidth*ZoomFactor;
			float NewCharHeight = this->CharacterHeight*ZoomFactor;
			this->TerminalZoomed(NewCharWidth, NewCharHeight);
			this->TerminalZoomedEvent.Broadcast(NewCharWidth, NewCharHeight);

			return FReply::Handled();
		}
	}


	return FReply::Unhandled();
}

#if WITH_EDITOR
const FText UPTerminalWidget::GetPaletteCategory()
{
	return FText::FromString(TEXT("Peacegate OS"));
}
#endif

float UPTerminalWidget::GetLineHeight()
{
	// Get the size in pixels of the geometry so we know how big our widget is.
	FVector2D size = this->GetCachedGeometry().GetLocalSize();
	float char_x = 0.f; //Where will the character be rendered on the X coordinate?
	float char_y = 0.f; //The current text position on the Y axis, accounting for the scroll offset.
	FSlateFontInfo font = this->RegularTextFont;
	const UFont* FontPtr = Cast<UFont>(font.FontObject);
	//The width of a character.
	float char_w = CharacterWidth * ZoomFactor;
	//The height of a character.
	float char_h = CharacterHeight * ZoomFactor;
	TArray<TCHAR> arr = TextBuffer.GetCharArray();
	TCHAR Last = TEXT('\0');
	//How many elements are in the text array?
	int arrNum = arr.Num() - 1;

	for (int i = 0; i < arrNum; i++)
	{
		bool wasLiteral = false;
		if (this->SkipControlCode(TextBuffer, i, wasLiteral))
		{
			if (!wasLiteral)
			{
				continue;
			}
		}

		//Get the character.
		TCHAR c = arr[i];

		float kerning = 0.f;

		if (Last != TEXT('\0'))
		{
			kerning = FontPtr->GetCharKerning(Last, c) * ZoomFactor;
		}

		// If the character is a tab ('\t'), handle it.
		if (c == TEXT('\t'))
		{
			// Get remainder of char_x divided by 8 * char_w.
			float space = FMath::Fmod(char_x, char_w * 8);

			// Realistically, truncating the value won't make much of a difference.
			// But we'll store it as a float simply to make Unreal's life easier.
			// All we need to do is add this value to char_x.
			char_x += (char_w * 8) - space;

			Last = TEXT('\0');

			// Go to next char.
			continue;
		}

		//return to char 0 if we're a \r
		if (c == TEXT('\r'))
		{
			char_x = 0;
			kerning = 0;
			Last = TEXT('\0');
			continue;
		}

		if (c == TEXT('\0'))
			break; //Stop rendering at a null terminator because this is C++. I hate this.

		//Drop down to a new line.
		if (c == TEXT('\n'))
		{
			char_x = 0;
			char_y += char_h;
			Last = TEXT('\0');
			continue;
		}

		//Skip vertical tabs.
		if (c == TEXT('\v'))
			continue;

		UCommonUtils::MeasureChar(c, font, char_w, char_h);
		char_w = char_w * ZoomFactor;
		char_h = char_h * ZoomFactor;

		char_x += kerning;

		//If char_x + char_w is greater than our width, drop down to a new line.
		if (char_x + char_w > size.X)
		{
			kerning = 0;
			Last = TEXT('\0');
			char_x = 0;
			char_y += char_h;
		}

		Last = c;

		//Advance the x coordinate by a single char.
		char_x += char_w;
	}

	//One last time so the cursor isn't cut off and is an accurate representation of where new text will be placed.
	if (char_x + char_w > size.X)
	{
		char_x = 0;
		char_y += char_h;
	}

	return char_y + char_h;
}

FString UPTerminalWidget::NewLine()
{
	return TEXT("\n");
}

FString UPTerminalWidget::CarriageReturn()
{
	return TEXT("\r");
}

FString UPTerminalWidget::Tab()
{
	return TEXT("\t");
}

