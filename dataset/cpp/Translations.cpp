/*  PCSX2 - PS2 Emulator for PCs
 *  Copyright (C) 2002-2023 PCSX2 Dev Team
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

#include "QtHost.h"

#include "common/Assertions.h"
#include "common/Console.h"
#include "common/StringUtil.h"

#include "pcsx2/ImGui/ImGuiManager.h"

#include "fmt/format.h"
#include "imgui.h"

#include <QtCore/QFile>
#include <QtCore/QTranslator>
#include <QtGui/QGuiApplication>
#include <QtWidgets/QMessageBox>

#include <vector>

#ifdef _WIN32
#include "common/RedtapeWindows.h"
#include <KnownFolders.h>
#include <ShlObj.h>
#endif

#if 0
// Qt internal strings we'd like to have translated
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "Services")
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "Hide %1")
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "Hide Others")
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "Show All")
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "Preferences...")
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "Quit %1")
QT_TRANSLATE_NOOP("MAC_APPLICATION_MENU", "About %1")
#endif

namespace QtHost
{
	struct GlyphInfo
	{
		const char* language;
		const char* windows_font_name;
		const char* linux_font_name;
		const char* mac_font_name;
		const char16_t* used_glyphs;
	};

	static std::string GetFontPath(const GlyphInfo* gi);
	static void UpdateGlyphRanges(const std::string_view& language);
	static const GlyphInfo* GetGlyphInfo(const std::string_view& language);

	static std::vector<ImWchar> s_glyph_ranges;

	static QLocale s_current_locale;
	static QCollator s_current_collator;
} // namespace QtHost

static std::vector<QTranslator*> s_translators;

static QString getSystemLanguage() {
	std::vector<std::pair<QString, QString>> available = QtHost::GetAvailableLanguageList();
	QString locale = QLocale::system().name();
	locale.replace('_', '-');
	// Can we find an exact match?
	for (const std::pair<QString, QString>& entry : available)
	{
		if (entry.second == locale)
			return locale;
	}
	// How about a partial match?
	QStringView lang = QStringView(locale);
	lang = lang.left(lang.indexOf('-'));
	for (const std::pair<QString, QString>& entry : available)
	{
		QStringView avail = QStringView(entry.second);
		avail = avail.left(avail.indexOf('-'));
		if (avail == lang) {
			Console.Warning("Couldn't find translation for system language %s, using %s instead",
			                locale.toStdString().c_str(), entry.second.toStdString().c_str());
			return entry.second;
		}
	}
	// No matches :(
	Console.Warning("Couldn't find translation for system language %s, using en instead", locale.toStdString().c_str());
	return QStringLiteral("en");
}

void QtHost::InstallTranslator()
{
	for (QTranslator* translator : s_translators)
	{
		qApp->removeTranslator(translator);
		translator->deleteLater();
	}
	s_translators.clear();

	QString language =
		QString::fromStdString(Host::GetBaseStringSettingValue("UI", "Language", GetDefaultLanguage()));
	if (language == QStringLiteral("system"))
		language = getSystemLanguage();

	QString qlanguage = language;
	qlanguage.replace('-', '_');
	s_current_locale = QLocale(qlanguage);
	s_current_collator = QCollator(s_current_locale);

	// Install the base qt translation first.
#ifdef __APPLE__
	const QString base_dir = QStringLiteral("%1/../Resources/translations").arg(qApp->applicationDirPath());
#else
	const QString base_dir = QStringLiteral("%1/translations").arg(qApp->applicationDirPath());
#endif

	// Qt base uses underscores instead of hyphens.
	const QString qt_language = QString(language).replace(QChar('-'), QChar('_'));
	QString base_path = QStringLiteral("%1/qt_%2.qm").arg(base_dir).arg(qt_language);
	bool has_base_ts = QFile::exists(base_path);
	if (!has_base_ts)
	{
		// Try without the country suffix.
		const int index = language.lastIndexOf('-');
		if (index > 0)
		{
			base_path = QStringLiteral("%1/qt_%2.qm").arg(base_dir).arg(language.left(index));
			has_base_ts = QFile::exists(base_path);
		}
	}

	if (has_base_ts)
	{
		QTranslator* base_translator = new QTranslator(qApp);
		if (!base_translator->load(base_path))
		{
			QMessageBox::warning(nullptr, QStringLiteral("Translation Error"),
				QStringLiteral("Failed to find load base translation file for '%1':\n%2").arg(language).arg(base_path));
			delete base_translator;
		}
		else
		{
			s_translators.push_back(base_translator);
			qApp->installTranslator(base_translator);
		}
	}

	const QString path = QStringLiteral("%1/pcsx2-qt_%3.qm").arg(base_dir).arg(language);
	QTranslator* translator = nullptr;
	if (QFile::exists(path))
	{
		translator = new QTranslator(qApp);
		if (translator->load(path))
		{
			Console.WriteLn(
				Color_StrongYellow, "Loaded translation file for language %s", language.toUtf8().constData());
		}
		else
		{
			QMessageBox::warning(nullptr, QStringLiteral("Translation Error"),
				QStringLiteral("Failed to load translation file for language '%1':\n%2").arg(language).arg(path));
			delete translator;
			translator = nullptr;
		}
	}
	else
	{
#ifdef PCSX2_DEVBUILD
		// For now, until we're sure this works on all platforms, we won't block users from starting if they're missing.
		QMessageBox::warning(nullptr, QStringLiteral("Translation Error"),
			QStringLiteral("Failed to find translation file for language '%1':\n%2").arg(language).arg(path));
#endif
	}

	if (translator)
	{
		qApp->installTranslator(translator);
		s_translators.push_back(translator);
	}

	UpdateGlyphRanges(language.toStdString());

	// Clear translation cache after installing translators, to prevent races.
	Host::ClearTranslationCache();
}

static std::string QtHost::GetFontPath(const GlyphInfo* gi)
{
	std::string font_path;

#ifdef _WIN32
	if (gi->windows_font_name)
	{
		PWSTR folder_path;
		if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Fonts, 0, nullptr, &folder_path)))
		{
			font_path = StringUtil::WideStringToUTF8String(folder_path);
			CoTaskMemFree(folder_path);
			font_path += "\\";
			font_path += gi->windows_font_name;
		}
		else
		{
			font_path = fmt::format("C:\\Windows\\Fonts\\{}", gi->windows_font_name);
		}
	}
#elif defined(__APPLE__)
	if (gi->mac_font_name)
		font_path = fmt::format("/System/Library/Fonts/{}", gi->mac_font_name);
#endif

	return font_path;
}

const char* QtHost::GetDefaultLanguage()
{
	return "system";
}

s32 Host::Internal::GetTranslatedStringImpl(
	const std::string_view& context, const std::string_view& msg, char* tbuf, size_t tbuf_space)
{
	// This is really awful. Thankfully we're caching the results...
	const std::string temp_context(context);
	const std::string temp_msg(msg);
	const QString translated_msg = qApp->translate(temp_context.c_str(), temp_msg.c_str());
	const QByteArray translated_utf8 = translated_msg.toUtf8();
	const size_t translated_size = translated_utf8.size();
	if (translated_size > tbuf_space)
		return -1;
	else if (translated_size > 0)
		std::memcpy(tbuf, translated_utf8.constData(), translated_size);

	return static_cast<s32>(translated_size);
}

std::vector<std::pair<QString, QString>> QtHost::GetAvailableLanguageList()
{
	return {
		{QCoreApplication::translate("InterfaceSettingsWidget", "System Language [Default]"), QStringLiteral("system")},
		{QStringLiteral("Afrikaans (af-ZA)"), QStringLiteral("af-ZA")},
		{QStringLiteral("عربي (ar-SA)"), QStringLiteral("ar-SA")},
		{QStringLiteral("Català (ca-ES)"), QStringLiteral("ca-ES")},
		{QStringLiteral("Čeština (cs-CZ)"), QStringLiteral("cs-CZ")},
		{QStringLiteral("Dansk (da-DK)"), QStringLiteral("da-DK")},
		{QStringLiteral("Deutsch (de-DE)"), QStringLiteral("de-DE")},
		{QStringLiteral("Ελληνικά (el-GR)"), QStringLiteral("el-GR")},
		{QStringLiteral("English (en)"), QStringLiteral("en")},
		{QStringLiteral("Español (Hispanoamérica) (es-419)"), QStringLiteral("es-419")},
		{QStringLiteral("Español (España) (es-ES)"), QStringLiteral("es-ES")},
		{QStringLiteral("فارسی (fa-IR)"), QStringLiteral("fa-IR")},
		{QStringLiteral("Suomi (fi-FI)"), QStringLiteral("fi-FI")},
		{QStringLiteral("Français (fr-FR)"), QStringLiteral("fr-FR")},
		{QStringLiteral("עִבְרִית (he-IL)"), QStringLiteral("he-IL")},
		{QStringLiteral("मानक हिन्दी (hi-IN)"), QStringLiteral("hi-IN")},
		{QStringLiteral("Magyar (hu-HU)"), QStringLiteral("hu-HU")},
		{QStringLiteral("hrvatski (hr-HR)"), QStringLiteral("hr-HR")},
		{QStringLiteral("Bahasa Indonesia (id-ID)"), QStringLiteral("id-ID")},
		{QStringLiteral("Italiano (it-IT)"), QStringLiteral("it-IT")},
		{QStringLiteral("日本語 (ja-JP)"), QStringLiteral("ja-JP")},
		{QStringLiteral("한국어 (ko-KR)"), QStringLiteral("ko-KR")},
		{QStringLiteral("Latvija (lv-LV)"), QStringLiteral("lv-LV")},
		{QStringLiteral("Lietuvių (lt-LT)"), QStringLiteral("lt-LT")},
		{QStringLiteral("Nederlands (nl-NL)"), QStringLiteral("nl-NL")},
		{QStringLiteral("Norsk (no-NO)"), QStringLiteral("no-NO")},
		{QStringLiteral("Polski (pl-PL)"), QStringLiteral("pl-PL")},
		{QStringLiteral("Português (Brasil) (pt-BR)"), QStringLiteral("pt-BR")},
		{QStringLiteral("Português (Portugal) (pt-PT)"), QStringLiteral("pt-PT")},
		{QStringLiteral("Limba română (ro-RO)"), QStringLiteral("ro-RO")},
		{QStringLiteral("Русский (ru-RU)"), QStringLiteral("ru-RU")},
		{QStringLiteral("Српски језик (sr-SP)"), QStringLiteral("sr-SP")},
		{QStringLiteral("Svenska (sv-SE)"), QStringLiteral("sv-SE")},
		{QStringLiteral("Türkçe (tr-TR)"), QStringLiteral("tr-TR")},
		{QStringLiteral("Українська мова (uk-UA)"), QStringLiteral("uk-UA")},
		{QStringLiteral("Tiếng Việt (vi-VN)"), QStringLiteral("vi-VN")},
		{QStringLiteral("简体中文 (zh-CN)"), QStringLiteral("zh-CN")},
		{QStringLiteral("繁體中文 (zh-TW)"), QStringLiteral("zh-TW")},
	};
}

static constexpr const ImWchar s_base_latin_range[] = {
	0x0020, 0x00FF, // Basic Latin + Latin Supplement
};
static constexpr const ImWchar s_central_european_ranges[] = {
	0x0100, 0x017F, // Central European diacritics
};

void QtHost::UpdateGlyphRanges(const std::string_view& language)
{
	const GlyphInfo* gi = GetGlyphInfo(language);

	std::string font_path;
	s_glyph_ranges.clear();

	// Base Latin range is always included.
	s_glyph_ranges.insert(s_glyph_ranges.begin(), std::begin(s_base_latin_range), std::end(s_base_latin_range));

	if (gi)
	{
		if (gi->used_glyphs)
		{
			const char16_t* ptr = gi->used_glyphs;
			while (*ptr != 0)
			{
				// Always should be in pairs.
				pxAssert(ptr[0] != 0 && ptr[1] != 0);
				s_glyph_ranges.push_back(*(ptr++));
				s_glyph_ranges.push_back(*(ptr++));
			}
		}

		font_path = GetFontPath(gi);
	}

	// If we don't have any specific glyph range, assume Central European, except if English, then keep the size down.
	if ((!gi || !gi->used_glyphs) && language != "en")
	{
		s_glyph_ranges.insert(
			s_glyph_ranges.begin(), std::begin(s_central_european_ranges), std::end(s_central_european_ranges));
	}

	// List terminator.
	s_glyph_ranges.push_back(0);
	s_glyph_ranges.push_back(0);

	ImGuiManager::SetFontPath(std::move(font_path));
	ImGuiManager::SetFontRange(s_glyph_ranges.data());
}

// clang-format off
static constexpr const char16_t s_cyrillic_ranges[] = {
	/* Cyrillic + Cyrillic Supplement */ 0x0400, 0x052F, /* Extended-A */ 0x2DE0, 0x2DFF, /* Extended-B */ 0xA640, 0xA69F, 0, 0
};
static constexpr const QtHost::GlyphInfo s_glyph_info[] = {
	// Cyrillic languages
	{ "ru-RU", nullptr, nullptr, nullptr, s_cyrillic_ranges },
	{ "sr-SP", nullptr, nullptr, nullptr, s_cyrillic_ranges },
	{ "uk-UA", nullptr, nullptr, nullptr, s_cyrillic_ranges },

	{
		"ja-JP", "msgothic.ttc", nullptr, "ヒラギノ角ゴシック W3.ttc", 
		// auto update by update_glyph_ranges.py with pcsx2-qt_ja-JP.ts
		u"​​……□□△△◯◯✕✕、。々々「』〜〜ああいいううええおせそそたちっぬのばびびぶぶへべほぼまやょろわわをんァイウチッツテニネロワワンン・ー一一上下不与世世両両並並中中乗乗了了予予事事互互交交今介他他付付代以仮仮件件任任伸伸似似位低体体何何作作使使例例供供係係保保信信修修個個倍倍借借値値停停側側傍傍備備像像優優元元先光入入全全公公共共具典内内再再冗冗処処出出分切列列初初判別利利制刷則則削削前前副副割割力力加加効効動動勧勧化化区区十十半半南単印印即即去去参参及及反収取受古古可台右右号号各各合合同名向向否否含含告告周周味味命命品品商商問問善善四四回回囲囲固固国国圧在地地均均垂垂型型域域基基報報場場境境増増壊壊声声変変外外多多大大太太央央失失奇奇奨奨好好妙妙妨妨始始子子字存安安完完定定実実容容寄寄密密対対射射尊尊小小少少尾尾展展履履岐岐崩崩左左巨巨己己帰帰常常幅幅平平序序度座延延式式引引弱弱張張強強当当形形彩彩影影役役待待後後得得御御復復微微心心必必応応忠忠念念急急性性悪悪情情想想意意感感態態慣慣成成戻戻所所手手承承投投択択抱抱押押拒拒招招拡拡拳拳持持指指振振挿挿捗捗排排接接推推描提換換揮揮援援揺揺損損撮撮操操支支改改放放敗敗数数整整文文料料断断新新方方既既日日早早明明昨昨時時曲曲更更書書替最有有望望期期未本条条析析果果格格桁桁案案械械検検極極楽楽概概構構標標権権橙橙機機欄欄欠次止正歴歴殊残毎毎比比水水求求汎汎決決況況法法波波注注浪浪浮浮消消深深済済減減渡渡港港湾湾満満源源準準滑滑点点無無照照牲牲特特犠犠状状獲獲率率現現理理璧璧環環生生用用由由申申画画番番異異発登白白的的目目直直相相省省知知短短破破確確示示禁禁秒秒移移程程種種稿稿空空立立端端競競第第算算管管範範精精索索細細終終組組経経結結絞絞統統続続維維緑緒線線編編縦縦縮縮績績繰繰置置義義者者肢肢能能自自致致般般良良色色荷荷落落蔵蔵行行表表衰衰装装補補製製複複要要見見規規視視覚覚覧覧観観角角解解言言計計記記設設許許訳訳証証試試該詳認認語語誤誤説読調調識識警警象象負負責責費貼資資質質赤赤起起超超足足跡跡転転軸軸軽軽辞辞込込近近返返追追送送逆逆途途通通速速連連進進遅遅遊遊過過適適選選避避部部配配重量鉄鉄録録長長閉閉開開間間関関閾閾防防降降限限除除陽陽隅隅際際集集離難電電青青非非面面韓韓音音響響頂頂順順領領頻頼題題類類飾飾香香駄駄高高鮮鮮黄黄！！＞？～～"
	},
	{
		"ko-KR", "malgun.ttf", nullptr, "AppleSDGothicNeo.ttc",
		// auto update by update_glyph_ranges.py with pcsx2-qt_ko-KR.ts
		u"“”……←↓□□△△◯◯✕✕んん茶茶가각간간갈갈감값강강같같개개거거건건걸걸검겁것것게게겟겠겨격견견결결경경계계고곡곱곱공공과과관관교교구국군군권권규규균균그극근근글글금급긍긍기기긴긴길길김깁깃깃깅깅깊깊까까깝깝깨깨꺼꺼께께꾸꾸꿉꿉끄끄끊끊끔끔끝끝나나난난날날남남낭낮내내낸낸냅냅너너널널넣네넬넬넷넷노녹높놓누누눈눈눌눌뉴뉴느느는는늘늘능능니니닌닌님닙닛닛다다단단닫달담담당당대대댑댑더더덜덜덤덤덩덩덮덮데덱덴덴델델도독돌돌동동됐됐되되된된될될됨됩두두둘둘뒤뒤듀듀듈듈드득든든들들듬듭등등디디딩딩때때떠떠떤떤떨떨또또뛰뛰뛸뛸뜁뜁뜨뜨라락란란랍랍랑랑래랙랜랜램램랫랫량량러럭런런럴럴럼럽렀렀렇렉렌렌렛렛려력련련렬렬렸령로록롤롤롭롭롯롯료료루루룸룸류류률률르르른른를를름릅리릭린린릴릴림립릿릿링링마막만만많많말말맞맞매매맨맨맵맵맺맺머머먼먼멀멀멈멈멋멋메메며며면면명명모목몬몬못못몽몽무무문문물물뮬뮬므므미믹민민밀밀밉밉밍밍및및바바반반받밝밥밥방방배백밴밴뱃뱃버버번번벌벌범법벗벗베벡벤벤벨벨벼벽변변별별병병보복본본볼볼부부분분불불붙붙뷰뷰브브블블비빅빈빈빌빌빙빙빛빛빠빠빨빨사삭산산살살삼삽상상새색샘샘생생샷샷서서선선설설성성세섹센센셀셀셈셉셋셋션션셰셰소속손손솔솔송송쇄쇄쇠쇠쇼쇼수숙순순숨숨숫숫쉬쉬슈슈스스슬슬습습시식신신실실심십싱싱싶싶쌍쌍쓰쓰쓸쓸씬씬아아안안않않알알암압았앙애액앤앤앨앨앵앵야약양양어어언언얻얼업없었었에에엔엔여역연연열열영영예예오오온온올올옵옵와와완완왑왑왔왔외외왼왼요요용용우우운운울울움웁웃웃워워원원월월웠웠웨웨웹웹위위유유율율으으은은을을음음응응의의이이인인일읽임입있있자작잘잘잠잡장장재재잿잿저적전전절절점접정정제제젠젠젤젤젯젯져져조족존존종종좋좌죄죄주주준준줄줄줍줍중중즈즉즐즐즘즘증증지직진진질질짐집짝짝짧짧째째쪽쪽찌찍차착참참창찾채책챕챕처척천천철철첨첩첫첫청청체체쳐쳐초초총총촬촬최최추축출출춤춥충충춰춰취취츠측치칙칠칠침칩칭칭카카칸칸칼칼캐캐캔캔캘캘캠캡커커컨컨컬컬컴컴케케켓켓켜켜켬켬코코콘콘콜콜콩콩쿳쿳퀀퀀퀴퀴큐큐크크큰큰클클큽큽키키킨킨킬킬킵킵킹킹타타탈탈탐탐태택탠탠탬탭터터턴턴털털텀텀테텍텐텐텔텔템템토토톨톨톱톱통통투투트특튼튼틀틀티틱팀팀팅팅파파팔팔팝팝패패팻팻퍼퍼페펙편편평평포포폴폴폼폼표표푸푸풀풀품품퓨퓨프프픈픈플플피픽핀핀필필핑핑하하한한할할함합핫핫항항해핵했행향향허허헌헌헤헤현현형형호혹혼혼홈홈홍홍화확환환활활황황회획횟횟횡횡효효후후훨훨휠휠휴휴흐흐흔흔희희히히힌힌"
	},
	{
		"zh-CN", "msyh.ttc", nullptr, "Hiragino Sans GB.ttc",
		// auto update by update_glyph_ranges.py with pcsx2-qt_zh-CN.ts
		u"‘’□□△△○○、。一丁三下不与且且世世丢丢两严个个中中串串为主么义之之乎乎乐乐乘乘也也了了事二于于互互五五亚些交交产产享享亮亮亲亲人人什什仅仅今介仍从他他代以们们件价任任份份仿仿伍伍休休优优会会传传估估伸伸似似但但位住佑佑体体何何余余作作你你佳佳使使例例供供依依侧侧便便俄俄保保信信修修倍倍倒倒借借值值倾倾假假偏偏做做停停储储像像儿儿允允元元充充先光克克免免入入全全公六兰共关关兵兹兼兼内内册再写写冲决况况冻冻净净准准减减几凡凭凭出击函函刀刀刃刃分切列列则创初初删删利利别别到到制刷刹刹前前剪剪副副力力功务动助势势勾勿包包化北匹区十十升升半半协协单单南南占卡卫卫印印即即卸卸历历压压原原去去参参叉及双反发发取变叠叠口古另另只只可台史右号司各各合吉同后向向吗吗否否含含启启呈呈告告员员味味命命和和哈哈响响哪哪唤唤商商善善器器四四回回因因团团围围固固国图圈圈在在地地场场址址均均坏坐块块坛坛垂垂型型域域基基堆堆堪堪塔塔填填增增士士声声处处备备复复外外多多够够大大天太失失头头夹夹奇奇奏奏套套奥奥女女奶奶好好如如始始威威娜娜娱娱婴婴媒媒子子字存它它安安完完宏宏官官定定宝实害害家家容容宽宿寄寄密密富富寸对导导封封射射将将小小少少尔尔尘尘尚尚尝尝就就尺尺尼尾局局层层屏屏展展属属崩崩工左巨巨巫巫差差己已巴巴希希带帧帮帮常常幅幅幕幕干平并并幸幸幻幻序序库底度度延延建建开开异弃弊弊式式引引张张弹强归当录录形形彩彩影影径待很很律律得得循循微微德德心心必忆志志忙忙快快忽忽态态性性怪怪恢恢息息恶恶您您悬悬情情惑惑惩惩惯惯想想愉愉意意感感慢慢戏我或或战战截截戳戳户户所扁扇扇手手才才打打托托执执扩扩扫扬扳扳找找技技抑抑抖抗护护拆拆拉拉拒拒拟拟拥拦择择括括拳拳持挂指指按按挑挑挡挡挪挪振振捕捕损损换换据据捷捷掌掌排排接接控掩描提插插握握搜搜摇摇摔摔摘摘撕撕撤撤播播操擎支支收收改改放放故故效效敌敌敏敏散散数数整整文文斜斜断断斯新方方旁旁旋旋无无日日旧旧时时明明易易星映昨昨是是显显晕晕普普晰晰暂暂暗暗曲曲更更替最有有服服望望期期未未本本术术机机杂权杆杆束束条条来来板板极极果果枪枪柄柄某某染染查查栅栅标栈栏栏校校样根格格框框案案档档桥桥梦梦检检棕棕榜榜槽槽模模横横橙橙次欢欧欧止步死死殊殊段段每每比比毫毫水水求求汇汇污污沙沙没没油油法法波波注注泻泻洋洋洲洲活活流流浅浅测测浏浏浪浪浮浮海海消消淡淡深深混混添添清清港港渲渲游游湖湖湾湾溃溃源源溢溢滑滑满满滤滤演演潜潜澳澳激激火火灯灰灵灵点点烈烈热热焦焦然然煞煞照照爆爆片版牌牌牙牙物物特特状状独独狼狼猎猎猩猩率率王王玛玛玩玩环现班班理理瑞瑞甚甚生生用用由由电电画画畅畅界界留留略略疤疤登登白百的的皇皇监监盖盘目目直直相相省省看看真眠着着瞄瞄矢矢知知短短石石码码破破础础硬硬确确碌碌碎碎磁磁示示神神禁禁离离种种秒秒积称移移程程稍稍稳稳空空突突窗窗立立站站竞竞端端符符第第等等答答筛筛签签简简算算管管类类粉粉粘粘精精糊糊系系素素索索紫紫繁繁纠纠红红级级纳纳纵纵纹纹线线组组细终经经绑绑结结绕绕绘给络绝统统继继绪绪续续维维绿缀缓缓编编缘缘缩缩缺缺网网罗罗罚罚置置美美翻翻考考者者而而耗耗耳耳联联肩肩胖胖能能腊腊自自至致舍舍航航良良色色节节芬芬若若英英范范荐荐荒荒荷荷莱莱获获菜菜萄萄著著葡葡蓝蓝藏藏虑虑虚虚融融行行街街衡衡补补表表被被裁裂装装西西要要覆覆见观规规视视览觉角角解解触触言言警警计计认认让让议议记记许许论论设访证证识识诊诊译译试试话话询询该详语语误误说说请诸读读调调谍谍象象豹豹负负败账质质贴贴费费赖赖赛赛赫赫起起超超越越足足跃跃跟跟跨跨路路跳跳踏踏踪踪身身车车轨轨转转轮软轴轴轻轻载载较较辅辅辑辑输输辨辨边边达达过过迎迎运近返返还这进进连迟述述追追退适逆逆选选逐逐递递通通速造遇遇道道避避那那邻邻部部都都配配醒醒采采释释里量金金针针钟钟钮钮铁铁铺铺链链销锁锐锐错错键锯镜镜长长门门闭问闲闲间间阅阅队队防防阴阵附陆降降限限除除险险随隐隔隔隙隙障障雄雄集集零零雾雾需需震震静静非非靠靠面面韩韩音音页顶项须顿顿预预频频题题颜额颠颠风风饱饱馈馈首首香香马马驱驱驾驾验验骑骑骤骤高高鬼鬼魂魂魔魔麦麦黄黄黑黑默默鼓鼓鼠鼠齐齐齿齿，，：：？？"
	},
	{
		"zh-TW", "msyh.ttc", nullptr, "Hiragino Sans GB.ttc",
		// auto update by update_glyph_ranges.py with pcsx2-qt_zh-TW.ts
		u"□□△△○○、。『』一丁三下不不且且世世丟丟並並中中串串主主之之乎乎乘乙也也乾乾了了事二于于互互五五些些亞亞交交享享亮亮人人什什今介仍仍他他代以件件任任份份伍伍休休估估伸伸伺伺似似但佇位住佑佑佔何作作你你佳佳併併使使來來例例供供依依便便係係俄俄保保修修個個倍倍們倒借借值值假假偏偏做做停停側偵備備傳傳傾傾僅僅像像價價儘儘優優儲儲允允元元充充先光克克免免兒兒入入內兩公六共共兵典冊冊再再凍凍凡凡出出函函刀刀刃刃分切列列初初別別利刪到到制制則則前剎剛剛剪剪副副力力功加助助動動務務勢勢勾勿包包化北匯匯匹匹區十升升半半協協南南卡卡印印即即原原去去參參叉及反反取受口古另另只叫可可史右司司各各合吉同后向向否否含含呈呈告告味味呼命和和哈哈員員哪哪商商問問啓啓啟啟善善喚喚單單嗎嗎嘉嘉嘗嘗器器嚮嚮嚴嚴四四回回因因固固圈圈國國圍圍圖圖團團在在地地址址均均垂垂型型域埠執執基基堆堆堪堪場場塊塊塔塔填填塵塵增增壇壇壓壓壞壞士士外外多多夠夠夢夢大大天太失失夾夾奇奇奏奏套套奧奧女女奶奶好好如如始始威威娛娜媒媒嬰嬰子子字存它它安安完完宏宏官官定定害害家家容容宿宿密密富富實實寫寬寶寶寸寸封封射射將專對小少少尚尚就就尺尺尼尼尾尾屏屏展展層層屬屬崩崩工左巨巨巫巫差差己已巴巴希希帶帶常常幀幀幅幅幕幕幫幫平平幸幸幻幻幾幾序序底底度座庫庫延延建建弊弊式式引引張張強強彈彈彙彙形形彩彩影影待待很很律後徑徑得得從從復循微微德德心心必必忙忙快快忽忽性性怪怪恢恢息息您您情情惑惑惡惡想想愉愉意意感感態態慢慣慮慮憑憑憶憶應應懲懲懸懸成我或或截截戰戰戲戳戶戶所扁扇扇手手才才打打扳扳找找技技抑抑抖抗拆拆拉拉拒拒括括拳拳持持指指按按挑挑挪挪振振捕捕捨捨捷捷掃掃掌掌排排掛掛採採接接控掩描提插插揚換握握援援損損搖搖搜搜摔摔摘摘撕撕撤撤播播擁擁擇擇擊擋操擎據據擬擬擴擴擷擷攔攔支支收收改改放放故故效效敏敏敗敗整敵數數文文料料斜斜斯新斷斷方方於於旁旁旋旋日日明明易易星映昨昨是是時時普普晰晰暈暈暗暗暢暢暫暫曲曲更更替最會會有有服服望望期期未未本本束束板板析析果果柄柄某某染染查查柵柵校校核根格格框框案案桿桿條條棄棄棕棕極極榜榜槍槍槽槽樂樂標標模模樣樣橋橋橙橙機機橫橫檔檔檢檢檯檯欄欄權權次次歐歐歡步歷歸死死殊殊段段每每比比毫毫水水求求污污決決沒沒沙沙油油況況法法波波注注洋洋洲洲活活流流浪浪浮浮海海消消淡淡淨淨深深混混淺淺清清減減測測港港渲渲游游湊湊湖湖源源準準溢溢滑滑滿滿演演潛潛潰潰澳澳激激濾濾瀉瀉瀏瀏灣灣火火灰灰為為烈烈無無焦焦然然煞煞照照熱熱燈燈爆爆爲爲爾爾片版牌牌牙牙物物特特狀狀狼狼猩猩獨獨獲獲獵獵率率王王玩玩班班現現理理瑞瑞瑪瑪環環甚甚生生產產用用由由界界留留略略畫畫異異當當疊疊疤疤登百的的皇皇盜盜盡盡監盤目目直直相相省省看看真眠瞄瞄知知短短石石破破硬硬碌碌碎碎碟碟確確碼碼磁磁礎礎示示神神禁禁秒秒移移程程稍稍種種稱稱積積穩穩空空突突窗窗立立站站端端競競符符第第等等答答算算管管節節範範篩篩簡簡簿簿籤籤粉粉精精糊糊系系糾糾紅紅紋紋納納級級素素索索紫紫細細終終組組結結絕絕給給統統經經綠綠維維網網綴綴緒緒線線緣緣編緩縮縮縱縱繁繁織織繞繞繪繫繼繼續續缺缺置置罰罰羅羅美美義義翻翻考考者者而而耗耗耳耳聯聯聲聲肩肩胖胖能能臘臘自自至致臺臺與與舊舊舍舍航航良良色色芬芬芽芽若若英英茲茲荒荒荷荷菜菜萄萄萊萊著著葡葡蓋蓋薦薦藍藍藏藏蘭蘭處處虛虛號號融融螢螢行行術術街街衛衛衝衝衡衡表表被被裁裂補裝裡裡製製複複西西要要覆覆見見規規視視親親覺覺覽覽觀觀角角解解觸觸言言計計訊訊託記訪訪設設許許診註詢詢試試話詳誌認語語誤誤說說調調請請論論諜諜諸諸證證識識警警譯議護護讀讀變變讓讓象象豹豹負負費貼資資賓賓質質賬賬賴賴賽賽赫赫起起超超越越足足跟跟跨跨路路跳跳踏踏蹤蹤躍躍身身車車軌軌軟軟軸軸較較載載輔輕輪輪輯輯輸輸轉轉迎迎近近返返述述迴迴追追退送逆逆逐逐這通速造連連進進遇遇遊運過過道達遞遞適適遲遲選選避避還還邊邊那那部部都都鄰鄰配配醒醒重量金金針針鈕鈕銳銳銷銷鋪鋪鋸鋸錄錄錯錯鍵鍵鎖鎖鏈鏈鏡鏡鐘鐘鐵鐵長長門門閉閉開開閑閑間間閘閘閱閱關關防防附附降降限限陣除陰陰陸陸隊隊隔隔隙隙際障隨隨險險隱隱雄雄集集雙雙雜雜離離零零電電需需震震霧霧靈靈靜靜非非靠靠面面韓韓音音響響頁頂項順須須預預頓頓頭頭頻頻題額顏顏顛顛類類顯顯風風飽飽餘餘饋饋首首香香馬馬駕駕騎騎驅驅驗驗驟驟體體高高髮髮鬼鬼魂魂魔魔麥麥麼麼黃黃黑黑點點鼓鼓鼠鼠齊齊齒齒，，：：？？"
	},
};
// clang-format on

const QtHost::GlyphInfo* QtHost::GetGlyphInfo(const std::string_view& language)
{
	for (const GlyphInfo& it : s_glyph_info)
	{
		if (language == it.language)
			return &it;
	}

	return nullptr;
}

int QtHost::LocaleSensitiveCompare(QStringView lhs, QStringView rhs)
{
	return s_current_collator.compare(lhs, rhs);
}
