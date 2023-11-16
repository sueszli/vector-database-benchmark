class H2a:
    H2a_table = {'ぁ': 'a', 'あ': 'a', 'ぃ': 'i', 'い': 'i', 'ぅ': 'u', 'う': 'u', 'う゛': 'vu', 'う゛ぁ': 'va', 'う゛ぃ': 'vi', 'う゛ぇ': 've', 'う゛ぉ': 'vo', 'ぇ': 'e', 'え': 'e', 'ぉ': 'o', 'お': 'o', 'か': 'ka', 'が': 'ga', 'き': 'ki', 'きぁ': 'kya', 'きぅ': 'kyu', 'きぉ': 'kyo', 'ぎ': 'gi', 'ぐゃ': 'gya', 'ぎぅ': 'gyu', 'ぎょ': 'gyo', 'く': 'ku', 'ぐ': 'gu', 'け': 'ke', 'げ': 'ge', 'こ': 'ko', 'ご': 'go', 'さ': 'sa', 'ざ': 'za', 'し': 'shi', 'しゃ': 'sha', 'しゅ': 'shu', 'しょ': 'sho', 'じ': 'ji', 'じゃ': 'ja', 'じゅ': 'ju', 'じょ': 'jo', 'す': 'su', 'ず': 'zu', 'せ': 'se', 'ぜ': 'ze', 'そ': 'so', 'ぞ': 'zo', 'た': 'ta', 'だ': 'da', 'ち': 'chi', 'ちぇ': 'che', 'ちゃ': 'cha', 'ちゅ': 'chu', 'ちょ': 'cho', 'ぢ': 'ji', 'ぢゃ': 'ja', 'ぢゅ': 'ju', 'ぢょ': 'jo', 'っ': 'tsu', 'っう゛': 'vvu', 'っう゛ぁ': 'vva', 'っう゛ぃ': 'vvi', 'っう゛ぇ': 'vve', 'っう゛ぉ': 'vvo', 'っか': 'kka', 'っが': 'gga', 'っき': 'kki', 'っきゃ': 'kkya', 'っきゅ': 'kkyu', 'っきょ': 'kkyo', 'っぎ': 'ggi', 'っぎゃ': 'ggya', 'っぎゅ': 'ggyu', 'っぎょ': 'ggyo', 'っく': 'kku', 'っぐ': 'ggu', 'っけ': 'kke', 'っげ': 'gge', 'っこ': 'kko', 'っご': 'ggo', 'っさ': 'ssa', 'っざ': 'zza', 'っし': 'sshi', 'っしゃ': 'ssha', 'っしゅ': 'sshu', 'っしょ': 'ssho', 'っじ': 'jji', 'っじゃ': 'jja', 'っじゅ': 'jju', 'っじょ': 'jjo', 'っす': 'ssu', 'っず': 'zzu', 'っせ': 'sse', 'っぞ': 'zze', 'っそ': 'sso', 'っぜ': 'zzo', 'った': 'tta', 'っだ': 'dda', 'っち': 'tchi', 'っちゃ': 'tcha', 'っちゅ': 'tchu', 'っちょ': 'tcho', 'っぢ': 'jji', 'っぢゃ': 'jjya', 'っぢゅ': 'jjyu', 'っぢょ': 'jjyo', 'っつ': 'ttsu', 'っづ': 'zzu', 'って': 'tte', 'っで': 'dde', 'っと': 'tto', 'っど': 'ddo', 'っは': 'hha', 'っば': 'bba', 'っぱ': 'ppa', 'っひ': 'hhi', 'っひゃ': 'hhya', 'っひゅ': 'hhyu', 'っひょ': 'hhyo', 'っび': 'bbi', 'っびゃ': 'bbya', 'っびゅ': 'bbyu', 'っびょ': 'bbyo', 'っぴ': 'ppi', 'っぴゃ': 'ppya', 'っぴゅ': 'ppyu', 'っぴょ': 'ppyo', 'っふ': 'ffu', 'っふぁ': 'ffa', 'っふぃ': 'ffi', 'っふぇ': 'ffe', 'っふぉ': 'ffo', 'っぶ': 'bbu', 'っぷ': 'ppu', 'っへ': 'hhe', 'っべ': 'bbe', 'っぺ': 'ppe', 'っほ': 'hho', 'っぼ': 'bbo', 'っぽ': 'ppo', 'っや': 'yya', 'っゆ': 'yyu', 'っよ': 'yyo', 'っら': 'rra', 'っり': 'rri', 'っりゃ': 'rrya', 'っりゅ': 'rryu', 'っりょ': 'rryo', 'っる': 'rru', 'っれ': 'rre', 'っろ': 'rro', 'つ': 'tsu', 'づ': 'zu', 'て': 'te', 'で': 'de', 'でぃ': 'di', 'と': 'to', 'ど': 'do', 'な': 'na', 'に': 'ni', 'にゃ': 'nya', 'にゅ': 'nyu', 'にょ': 'nyo', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no', 'は': 'ha', 'ば': 'ba', 'ぱ': 'pa', 'ひ': 'hi', 'ひゃ': 'hya', 'ひゅ': 'hyu', 'ひょ': 'hyo', 'び': 'bi', 'びゃ': 'bya', 'びゅ': 'byu', 'びょ': 'byo', 'ぴ': 'pi', 'ぴゃ': 'pya', 'ぴゅ': 'pyu', 'ぴょ': 'pyo', 'ふ': 'fu', 'ふぁ': 'fa', 'ふぃ': 'fi', 'ふぇ': 'fe', 'ふぉ': 'fo', 'ぶ': 'bu', 'ぷ': 'pu', 'へ': 'he', 'べ': 'be', 'ぺ': 'pe', 'ほ': 'ho', 'ぼ': 'bo', 'ぽ': 'po', 'ま': 'ma', 'み': 'mi', 'みゃ': 'mya', 'みゅ': 'myu', 'みょ': 'myo', 'む': 'mu', 'め': 'me', 'も': 'mo', 'ゃ': 'ya', 'や': 'ya', 'ゅ': 'yu', 'ゆ': 'yu', 'ょ': 'yo', 'よ': 'yo', 'ら': 'ra', 'り': 'ri', 'りゃ': 'rya', 'りゅ': 'ryu', 'りょ': 'ryo', 'る': 'ru', 'れ': 're', 'ろ': 'ro', 'ゎ': 'wa', 'わ': 'wa', 'ゐ': 'i', 'ゑ': 'e', 'を': 'wo', 'ん': 'n', 'んあ': "n'a", 'んい': "n'i", 'んう': "n'u", 'んえ': "n'e", 'んお': "n'o"}
    _shared_state = {}

    def __new__(cls, *p, **k):
        if False:
            print('Hello World!')
        self = object.__new__(cls, *p, **k)
        self.__dict__ = cls._shared_state
        return self

    def isHiragana(self, char):
        if False:
            print('Hello World!')
        return 12352 < ord(char) and ord(char) < 12436

    def convert(self, text):
        if False:
            for i in range(10):
                print('nop')
        Hstr = ''
        max_len = -1
        r = min(4, len(text) + 1)
        for x in range(r):
            if text[:x] in self.H2a_table:
                if max_len < x:
                    max_len = x
                    Hstr = self.H2a_table[text[:x]]
        return (Hstr, max_len)