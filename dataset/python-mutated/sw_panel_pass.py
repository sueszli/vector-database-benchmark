import os, sys, re, public
_title = '面板密码是否安全'
_version = 1.0
_ps = '检测面板帐号密码是否安全'
_level = 0
_date = '2020-08-04'
_ignore = os.path.exists('data/warning/ignore/sw_panel_pass.pl')
_tips = ['请到【设置】页面修改面板帐号密码', '注意：请不要使用过于简单的帐号密码，以免造成安全隐患', '推荐使用高安全强度的密码：分别包含数字、大小写、特殊字符混合，且长度不少于7位。']
_help = ''

def check_run():
    if False:
        for i in range(10):
            print('nop')
    '\n        @name 开始检测\n        @author hwliang<2020-08-04>\n        @return tuple (status<bool>,msg<string>)\n    '
    default_file = '/www/server/panel/default.pl'
    if not os.path.exists(default_file):
        return (True, '无风险')
    default_pass = public.readFile(default_file).strip()
    p1 = password_salt(public.md5(default_pass), uid=1)
    find = public.M('users').where('id=?', (1,)).field('username,password').find()
    if p1 == find['password']:
        return (False, '未修改面板默认密码，存在安全隐患')
    lower_pass_txt = '12123\nchina\ntest\ntest12\ntest11\ntest1\ntest2\ntest123\nbt.cn\nwww.bt.cn\nadmin\nroot\n12345\n123456\n123456789\n111111\nfrom91\n12345678\n123123\n5201314\n000000\n11111111\na123456\n163.com\nfill.com\n123321\n123123123\n00000000\n1314520\n7758521\n1234567\n666666\n123456a\n1234567890\nwoaini\na123456789\n888888\n88888888\n147258369\nqq123456\n654321\nzxcvbnm\nwoaini1314\n112233\n5211314\n123456abc\n520520\naaaaaa\n123654\n987654321\n123456789a\n12345\n7758258\n100200\n147258\n111222\nabc123456\n111222tianya\n121212\n1111111\nabc123\n110110\nadmin123\n789456\nq123456\n123456aa\naa123456\nasdasd\n999999\n123qwe\n789456123\n1111111111\n1314521\niloveyou\nqwerty\npassword\nqazwsx\n159357\n222222\nwoaini520\nwoaini123\n521521\nasd123\nqqqqqq\nqq1111\n1234\nqwe123\n111111111\n1qaz2wsx\nqwertyuiop\n5201314520\nasd123456\n159753\n31415926\nqweqwe\n555555\n333333\nwoaini521\nabcd1234\nASDFGHJKL\n123456qq\n11223344\n456123\n123000\n123698745\nwangyut2\n201314\nzxcvbnm123\nqazwsxedc\n1q2w3e4r\nz123456\n123abc\na123123\n12345678910\nasdfgh\n456789\nqwe123456\n321321\n123654789\n456852\n0000000000\nWOAIWOJIA\n741852963\n5845201314\naini1314\n0123456789\na321654\n123456123\n584520\n778899\n520520520\n7777777\nq123456789\n123789\nzzzzzz\nqweasdzxc\n5845211314\n123456q\nw123456\n12301230\nqq123456789\nwocaonima\nqq123123\na5201314\na12345678\nasdasdasd\na1234567\n147852\n110120\n135792468\nCAONIMA\n963852741\n3.1415926\n1234560\n101010\n7758520\n753951\n666888\nzxc123\n0000000\nzhang123\n987654\na111111\n1233211234567\n789789\n25257758\n7708801314520\nzzzxxx\n1111\n999999999\n1357924680\nyahoo.com.cn\n123456789q\n12341234\n5841314520\nzxc123456\nyangyang\n168168\n123123qaz\nabcd123456\n456456\n963852\nas123456\n741852\nxiaoxiao\n1230123\n555666\n000000000\n369369\n211314\n102030\naaa123456\nzxcvbn\n110110110\nbuzhidao\nqaz123\n123456.\nasdfasdf\n123456789.\nwoainima\n123456ASD\nwoshishui\n131421\n123321123\ndearbook\n1234qwer\nqaz123456\naaaaaaaa\n111222333\nqq5201314\n3344520\n147852369\n1q2w3e\nwindows\n123456987\nzz12369\nqweasd\nqiulaobai\n66666666\n12344321\nqwer1234\na12345\n7894561230\nqwqwqw\n777777\n110120119\n951753\nwmsxie123\n131420\n1314520520\n369258147\n321321321\n110119\nbeijing2008\n321654\na000000\n147896325\n12121212\n123456aaa\n521521521\n22222222\n888999\n123456789ABC\nabc123456789\n12345678900\n1q2w3e4r5t\n1234554321\nwww123456\nw123456789\n336699\nabcdefg\n709394\n258369\nz123456789\n314159\n584521\n12345678a\n7788521\n9876543210\n258258\n111111a\n87654321\n123asd\n5201314a\n134679\n135246\nhotmail.com\n123123a\n11112222\n131313\n100200300\n11111\n1234567899\n520530\n251314\nqq66666\nyahoo.cn\n123456qwe\nworinima\nsohu.com\nNULL\n518518\n123457\nq1w2e3r4\n721521\n123456789QQ\n584131421\nqw123456\n123456..\n0123456\n135790\n3344521\n980099\na1314520\n123456123456\nqazwsx123\nasdf1234\n444444\n123456z\n120120\nwang123456\n12345600\n7758521521\n12369874\nabcd123\na12369\nli123456\n1234567891\nwang123\n1234abcd\n147369\nzhangwei\nqqqqqqqq\n521125\n010203\n369258\n654123\nwoailaopo\nQAZQAZ\n121314\n1qazxsw2\nzxczxc\nl123456\n111000\njingjing\n0000\n1472583690\n25251325\nlangzi123\nwojiushiwo\n7895123\nwangjian\n123qweasd\n110120130\n1123581321\n142536\n584131420\naaa123\naaa111\nwoaiwoziji\n520123\n665544\nab123456\na123456a\nfuckyou\n99999999\n5203344\nqwertyui\n521314\n18881888\n584201314\nwoaini@\n7654321\n20082008\n520131\n124578\n852456\nnihaoma\n74108520\n232323\n55555555\nzx123456\nwwwwww\n119119\nweiwei\n13145200\nLOVE1314\n564335\n123456789123\nwo123456\n123520\n52013145201314\nloveyou\nwolf8637\n112358\n5201314123\nyuanyuan\nzhanglei\nzz123456\n1234567A\na11111\n000000a\n321654987\nxiaolong\n5841314521\nshmily\n520025\n159951\n77585210\ntiantian\n134679852\nQWASZX\n123456654321\n20080808\nzhangjian\n123465\n9958123\n159159\n5508386\nwangwei\n5205201314\nwoaini5201314\n888666\n52013141314\nqweqweqwe\n1122334455\n123456789z\n585858\n33333333\naa123123\nqwertyuiop123\nq111111\n9638527410\n911911\nqqq111\n5213344\nsunshine\nliu123456\nabcdef\nzhendeaini\n007007\n555888\nqq111111\njiushiaini\nmnbvcxz\nxiaoqiang\n445566\nnicholas\ndongdong\n123456abcd\n111qqq\naptx4869\n258456\nwobuzhidao\nqazxsw\n123789456\nzhang123456\n7215217758991\n1234567890123\n......\nhuang123\nmaomao\n222333\nwangyang\n123456789aa\n1.23457E+11\n1234566\n1230456\n1a2b3c4d\n13141314\na7758521\n123456zxc\n123456as\nforever\ns123456\n12348765\nxxxxxx\nasdf123\na1b2c3d4\n246810\n333666\nmingming\n000123\njiajia\n12qwaszx\nffffff\n112233445566\n77585211314\n520131400\naa123456789\nwpc000821\nWANGJING\nwoaini1314520\n&nbsp\n000111\nqq1314520\n1234512345\n147147\n123456qaz\nq123123\n123456ab\nxiaofeng\nwodemima\nshanshan\nw2w2w2\n666999\n123456w\n321456\nfeifei\ndragon\ncomputer\ndddddd\nzhangjie\nbaobao\nx123456\nq1w2e3\nchenchen\n12345679\n131452\ncaonima123\nasdf123456\ntangkai\n52013140\nlonglong\nssssss\nwww123\n1234568\nq1q1q1q1\nasdfghjkl123\n14789632\n123456711\nmichael\ntingting\nwoshishei\nasd123456789\n1314258\nsunliu66\nqwert12345\n235689\n565656\n1234569\nww123456\n1314159\n5211314521\n123456789w\n123123aa\n139.com@163.com\n111111q\nhao123456\n52tiance\n19830122\ny123456\n110119120\n1231230\nsj811212\n13579246810\n123.123\nsuperman\n789123\n12345qwert\n770880\njs77777\nzhangyang\n686868\n@163.com\nimzzhan\nxiaoyu\n7758521a\nabc12345\nnihao123\nwokaonima\nq11111\n623623623\n989898\n122333\n13800138000\nlaopowoaini\n787878\n123456l\na123123123\n198611\n332211\ntom.com\n212121\nwoaini123456\nwanglei\nyang123456\nzhangqiang\nzxcvbnm,./\nzhangyan\n181818\n234567\nstryker\n167669123\nlaopo520\n2597758\naa5201314\n139.com\n5201314.\n8888888888\n74107410\nzhanghao\n77777777\nzhangyu\nzzb19860526\nqwertyu\n5201314qq\n198612\nq5201314\n999888\n369852\n121121\n1122334\n123456789asd\n123zxc\na123321\nQWErtyUIO\n456456456\nqq000000\nm123456\nq1w2e3r4t5\nwoainilaopo\n123456789*\n131425\nliuchang\n85208520\nzhangjing\nc123456\nasdfghjk\nqq1234\nasdzxc\nhao123\n777888\n131131\nwoainia\nbeyond\nzhang520\n556688\n123456qw\nwangchao\nwoshiniba\n168888\n7758991\nwoshizhu\nainiyiwannian\nLAOpo521\nabcd123456789\nqwerasdf\n123456ok\nwoshinidie\nhuanhuan\n1hxboqg2s\nmeiyoumima\n456321\nQQQ123456\n1314\n898989\n123456798\npp.com@163.com\nmm123456\n123698741\na520520\nz321321\nasasas\nYANG123\n584211314\n1234561\n123456789+\nmiaomiao\n789789789\n7788520\nAAAAAAa\nh123456\n3838438\nl123456789\n198511\nABCDEFG123\nzhangjun\n123qaz\n198512\n2525775\n54545454\n789632145\n831213\n10101010\nxiaohe\n19861010\n10203\nwoshishen\n0987654321\nyj2009\nwangqiang\n198411\n1314520a\nxiaowei\n123456000\n123987\nlove520\ncaonimabi\nqwe123123\n010101\nqq666666\n789987\n10161215\nliangliang\nqwert123\n112112\nqianqian\n1a2b3c\n198410\nnuttertools\ngoodluck\nzhangxin\n18n28n24a5\nliuyang\n998877\nwoxiangni\n7788250\na147258369\nzhangliang\n16897168\n223344\n123123456\na1b2c3\nkiller\n321123\npp.com\nchen123456\nwangpeng\n753159\n775852100\n1478963\n1213141516\n369369369\n1236987\n123369\n12345a\nbugaosuni\n13145201314520\n110112\n123456...\nJIAOJIAO\n100100\n1314520123\n19841010\n7758521123\nshangxin\nwoshiwo\n12312300\nxingxing\nyingying\n1233210\n34416912\nqq12345\nqweasd123\nnishizhu\n19861020\nqwe123456789\n808080\n1310613106\n456789123\n44444444\n123123qq\n3141592653\n556677\nxx123456\njianjian\na1111111\n0.123456\n198610\nloveme\ntianshi\nwoxihuanni\n11235813\n252525\n225588\nlovelove\nmengmeng\n7758258520\nxiaoming\nshanghai\nhuyiming\n6543210\na7758258\n7788414\n123456789..\nJordan\nnishiwode\nZHUZHU\n1314woaini\nchenjian\n131415\nxy123456\n123456520\na00000\njiang123\nWOAIMAMA\nmonkey\n7418529630\nlingling\n987456321\nw5201314\nqwer123456\n198412\nasdasd123\nzzzzzzzz\n1q1q1q\n741741\n987456\n19851010\n2587758\n456654\nIloveyou1314\nq12345\nimissyou\ndaniel\naipai\n2222222\n0147258369\n123456789l\nq1234567\n963963\n123123123123\n125521\nwomendeai\nbaobei520\n19861015\n667788\n000000.\nzhangtao\nyy123456\nchen123\nnishishui\n789654\nliu123\n19861212\n1230\n19841020\nwangjun\nwangliang\nzhangpeng\nwoainimama\nzhangchao\n5201314q\n19841025\n123567\naaaa1111\n123456+\n134679258\n668899\n811009\nqaz123456789\n123456789qwe\n111112\n130130\n19861016\nwozhiaini\n198712\n123...\nabcde12345\nabcd12345\nwanggang\nllllll\n5121314\n456258\n125125\nqq7758521\n369963\n987987\n142857\npoiuytrewq\nqqq123\n323232\nbaobei\ng227w212\n962464\nmylove\np1a6s3m\n202020\n19491001\n963258\nhhhhhh\n2582587758\nwangfeng\ntiancai\n11111111111\nsummer\nwangwang\nasd123123\n19841024\nxinxin\n0.0.0.\n19861012\n19861210\n8888888\nzhanghui\nwenwen\n635241\nASDFGHJ\n19861023\n1234567890.\n888168\n19861120\ntianya\n123aaa\n111aaa\n123456789aaa\n8008208820\n123123q\nfootball\ndandan\nwww123456789\n19861026\nqingqing\n315315\n1111122222\n171204jg\n19861021\n5555555\nAS123456789\nqqqwww\n19861024\nyahoo.com\n19861225\n1qaz1qaz\n19871010\n1029384756\n123258\nzxcv123\n19861123\n1314520.\naidejiushini\n123qwe123\n198711\noperation\n19861025\nyu123456\n19851225\nwangshuai\n19841015\n520521\nwangyan\n19861011\n7007\n123456zz\n521000\n198311\n299792458\n112211\n******\n00000\nqwer123\n51201314\nqazwsxedcrfv\nLOVE5201314\n198312\n198510\n888888888\n1314521521\ninternet\nz123123\na147258\n696969\n1234321\n476730751\n5201314789\n012345\n19861022\nwelcome\naqwe518951\n19861121\nHUANGwei\n868686\nwanghao\nNIAIWOMA\nxiaojian\n19851120\n19851212\n100000\n19841022\nzhangbin\nshadow\nmmmmmm\n000...\n1357913579\n77585217758521\n19861216\n19841016\naz123456\nzxcv1234\n19841023\nwu123456\n163163\n2008520085\npppppp\n789654123\nEtnXtxSa65\n19851025\nwoaiwolaopo\nww111111\nwoaini110\n123455\n19841026\n19881010\nwww163com\n159357456\nfangfang\n19851015\n19861013\n19861220\n12312\n19861018\n19861028\na11111111\n19841018\n119911\nAI123456\n198211\n55555\nzhangkai\nwangxin\nxihuanni\n19871024\n19861218\n16899168\n1010110\nnimabi\n19861125\n52013143344\n131452000\n19871020\nfreedom\nbaobao520\nwinner\n123456m\n12312312\n'
    lower_pass = lower_pass_txt.split('\n')
    for lp in lower_pass:
        if not lp:
            continue
        if lp == find['username']:
            return (False, '当前面板用户名为：{} ，过于简单，存在安全隐患'.format(lp))
        p1 = password_salt(public.md5(lp), uid=1)
        if p1 == find['password']:
            return (False, '当前面板密码过于简单，存在安全隐患')
        lp = lp.upper()
        if lp == find['username']:
            return (False, '当前面板用户名为：{} ，过于简单，存在安全隐患'.format(lp))
        p1 = password_salt(public.md5(lp), uid=1)
        if p1 == find['password']:
            return (False, '当前面板密码过于简单，存在安全隐患')
    lower_rule = 'qwertyuiopasdfghjklzxcvbnm1234567890'
    for s in lower_rule:
        for i in range(12):
            if not i:
                continue
            lp = s * i
            if lp == find['username']:
                return (False, '当前面板用户名为：{} ，过于简单，存在安全隐患'.format(lp))
            p1 = password_salt(public.md5(lp), uid=1)
            if p1 == find['password']:
                return (False, '当前面板密码过于简单，存在安全隐患')
            lp = s.upper() * i
            if lp == find['username']:
                return (False, '当前面板用户名为：{} ，过于简单，存在安全隐患'.format(lp))
            p1 = password_salt(public.md5(lp), uid=1)
            if p1 == find['password']:
                return (False, '当前面板密码过于简单，存在安全隐患')
    if not is_strong_password(find['password']):
        return (False, '当前面板密码过于简单，存在安全隐患')
    return (True, '无风险')
salt = None

def password_salt(password, username=None, uid=None):
    if False:
        i = 10
        return i + 15
    '\n        @name 为指定密码加盐\n        @author hwliang<2020-07-08>\n        @param password string(被md5加密一次的密码)\n        @param username string(用户名) 可选\n        @param uid int(uid) 可选\n        @return string\n    '
    global salt
    if not salt:
        salt = public.M('users').where('id=?', (uid,)).getField('salt')
        if salt:
            salt = salt[0]
        else:
            salt = ''
    return public.md5(public.md5(password + '_bt.cn') + salt)

def is_strong_password(password):
    if False:
        i = 10
        return i + 15
    '判断密码复杂度是否安全\n\n    非弱口令标准：长度大于等于7，分别包含数字、小写、大写、特殊字符。\n    @password: 密码文本\n    @return: True/False\n    @author: linxiao<2020-9-19>\n    '
    if len(password) < 7:
        return False
    import re
    digit_reg = '[0-9]'
    lower_case_letters_reg = '[a-z]'
    upper_case_letters_reg = '[A-Z]'
    special_characters_reg = '((?=[\\x21-\\x7e]+)[^A-Za-z0-9])'
    regs = [digit_reg, lower_case_letters_reg, upper_case_letters_reg, special_characters_reg]
    grade = 0
    for reg in regs:
        if re.search(reg, password):
            grade += 1
    if grade == 4 or (grade == 3 and len(password) >= 9):
        return True
    return False