"""Test how SVG simple paths are drawn."""
from ...testing_utils import assert_no_logs

@assert_no_logs
def test_path_Hh(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        BBBBBBBB__\n        BBBBBBBB__\n        __________\n        RRRRRRRR__\n        RRRRRRRR__\n        __________\n        GGGGGGGG__\n        GGGGGGGG__\n        BBBBBBBB__\n        BBBBBBBB__\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 0 1 H 8 H 1"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 0 4 H 8 4"\n          stroke="red" stroke-width="2" fill="none"/>\n        <path d="M 0 7 h 8 h 0"\n          stroke="lime" stroke-width="2" fill="none"/>\n        <path d="M 0 9 h 8 0"\n          stroke="blue" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Vv(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        BB____GG__\n        BB____GG__\n        BB____GG__\n        BB____GG__\n        ___RR_____\n        ___RR_____\n        ___RR___BB\n        ___RR___BB\n        ___RR___BB\n        ___RR___BB\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 0 V 1 V 4"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 4 6 V 4 10"\n          stroke="red" stroke-width="2" fill="none"/>\n        <path d="M 7 0 v 0 v 4"\n          stroke="lime" stroke-width="2" fill="none"/>\n        <path d="M 9 6 v 0 4"\n          stroke="blue" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Ll(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        ______RR__\n        ______RR__\n        ______RR__\n        ___BB_RR__\n        ___BB_RR__\n        ___BB_RR__\n        ___BB_____\n        ___BB_____\n        ___BB_____\n        ___BB_____\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 4 3 L 4 10"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 7 0 l 0 6"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Zz(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        BBBBBBB___\n        BBBBBBB___\n        BB___BB___\n        BB___BB___\n        BBBBBBB___\n        BBBBBBB___\n        ____RRRRRR\n        ____RRRRRR\n        ____RR__RR\n        ____RRRRRR\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 1 H 6 V 5 H 1 Z"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 9 10 V 7 H 5 V 10 z"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Zz_fill(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        BBBBBBB___\n        BBBBBBB___\n        BBGGGBB___\n        BBGGGBB___\n        BBBBBBB___\n        BBBBBBB___\n        ____RRRRRR\n        ____RRRRRR\n        ____RRGGRR\n        ____RRRRRR\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 1 H 6 V 5 H 1 Z"\n          stroke="blue" stroke-width="2" fill="lime"/>\n        <path d="M 9 10 V 7 H 5 V 10 z"\n          stroke="red" stroke-width="2" fill="lime"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Cc(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        __________\n        __________\n        __________\n        __________\n        __BBB_____\n        __BBB_____\n        __________\n        __RRR_____\n        __RRR_____\n        __________\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 2 5 C 2 5 3 5 5 5"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 2 8 c 0 0 1 0 3 0"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Ss(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        __________\n        __________\n        __________\n        __________\n        __BBB_____\n        __BBB_____\n        __________\n        __RRR_____\n        __RRR_____\n        __________\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 2 5 S 3 5 5 5"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 2 8 s 1 0 3 0"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_CcSs(assert_pixels):
    if False:
        i = 10
        return i + 15
    assert_pixels('\n        __BBBBBB__\n        __BBBBBBB_\n        _____BBBB_\n        __RRRRRR__\n        __RRRRRRR_\n        _____RRRR_\n        __GGGGGG__\n        __GGGGGGG_\n        _____GGGG_\n        __BBBBBB__\n        __BBBBBBB_\n        _____BBBB_\n    ', '\n      <style>\n        @page { size: 10px 12px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 2 1 C 2 1 3 1 5 1 S 8 3 8 1"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 2 4 C 2 4 3 4 5 4 s 3 2 1 0"\n          stroke="red" stroke-width="2" fill="none"/>\n        <path d="M 2 7 c 0 0 1 0 3 0 S 8 9 8 7"\n          stroke="lime" stroke-width="2" fill="none"/>\n        <path d="M 2 10 c 0 0 1 0 3 0 s 3 2 1 0"\n          stroke="blue" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Qq(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        __________\n        __________\n        __________\n        __________\n        __BBBB____\n        __BBBB____\n        __________\n        __RRRR____\n        __RRRR____\n        __________\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 2 5 Q 4 5 6 5"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 2 8 q 2 0 4 0"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Tt(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        __________\n        __________\n        __________\n        __________\n        __BBBB____\n        __BBBB____\n        __________\n        __RRRR____\n        __RRRR____\n        __________\n    ', '\n      <style>\n        @page { size: 10px }\n        svg { display: block }\n      </style>\n      <svg width="10px" height="10px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 2 5 T 6 5"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 2 8 t 4 0"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_QqTt(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        _BBBB_______\n        BBBBBBB_____\n        BBBBBBBB__BB\n        BB__BBBBBBBB\n        _____BBBBBBB\n        _______BBBB_\n        _RRRR_______\n        RRRRRRR_____\n        RRRRRRRR__RR\n        RR__RRRRRRRR\n        _____RRRRRRR\n        _______RRRR_\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 0 3 Q 3 0 6 3 T 12 3"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 0 9 Q 3 6 6 9 t 6 0"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_QqTt2(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        _BBBB_______\n        BBBBBBB_____\n        BBBBBBBB__BB\n        BB__BBBBBBBB\n        _____BBBBBBB\n        _______BBBB_\n        _RRRR_______\n        RRRRRRR_____\n        RRRRRRRR__RR\n        RR__RRRRRRRR\n        _____RRRRRRR\n        _______RRRR_\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 0 3 q 3 -3 6 0 T 12 3"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 0 9 q 3 -3 6 0 t 6 0"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        __BBBB______\n        _BBBBB______\n        BBBBBB______\n        BBBB________\n        BBB_________\n        BBB____RRRR_\n        ______RRRRR_\n        _____RRRRRR_\n        _____RRRR___\n        _____RRR____\n        _____RRR____\n        ____________\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 6 A 5 5 0 0 1 6 1"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 6 11 a 5 5 0 0 1 5 -5"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa2(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ______GGGG__\n        ______GGGGG_\n        ______GGGGGG\n        ________GGGG\n        _________GGG\n        _________GGG\n        GGG______GGG\n        GGG______GGG\n        GGGG____GGGG\n        GGGGGGGGGGGG\n        _GGGGGGGGGG_\n        __GGGGGGGG__\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 6 A 5 5 0 1 0 6 1"\n          stroke="lime" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa3(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        ______GGGG__\n        ______GGGGG_\n        ______GGGGGG\n        ________GGGG\n        _________GGG\n        _________GGG\n        GGG______GGG\n        GGG______GGG\n        GGGG____GGGG\n        GGGGGGGGGGGG\n        _GGGGGGGGGG_\n        __GGGGGGGG__\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 6 a 5 5 0 1 0 5 -5"\n          stroke="lime" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa4(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        ____________\n        ____BBB_____\n        ____BBB_____\n        ___BBBB_____\n        _BBBBBB_____\n        _BBBBB______\n        _BBBB____RRR\n        _________RRR\n        ________RRRR\n        ______RRRRRR\n        ______RRRRR_\n        ______RRRR__\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 6 A 5 5 0 0 0 6 1"\n          stroke="blue" stroke-width="2" fill="none"/>\n        <path d="M 6 11 a 5 5 0 0 0 5 -5"\n          stroke="red" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa5(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        __BBBBBBBB__\n        _BBBBBBBBBB_\n        BBBBBBBBBBBB\n        BBBB____BBBB\n        BBB______BBB\n        BBB______BBB\n        BBB_________\n        BBB_________\n        BBBB________\n        BBBBBB______\n        _BBBBB______\n        __BBBB______\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 6 11 A 5 5 0 1 1 11 6"\n          stroke="blue" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa6(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        __BBBBBBBB__\n        _BBBBBBBBBB_\n        BBBBBBBBBBBB\n        BBBB____BBBB\n        BBB______BBB\n        BBB______BBB\n        BBB_________\n        BBB_________\n        BBBB________\n        BBBBBB______\n        _BBBBB______\n        __BBBB______\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 6 11 a 5 5 0 1 1 5 -5"\n          stroke="blue" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_Aa7(assert_pixels):
    if False:
        for i in range(10):
            print('nop')
    assert_pixels('\n        ____________\n        ____________\n        ____________\n        ____________\n        ____________\n        ____________\n        GGG______GGG\n        GGG______GGG\n        GGGG____GGGG\n        GGGGGGGGGGGG\n        _GGGGGGGGGG_\n        __GGGGGGGG__\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 6 A 5 5 0 0 0 11 6"\n          stroke="lime" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_wrong_point(assert_pixels):
    if False:
        while True:
            i = 10
    assert_pixels('\n        ____________\n        GG__________\n        GG__________\n        GG__________\n        GG__________\n        ____________\n        ____________\n        ____________\n        ____________\n        ____________\n        ____________\n        ____________\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <path d="M 1 1 L 1 5 L"\n          stroke="lime" stroke-width="2" fill="none"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_markers_l(assert_pixels):
    if False:
        print('Hello World!')
    assert_pixels('\n        _________zz_\n        _RR_____zzRz\n        _RRGGGGzzRzz\n        _RRGGGzzRzz_\n        _RR___zRzz__\n        ________zG__\n        _______RRRR_\n        _______RRRR_\n        ________GG__\n        _______RRRR_\n        _______RRRR_\n        ____________\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <marker id="line"\n          viewBox="0 0 1 2" refX="0.5" refY="1"\n          markerUnits="strokeWidth"\n          markerWidth="1" markerHeight="2"\n          orient="auto">\n          <rect x="0" y="0" width="1" height="2" fill="red" />\n        </marker>\n        <path d="M 2 3 l 7 0 l 0 4 l 0 3"\n          stroke="lime" stroke-width="2" fill="none" marker="url(\'#line\')"/>\n      </svg>\n    ')

@assert_no_logs
def test_path_markers_hv(assert_pixels):
    if False:
        return 10
    assert_pixels('\n        _________zz_\n        _RR_____zzRz\n        _RRGGGGzzRzz\n        _RRGGGzzRzz_\n        _RR___zRzz__\n        ________zG__\n        _______RRRR_\n        _______RRRR_\n        ________GG__\n        _______RRRR_\n        _______RRRR_\n        ____________\n    ', '\n      <style>\n        @page { size: 12px }\n        svg { display: block }\n      </style>\n      <svg width="12px" height="12px" xmlns="http://www.w3.org/2000/svg">\n        <marker id="line"\n          viewBox="0 0 1 2" refX="0.5" refY="1"\n          markerUnits="strokeWidth"\n          markerWidth="1" markerHeight="2"\n          orient="auto">\n          <rect x="0" y="0" width="1" height="2" fill="red" />\n        </marker>\n        <path d="M 2 3 h 7 v 4 v 3"\n          stroke="lime" stroke-width="2" fill="none" marker="url(#line)"/>\n      </svg>\n    ')