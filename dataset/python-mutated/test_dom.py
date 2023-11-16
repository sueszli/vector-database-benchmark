from unittest import mock
import pytest
from pyscript import document, when
from pyweb import pydom

class TestDocument:

    def test__element(self):
        if False:
            for i in range(10):
                print('nop')
        assert pydom._js == document

    def test_no_parent(self):
        if False:
            i = 10
            return i + 15
        assert pydom.parent is None

    def test_create_element(self):
        if False:
            while True:
                i = 10
        new_el = pydom.create('div')
        assert isinstance(new_el, pydom.BaseElement)
        assert new_el._js.tagName == 'DIV'
        assert new_el.parent == None

def test_getitem_by_id():
    if False:
        for i in range(10):
            print('nop')
    id_ = 'test_id_selector'
    txt = 'You found test_id_selector'
    selector = f'#{id_}'
    result = pydom[selector]
    div = result[0]
    assert document.querySelector(selector).innerHTML == div.html == txt
    assert isinstance(div, pydom.BaseElement)
    assert isinstance(result, pydom.ElementCollection)

def test_getitem_by_class():
    if False:
        while True:
            i = 10
    ids = ['test_class_selector', 'test_selector_w_children', 'test_selector_w_children_child_1']
    expected_class = 'a-test-class'
    result = pydom[f'.{expected_class}']
    div = result[0]
    assert len(result) == 3
    assert [el.id for el in result] == ids

def test_read_n_write_collection_elements():
    if False:
        i = 10
        return i + 15
    elements = pydom['.multi-elems']
    for element in elements:
        assert element.html == f"Content {element.id.replace('#', '')}"
    new_content = 'New Content'
    elements.html = new_content
    for element in elements:
        assert element.html == new_content

class TestElement:

    def test_query(self):
        if False:
            print('Hello World!')
        id_ = 'test_selector_w_children'
        parent_div = pydom[f'#{id_}'][0]
        div = parent_div.find('div')[0]
        assert div.parent == parent_div
        assert isinstance(div, pydom.BaseElement)
        assert div.html == 'Child 1'
        assert div.id == 'test_selector_w_children_child_1'

    def test_equality(self):
        if False:
            while True:
                i = 10
        id_ = 'test_id_selector'
        selector = f'#{id_}'
        div = pydom[selector][0]
        div2 = pydom[selector][0]
        assert div == div2
        assert div is not div2
        assert div.html == div2.html
        div.html = 'some value'
        assert div.html == div2.html == 'some value'

    def test_append_element(self):
        if False:
            i = 10
            return i + 15
        id_ = 'element-append-tests'
        div = pydom[f'#{id_}'][0]
        len_children_before = len(div.children)
        new_el = div.create('p')
        div.append(new_el)
        assert len(div.children) == len_children_before + 1
        assert div.children[-1] == new_el

    def test_append_js_element(self):
        if False:
            return 10
        id_ = 'element-append-tests'
        div = pydom[f'#{id_}'][0]
        len_children_before = len(div.children)
        new_el = div.create('p')
        div.append(new_el._js)
        assert len(div.children) == len_children_before + 1
        assert div.children[-1] == new_el

    def test_append_collection(self):
        if False:
            while True:
                i = 10
        id_ = 'element-append-tests'
        div = pydom[f'#{id_}'][0]
        len_children_before = len(div.children)
        collection = pydom['.collection']
        div.append(collection)
        assert len(div.children) == len_children_before + len(collection)
        for i in range(len(collection)):
            assert div.children[-1 - i] == collection[-1 - i]

    def test_read_classes(self):
        if False:
            print('Hello World!')
        id_ = 'test_class_selector'
        expected_class = 'a-test-class'
        div = pydom[f'#{id_}'][0]
        assert div.classes == [expected_class]

    def test_add_remove_class(self):
        if False:
            return 10
        id_ = 'div-no-classes'
        classname = 'tester-class'
        div = pydom[f'#{id_}'][0]
        assert not div.classes
        div.add_class(classname)
        same_div = pydom[f'#{id_}'][0]
        assert div.classes == [classname] == same_div.classes
        div.remove_class(classname)
        assert div.classes == [] == same_div.classes

    def test_when_decorator(self):
        if False:
            print('Hello World!')
        called = False
        just_a_button = pydom['#a-test-button'][0]

        @when('click', just_a_button)
        def on_click(event):
            if False:
                while True:
                    i = 10
            nonlocal called
            called = True
        assert not called
        just_a_button._js.click()
        assert called

class TestCollection:

    def test_iter_eq_children(self):
        if False:
            print('Hello World!')
        elements = pydom['.multi-elems']
        assert [el for el in elements] == [el for el in elements.children]
        assert len(elements) == 3

    def test_slices(self):
        if False:
            print('Hello World!')
        elements = pydom['.multi-elems']
        assert elements[0]
        _slice = elements[:2]
        assert len(_slice) == 2
        for (i, el) in enumerate(_slice):
            assert el == elements[i]
        assert elements[:] == elements

    def test_style_rule(self):
        if False:
            while True:
                i = 10
        selector = '.multi-elems'
        elements = pydom[selector]
        for el in elements:
            assert el.style['background-color'] != 'red'
        elements.style['background-color'] = 'red'
        for (i, el) in enumerate(pydom[selector]):
            assert elements[i].style['background-color'] == 'red'
            assert el.style['background-color'] == 'red'
        elements.style.remove('background-color')
        for (i, el) in enumerate(pydom[selector]):
            assert el.style['background-color'] != 'red'
            assert elements[i].style['background-color'] != 'red'

    def test_when_decorator(self):
        if False:
            return 10
        called = False
        buttons_collection = pydom['button']

        @when('click', buttons_collection)
        def on_click(event):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal called
            called = True
        assert not called
        for button in buttons_collection:
            button._js.click()
            assert called
            called = False

class TestCreation:

    def test_create_document_element(self):
        if False:
            for i in range(10):
                print('nop')
        new_el = pydom.create('div')
        new_el.id = 'new_el_id'
        assert isinstance(new_el, pydom.BaseElement)
        assert new_el._js.tagName == 'DIV'
        assert new_el.parent == None
        pydom.body.append(new_el)
        assert pydom['#new_el_id'][0].parent == pydom.body

    def test_create_element_child(self):
        if False:
            while True:
                i = 10
        selector = '#element-creation-test'
        parent_div = pydom[selector][0]
        new_el = parent_div.create('p', classes=['code-description'], html='Ciao PyScripters!')
        assert isinstance(new_el, pydom.BaseElement)
        assert new_el._js.tagName == 'P'
        assert new_el.parent == parent_div
        assert pydom[selector][0].children[0] == new_el

class TestInput:
    input_ids = ['test_rr_input_text', 'test_rr_input_button', 'test_rr_input_email', 'test_rr_input_password']

    def test_value(self):
        if False:
            while True:
                i = 10
        for id_ in self.input_ids:
            expected_type = id_.split('_')[-1]
            result = pydom[f'#{id_}']
            input_el = result[0]
            assert input_el._js.type == expected_type
            assert input_el.value == f'Content {id_}' == input_el._js.value
            new_value = f'New Value {expected_type}'
            input_el.value = new_value
            assert input_el.value == new_value
            new_value = f'Content {id_}'
            result.value = new_value
            assert input_el.value == new_value

    def test_set_value_collection(self):
        if False:
            i = 10
            return i + 15
        for id_ in self.input_ids:
            input_el = pydom[f'#{id_}']
            assert input_el.value[0] == f'Content {id_}' == input_el[0].value
            new_value = f'New Value {id_}'
            input_el.value = new_value
            assert input_el.value[0] == new_value == input_el[0].value

    def test_element_without_value(self):
        if False:
            i = 10
            return i + 15
        result = pydom[f'#tests-terminal'][0]
        with pytest.raises(AttributeError):
            result.value = 'some value'

    def test_element_without_collection(self):
        if False:
            return 10
        result = pydom[f'#tests-terminal']
        with pytest.raises(AttributeError):
            result.value = 'some value'