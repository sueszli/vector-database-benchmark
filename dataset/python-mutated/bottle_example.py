from __future__ import absolute_import, print_function
from bottle import default_app, install, route, request, redirect, run, template
from pony.orm.examples.estore import *
from pony.orm.integration.bottle_plugin import PonyPlugin
install(PonyPlugin())

@route('/')
@route('/products/')
def all_products():
    if False:
        for i in range(10):
            print('nop')
    products = select((p for p in Product))
    return template('\n    <h1>List of products</h1>\n    <ul>\n    %for p in products:\n        <li><a href="/products/{{ p.id }}/">{{ p.name }}</a>\n    %end\n    </ul>\n    ', products=products)

@route('/products/:id/')
def show_product(id):
    if False:
        return 10
    p = Product[id]
    return template('\n    <h1>{{ p.name }}</h1>\n    <p>Price: {{ p.price }}</p>\n    <p>Product categories:</p>\n    <ul>\n    %for c in p.categories:\n        <li>{{ c.name }}\n    %end\n    </ul>\n    <a href="/products/{{ p.id }}/edit/">Edit product info</a>\n    <a href="/products/">Return to all products</a>\n    ', p=p)

@route('/products/:id/edit/')
def edit_product(id):
    if False:
        while True:
            i = 10
    p = Product[id]
    return template('\n    <form action=\'/products/{{ p.id }}/edit/\' method=\'post\'>\n      <table>\n        <tr>\n          <td>Product name:</td>\n          <td><input type="text" name="name" value="{{ p.name }}">\n        </tr>\n        <tr>\n          <td>Product price:</td>\n          <td><input type="text" name="price" value="{{ p.price }}">\n        </tr>\n      </table>\n      <input type="submit" value="Save!">\n    </form>\n    <p><a href="/products/{{ p.id }}/">Discard changes</a>\n    <p><a href="/products/">Return to all products</a>\n    ', p=p)

@route('/products/:id/edit/', method='POST')
def save_product(id):
    if False:
        while True:
            i = 10
    p = Product[id]
    p.name = request.forms.get('name')
    p.price = request.forms.get('price')
    redirect('/products/%d/' % p.id)
run(debug=True, host='localhost', port=8080, reloader=True)