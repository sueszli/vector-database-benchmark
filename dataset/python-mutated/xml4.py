from __future__ import absolute_import
from bzrlib.xml_serializer import Element, SubElement, XMLSerializer, escape_invalid_chars
from bzrlib.inventory import ROOT_ID, Inventory
import bzrlib.inventory as inventory
from bzrlib.revision import Revision
from bzrlib.errors import BzrError

class _Serializer_v4(XMLSerializer):
    """Version 0.0.4 serializer

    You should use the serializer_v4 singleton.

    v4 serialisation is no longer supported, only deserialisation.
    """
    __slots__ = []

    def _pack_entry(self, ie):
        if False:
            for i in range(10):
                print('nop')
        'Convert InventoryEntry to XML element'
        e = Element('entry')
        e.set('name', ie.name)
        e.set('file_id', ie.file_id)
        e.set('kind', ie.kind)
        if ie.text_size is not None:
            e.set('text_size', '%d' % ie.text_size)
        for f in ['text_id', 'text_sha1', 'symlink_target']:
            v = getattr(ie, f)
            if v is not None:
                e.set(f, v)
        if ie.parent_id != ROOT_ID:
            e.set('parent_id', ie.parent_id)
        e.tail = '\n'
        return e

    def _unpack_inventory(self, elt, revision_id=None, entry_cache=None, return_from_cache=False):
        if False:
            while True:
                i = 10
        'Construct from XML Element\n\n        :param revision_id: Ignored parameter used by xml5.\n        '
        root_id = elt.get('file_id') or ROOT_ID
        inv = Inventory(root_id)
        for e in elt:
            ie = self._unpack_entry(e, entry_cache=entry_cache, return_from_cache=return_from_cache)
            if ie.parent_id == ROOT_ID:
                ie.parent_id = root_id
            inv.add(ie)
        return inv

    def _unpack_entry(self, elt, entry_cache=None, return_from_cache=False):
        if False:
            for i in range(10):
                print('nop')
        parent_id = elt.get('parent_id')
        if parent_id is None:
            parent_id = ROOT_ID
        kind = elt.get('kind')
        if kind == 'directory':
            ie = inventory.InventoryDirectory(elt.get('file_id'), elt.get('name'), parent_id)
        elif kind == 'file':
            ie = inventory.InventoryFile(elt.get('file_id'), elt.get('name'), parent_id)
            ie.text_id = elt.get('text_id')
            ie.text_sha1 = elt.get('text_sha1')
            v = elt.get('text_size')
            ie.text_size = v and int(v)
        elif kind == 'symlink':
            ie = inventory.InventoryLink(elt.get('file_id'), elt.get('name'), parent_id)
            ie.symlink_target = elt.get('symlink_target')
        else:
            raise BzrError('unknown kind %r' % kind)
        return ie

    def _pack_revision(self, rev):
        if False:
            i = 10
            return i + 15
        'Revision object -> xml tree'
        root = Element('revision', committer=rev.committer, timestamp='%.9f' % rev.timestamp, revision_id=rev.revision_id, inventory_id=rev.inventory_id, inventory_sha1=rev.inventory_sha1)
        if rev.timezone:
            root.set('timezone', str(rev.timezone))
        root.text = '\n'
        msg = SubElement(root, 'message')
        msg.text = escape_invalid_chars(rev.message)[0]
        msg.tail = '\n'
        if rev.parents:
            pelts = SubElement(root, 'parents')
            pelts.tail = pelts.text = '\n'
            for (i, parent_id) in enumerate(rev.parents):
                p = SubElement(pelts, 'revision_ref')
                p.tail = '\n'
                p.set('revision_id', parent_id)
                if i < len(rev.parent_sha1s):
                    p.set('revision_sha1', rev.parent_sha1s[i])
        return root

    def _unpack_revision(self, elt):
        if False:
            return 10
        'XML Element -> Revision object'
        if elt.tag not in ('revision', 'changeset'):
            raise BzrError('unexpected tag in revision file: %r' % elt)
        rev = Revision(committer=elt.get('committer'), timestamp=float(elt.get('timestamp')), revision_id=elt.get('revision_id'), inventory_id=elt.get('inventory_id'), inventory_sha1=elt.get('inventory_sha1'))
        precursor = elt.get('precursor')
        precursor_sha1 = elt.get('precursor_sha1')
        pelts = elt.find('parents')
        if pelts:
            for p in pelts:
                rev.parent_ids.append(p.get('revision_id'))
                rev.parent_sha1s.append(p.get('revision_sha1'))
            if precursor:
                prec_parent = rev.parent_ids[0]
        elif precursor:
            rev.parent_ids.append(precursor)
            rev.parent_sha1s.append(precursor_sha1)
        v = elt.get('timezone')
        rev.timezone = v and int(v)
        rev.message = elt.findtext('message')
        return rev
'singleton instance'
serializer_v4 = _Serializer_v4()