"""GenomeDiagram script to mimic Proux et al 2002 Figure 6.

You can use the Entrez module to download the 3 required GenBank files

This is an extended version of the example in the Biopython Tutorial
which produces a GenomeDiagram figure close to Proux et al 2002 Figure 6.

See https://doi.org/10.1128/JB.184.21.6026-6036.2002
"""
from reportlab.lib import colors
from reportlab.lib.colors import red, grey, orange, green, brown
from reportlab.lib.colors import blue, lightblue, purple
from Bio.Graphics import GenomeDiagram
from Bio.Graphics.GenomeDiagram import CrossLink
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, SimpleLocation
name = 'Proux Fig 6'
A_rec = SeqIO.read('NC_002703.gbk', 'gb')
B_rec = SeqIO.read('AF323668.gbk', 'gb')
C_rec = SeqIO.read('NC_003212.gbk', 'gb')[2587879:2625807].reverse_complement(name=True)
records = {rec.name: rec for rec in [A_rec, B_rec, C_rec]}
A_colors = [red] * 5 + [grey] * 7 + [orange] * 2 + [grey] * 2 + [orange] + [grey] * 11 + [green] * 4 + [grey] + [green] * 2 + [grey, green] + [brown] * 5 + [blue] * 4 + [lightblue] * 5 + [grey, lightblue] + [purple] * 2 + [grey]
B_colors = [red] * 6 + [grey] * 8 + [orange] * 2 + [grey] + [orange] + [grey] * 21 + [green] * 5 + [grey] + [brown] * 4 + [blue] * 3 + [lightblue] * 3 + [grey] * 5 + [purple] * 2
C_colors = [grey] * 33 + [green] * 5 + [brown] * 4 + [blue] * 2 + [grey, blue] + [lightblue] * 2 + [grey] * 8
A_vs_B = [(99, 'Tuc2009_01', 'int'), (33, 'Tuc2009_03', 'orf4'), (94, 'Tuc2009_05', 'orf6'), (100, 'Tuc2009_06', 'orf7'), (97, 'Tuc2009_07', 'orf8'), (98, 'Tuc2009_08', 'orf9'), (98, 'Tuc2009_09', 'orf10'), (100, 'Tuc2009_10', 'orf12'), (100, 'Tuc2009_11', 'orf13'), (94, 'Tuc2009_12', 'orf14'), (87, 'Tuc2009_13', 'orf15'), (94, 'Tuc2009_14', 'orf16'), (94, 'Tuc2009_15', 'orf17'), (88, 'Tuc2009_17', 'rusA'), (91, 'Tuc2009_18', 'orf20'), (93, 'Tuc2009_19', 'orf22'), (71, 'Tuc2009_20', 'orf23'), (51, 'Tuc2009_22', 'orf27'), (97, 'Tuc2009_23', 'orf28'), (88, 'Tuc2009_24', 'orf29'), (26, 'Tuc2009_26', 'orf38'), (19, 'Tuc2009_46', 'orf52'), (77, 'Tuc2009_48', 'orf54'), (91, 'Tuc2009_49', 'orf55'), (95, 'Tuc2009_52', 'orf60')]
B_vs_C = [(42, 'orf39', 'lin2581'), (31, 'orf40', 'lin2580'), (49, 'orf41', 'lin2579'), (54, 'orf42', 'lin2578'), (55, 'orf43', 'lin2577'), (33, 'orf44', 'lin2576'), (51, 'orf46', 'lin2575'), (33, 'orf47', 'lin2574'), (40, 'orf48', 'lin2573'), (25, 'orf49', 'lin2572'), (50, 'orf50', 'lin2571'), (48, 'orf51', 'lin2570'), (24, 'orf52', 'lin2568'), (30, 'orf53', 'lin2567'), (28, 'orf54', 'lin2566')]

def get_feature(features, id, tags=('locus_tag', 'gene', 'old_locus_tag')):
    if False:
        print('Hello World!')
    'Search list of SeqFeature objects for an identifier under the given tags.'
    for f in features:
        for key in tags:
            for x in f.qualifiers.get(key, []):
                if x == id:
                    return f
    raise KeyError(id)
gd_diagram = GenomeDiagram.Diagram(name)
feature_sets = {}
max_len = 0
for (i, record) in enumerate([A_rec, B_rec, C_rec]):
    max_len = max(max_len, len(record))
    gd_track_for_features = gd_diagram.new_track(5 - 2 * i, name=record.name, greytrack=True, height=0.5, start=0, end=len(record))
    assert record.name not in feature_sets
    feature_sets[record.name] = gd_track_for_features.new_set()
for (X, Y, X_vs_Y) in [('NC_002703', 'AF323668', A_vs_B), ('AF323668', 'NC_003212', B_vs_C)]:
    features_X = records[X].features
    features_Y = records[Y].features
    set_X = feature_sets[X]
    set_Y = feature_sets[Y]
    for (score, x, y) in X_vs_Y:
        color = colors.linearlyInterpolatedColor(colors.white, colors.firebrick, 0, 100, score)
        border = colors.lightgrey
        f_x = get_feature(features_X, x)
        F_x = set_X.add_feature(SeqFeature(SimpleLocation(f_x.location.start, f_x.location.end, strand=0)), color=color, border=border)
        f_y = get_feature(features_Y, y)
        F_y = set_Y.add_feature(SeqFeature(SimpleLocation(f_y.location.start, f_y.location.end, strand=0)), color=color, border=border)
        gd_diagram.cross_track_links.append(CrossLink(F_x, F_y, color, border))
for (record, gene_colors) in zip([A_rec, B_rec, C_rec], [A_colors, B_colors, C_colors]):
    gd_feature_set = feature_sets[record.name]
    i = 0
    for feature in record.features:
        if feature.type != 'gene':
            continue
        try:
            g_color = gene_colors[i]
        except IndexError:
            print("Don't have color for %s gene %i" % (record.name, i))
            g_color = grey
        gd_feature_set.add_feature(feature, sigil='BIGARROW', color=g_color, label=True, name=str(i + 1), label_position='start', label_size=6, label_angle=0)
        i += 1
gd_diagram.draw(format='linear', pagesize='A4', fragments=1, start=0, end=max_len)
gd_diagram.write(name + '.pdf', 'PDF')
gd_diagram.write(name + '.eps', 'EPS')
gd_diagram.write(name + '.svg', 'SVG')