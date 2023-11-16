import os
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from fpdf import FPDF
test_data_dir = 'Test'
noWeapon_data_samples = 252
weapon_data_samples = 260
(img_width, img_height) = (160, 120)
json_file = open('modelBK3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('BK3.h5')
print('Model generated')
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
test_data = np.load(open('convolutionedTestImages.npy', 'rb'))
print('Test data: ' + repr(test_data.shape))
nb_test_samples = test_data.shape[0]
print('Test samples: ' + repr(nb_test_samples))
datagen = ImageDataGenerator(rescale=1.0 / 255, data_format='channels_first')
generatorTest = datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical', shuffle=False)
labels = []
i = 0
for (_, y) in generatorTest:
    i += len(y)
    labels.append(y)
    if i == nb_test_samples:
        break
labels = np.concatenate(labels)
test_labels = labels
print('Labels shape: ' + repr(test_labels.shape))
m = loaded_model.predict_classes([test_data], batch_size=1)
mProb = loaded_model.predict_proba([test_data], batch_size=1)
print(m.shape)
noWeapon_pre = m[:noWeapon_data_samples]
print(noWeapon_pre)
print(noWeapon_pre.shape)
noWeapon_pre_probs = mProb[:noWeapon_data_samples, :]
weapon_pre = m[noWeapon_data_samples:]
print(weapon_pre)
print(weapon_pre.shape)
weapon_pre_probs = mProb[noWeapon_data_samples:, :]
mypathP = 'Test/Pistolas'
onlyfilesP = [f for f in listdir(mypathP) if isfile(join(mypathP, f))]
print(onlyfilesP)
print('Files in weapon directory: ' + repr(len(onlyfilesP)))
total_weapons = weapon_data_samples
mypathN = 'Test/NoArma'
onlyfilesN = [f for f in listdir(mypathN) if isfile(join(mypathN, f))]
print(onlyfilesN)
print('Files in no weapon directory: ' + repr(len(onlyfilesN)))
total_noWeapons = noWeapon_data_samples
TP = sum(weapon_pre == 0)
FP = sum(noWeapon_pre == 0)
print('Total Clase Pistola Real')
print(total_weapons)
print('Total Clase Pistola Aciertos')
print(TP)
print('Total Clase NoArma Real')
print(total_noWeapons)
print('Total Clase NoArma Errores')
print(FP)
Acc = float(TP) / (TP + FP)
print('\nPrecision: ' + repr(Acc))
FN = total_weapons - TP
Rec = float(TP) / (TP + FN)
print('Recall: ' + repr(Rec))
F1m = Acc * Rec / (Acc + Rec) * 2
print('F1means: ' + repr(F1m))
print('\n\n')
threshold = 0.5
while threshold <= 1.01:
    aboveThrehold = sum(weapon_pre_probs[:, 0] >= threshold)
    print('Threshold: ' + repr(threshold) + '  -  hits: ' + repr(aboveThrehold))
    threshold += 0.05
    threshold = round(threshold, 2)

def ReporterPDF():
    if False:
        i = 10
        return i + 15
    print('\n\nReporterPDF working ...')
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    pdf.set_font('Arial', 'B', 9)
    pdf.cell(50, 10, 'KNIFES mal clasificadas', 1, 0, 'C')
    pdf.ln(11)
    for i in range(0, len(onlyfilesP)):
        if weapon_pre[i] != 0:
            pdf.cell(50, 10, str(['%.5f' % elem for elem in weapon_pre_probs[i, 0:10]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in weapon_pre_probs[i, 10:20]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in weapon_pre_probs[i, 20:28]]))
            pdf.ln(4)
            pdf.cell(50, 10, 'Class:' + repr(weapon_pre[i]))
            pdf.ln()
            pdf.image('Test/Pistolas/' + onlyfilesP[i], w=44, h=33)
    pdf.add_page()
    pdf.cell(50, 10, 'NoKnifes mal clasificadas', 1, 0, 'C')
    pdf.ln(11)
    for j in range(0, len(onlyfilesN)):
        if noWeapon_pre[j] == 0:
            pdf.cell(50, 10, str(['%.5f' % elem for elem in noWeapon_pre_probs[j, 0:10]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in noWeapon_pre_probs[j, 10:20]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in noWeapon_pre_probs[j, 20:28]]))
            pdf.ln()
            pdf.image('Test/NoArma/' + onlyfilesN[j], w=44, h=33)
    pdf.add_page()
    pdf.cell(50, 10, 'KNIFES bien clasificadas', 1, 0, 'C')
    pdf.ln(11)
    for i in range(0, len(onlyfilesP)):
        if weapon_pre[i] == 0:
            pdf.cell(50, 10, str(['%.5f' % elem for elem in weapon_pre_probs[i, 0:10]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in weapon_pre_probs[i, 10:20]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in weapon_pre_probs[i, 20:28]]))
            pdf.ln()
            pdf.image('Test/Pistolas/' + onlyfilesP[i], w=44, h=33)
    pdf.add_page()
    pdf.cell(50, 10, 'NoKnifes bien clasificadas', 1, 0, 'C')
    pdf.ln(11)
    for j in range(0, len(onlyfilesN)):
        if noWeapon_pre[j] != 0:
            pdf.cell(50, 10, str(['%.5f' % elem for elem in noWeapon_pre_probs[j, 0:10]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in noWeapon_pre_probs[j, 10:20]]))
            pdf.ln(4)
            pdf.cell(50, 10, str(['%.5f' % elem for elem in noWeapon_pre_probs[j, 20:28]]))
            pdf.ln()
            pdf.image('Test/NoArma/' + onlyfilesN[j], w=44, h=33)
    pdf.output('report.pdf', 'F')
    print('Report: succesful generation\n')
ReporterPDF()