/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "MaterialDataWidget.h"
#include <limits>

namespace sofa::gui::qt::materialdatawidget_h
{
helper::Creator<DataWidgetFactory,MaterialDataWidget> DWClass_MeshMaterial("default",true);
helper::Creator<DataWidgetFactory,VectorMaterialDataWidget> DWClass_MeshVectorMaterial("default",true);

bool MaterialDataWidget::createWidgets()
{

    _nameEdit = new QLineEdit(this);
    _ambientPicker = new QRGBAColorPicker(this);
    _ambientCheckBox = new QCheckBox(this);
    _emissivePicker = new QRGBAColorPicker(this);
    _emissiveCheckBox = new QCheckBox(this);
    _specularPicker = new QRGBAColorPicker(this);
    _specularCheckBox = new QCheckBox(this);
    _diffusePicker = new QRGBAColorPicker(this);
    _diffuseCheckBox = new QCheckBox(this);
    _shininessEdit = new QLineEdit(this);
    _shininessEdit->setValidator(new QDoubleValidator(this) );
    _shininessCheckBox = new QCheckBox(this);
    QVBoxLayout* layout = new QVBoxLayout(this);

    //QGridLayout* grid = new QGridLayout(5,3);
    QGridLayout* grid = new QGridLayout();
    grid->setSpacing(1);
    QHBoxLayout* hlayout = new QHBoxLayout();
    hlayout->addWidget(new QLabel("Name",this));
    hlayout->addWidget(_nameEdit);

    grid->addWidget(_ambientCheckBox,0,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Ambient",this),0,1,Qt::AlignHCenter);
    grid->addWidget(_ambientPicker,0,2,Qt::AlignHCenter);

    grid->addWidget(_emissiveCheckBox,1,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Emissive",this),1,1,Qt::AlignHCenter);
    grid->addWidget(_emissivePicker,1,2,Qt::AlignHCenter);

    grid->addWidget(_diffuseCheckBox,2,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Diffuse",this),2,1,Qt::AlignHCenter);
    grid->addWidget(_diffusePicker,2,2,Qt::AlignHCenter);

    grid->addWidget(_specularCheckBox,3,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Specular",this),3,1,Qt::AlignHCenter);
    grid->addWidget(_specularPicker,3,2,Qt::AlignHCenter);

    grid->addWidget(_shininessCheckBox,4,0,Qt::AlignHCenter);
    grid->addWidget(new QLabel("Shininess",this),4,1,Qt::AlignHCenter);
    grid->addWidget(_shininessEdit,4,2,Qt::AlignHCenter);

    layout->addLayout(hlayout);
    layout->addLayout(grid);


    connect(_nameEdit, &QLineEdit::textChanged, this , [=](const QString &){ setWidgetDirty(true); });
    connect(_shininessEdit, &QLineEdit::textChanged, this , [=](const QString &){ setWidgetDirty(true); });

    connect(_ambientCheckBox, &QCheckBox::toggled, this , &MaterialDataWidget::setWidgetDirty);
    connect(_ambientCheckBox, &QCheckBox::toggled, _ambientPicker , &MaterialDataWidget::setEnabled);

    connect(_emissiveCheckBox, &QCheckBox::toggled, this , &MaterialDataWidget::setWidgetDirty);
    connect(_emissiveCheckBox, &QCheckBox::toggled, _emissivePicker, &MaterialDataWidget::setEnabled);

    connect(_specularCheckBox, &QCheckBox::toggled, this , &MaterialDataWidget::setWidgetDirty);
    connect(_specularCheckBox, &QCheckBox::toggled, _specularPicker, &MaterialDataWidget::setEnabled);

    connect(_diffuseCheckBox, &QCheckBox::toggled, this , &MaterialDataWidget::setWidgetDirty);
    connect(_diffuseCheckBox, &QCheckBox::toggled, _diffusePicker, &MaterialDataWidget::setEnabled);

    connect(_shininessCheckBox, &QCheckBox::toggled, this, &MaterialDataWidget::setWidgetDirty);
    connect(_shininessCheckBox, &QCheckBox::toggled, _shininessEdit, &MaterialDataWidget::setEnabled);

    connect(_ambientPicker, &QRGBAColorPicker::hasChanged, this, [=](){ setWidgetDirty(true); });
    connect(_emissivePicker, &QRGBAColorPicker::hasChanged, this, [=](){ setWidgetDirty(true); });
    connect(_specularPicker, &QRGBAColorPicker::hasChanged, this, [=](){ setWidgetDirty(true); });
    connect(_diffusePicker, &QRGBAColorPicker::hasChanged, this, [=](){ setWidgetDirty(true); });

    readFromData();

    return true;
}
void MaterialDataWidget::setDataReadOnly(bool readOnly)
{
    _nameEdit->setReadOnly(readOnly);
    _nameEdit->setEnabled(!readOnly);
    _ambientPicker->setEnabled(!readOnly);
    _ambientCheckBox->setEnabled(!readOnly);
    _emissivePicker->setEnabled(!readOnly);
    _emissiveCheckBox->setEnabled(!readOnly);
    _specularPicker->setEnabled(!readOnly);
    _specularCheckBox->setEnabled(!readOnly);
    _diffusePicker->setEnabled(!readOnly);
    _diffuseCheckBox->setEnabled(!readOnly);
    _shininessEdit->setReadOnly(readOnly);
    _shininessCheckBox->setEnabled(!readOnly);
}
void MaterialDataWidget::readFromData()
{
    const Material& material = getData()->getValue();
    _nameEdit->setText( QString( material.name.c_str() ) );
    _ambientCheckBox->setChecked( material.useAmbient );
    _emissiveCheckBox->setChecked( material.useEmissive );
    _diffuseCheckBox->setChecked( material.useDiffuse );
    _specularCheckBox->setChecked( material.useSpecular );
    _shininessCheckBox->setChecked(material.useShininess);
    QString str;
    str.setNum(material.shininess);
    _shininessEdit->setText(str);

    _ambientPicker->setColor( material.ambient );
    _emissivePicker->setColor( material.emissive );
    _specularPicker->setColor( material.specular );
    _diffusePicker->setColor( material.diffuse );

    _ambientPicker->setEnabled( _ambientCheckBox->isChecked() );
    _emissivePicker->setEnabled( _emissiveCheckBox->isChecked() );
    _specularPicker->setEnabled( _specularCheckBox->isChecked() );
    _diffusePicker->setEnabled( _diffuseCheckBox->isChecked() );


}
void MaterialDataWidget::writeToData()
{
    Material* material = getData()->beginEdit();

    material->name      = _nameEdit->text().toStdString();
    material->ambient   = _ambientPicker->getColor();
    material->diffuse   = _diffusePicker->getColor();
    material->emissive  = _emissivePicker->getColor();
    material->specular  = _specularPicker->getColor();
    material->shininess = _shininessEdit->text().toFloat();
    material->useAmbient = _ambientCheckBox->isChecked();
    material->useDiffuse = _diffuseCheckBox->isChecked();
    material->useShininess = _shininessCheckBox->isChecked();
    material->useEmissive = _emissiveCheckBox->isChecked();
    material->useSpecular = _specularCheckBox->isChecked();


    getData()->endEdit();

}


bool VectorMaterialDataWidget::createWidgets()
{
    if( getData()->getValue().empty() )
    {
        return false;
    }
    _comboBox = new QComboBox(this);
    _materialDataWidget = new MaterialDataWidget(this,this->objectName().toStdString().c_str(),&_currentMaterial);
    _materialDataWidget->createWidgets();
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(_comboBox);
    layout->addWidget(_materialDataWidget);

    connect( _comboBox, QOverload<int>::of(&QComboBox::activated), this, &VectorMaterialDataWidget::changeMaterial );
    connect( _materialDataWidget, &MaterialDataWidget::WidgetDirty, this, &VectorMaterialDataWidget::setWidgetDirty );

    readFromData();

    return true;
}

void VectorMaterialDataWidget::setDataReadOnly(bool readOnly)
{
    if (_materialDataWidget)
        _materialDataWidget->setDataReadOnly(readOnly);
}

void VectorMaterialDataWidget::readFromData()
{
    VectorMaterial::const_iterator iter;
    const VectorMaterial& vecMaterial = getData()->getValue();
    if( vecMaterial.empty() )
    {
        return;
    }
    _comboBox->clear();
    _vectorEditedMaterial.clear();
    std::copy(vecMaterial.begin(), vecMaterial.end(), std::back_inserter(_vectorEditedMaterial) );
    for( iter = _vectorEditedMaterial.begin(); iter != _vectorEditedMaterial.end(); ++iter )
    {
        _comboBox->addItem( QString( (*iter).name.c_str() ) );
    }
    _currentMaterialPos = 0;
    _comboBox->setCurrentIndex(_currentMaterialPos);
    _currentMaterial.setValue(_vectorEditedMaterial[_currentMaterialPos]);
    _materialDataWidget->setData(&_currentMaterial);
    _materialDataWidget->updateWidgetValue();
}

void VectorMaterialDataWidget::changeMaterial( int index )
{
    //Save previous Material
    _materialDataWidget->updateDataValue();
    const Material mat(_currentMaterial.getValue() );
    _vectorEditedMaterial[_currentMaterialPos] = mat;

    //Update current Material
    _currentMaterialPos = index;
    _currentMaterial.setValue(_vectorEditedMaterial[index]);

    //Update Widget
    _materialDataWidget->setData(&_currentMaterial);
    _materialDataWidget->updateWidgetValue();
}

void VectorMaterialDataWidget::writeToData()
{
    _materialDataWidget->updateDataValue();
    const Material mat(_currentMaterial.getValue() );
    _vectorEditedMaterial[_currentMaterialPos] = mat;

    VectorMaterial* vecMaterial = getData()->beginEdit();
    assert(vecMaterial->size() == _vectorEditedMaterial.size() );
    std::copy(_vectorEditedMaterial.begin(), _vectorEditedMaterial.end(), vecMaterial->begin() );

    getData()->endEdit();
}

} // namespace sofa::gui::qt::materialdatawidget_h
