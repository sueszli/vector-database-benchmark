#include "headless.h"

#include "vera/shaders/defaultShaders.h"


Headless::Headless() {
    m_sceneRender.showBBoxes = false;
};

Headless::~Headless() {
};

void Headless::init() {
    vera::WindowProperties props;
    props.style = vera::HEADLESS;
    vera::initGL(props);
    WatchFileList files;
    resetShaders( files );
}

void Headless::draw() {
    vera::updateGL();    
    uniforms.update();

    renderPrep();
    render();
    renderPost();
    renderDone();

    vera::renderGL();
}

void Headless::close() {
    vera::closeGL();
}

