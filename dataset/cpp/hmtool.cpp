#include <SDL.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <libtcod.hpp>
#include <libtcod/gui/gui.hpp>
#include <libtcod/timer.hpp>
#include <string>

#include "operation.hpp"

TCODHeightMap* hm = NULL;
static TCODHeightMap* hm_old = NULL;
TCODNoise* noise = NULL;
TCODRandom* rnd = NULL;
static TCODRandom* backupRnd = NULL;
static TCODConsole* guicon = NULL;
static bool greyscale = false;
static bool slope = false;
static bool normal = false;
bool isNormalized = true;
static bool oldNormalized = true;
static std::string msg = "";
float msgDelay = 0.0f;
static constexpr float hillRadius = 0.1f;
static constexpr float hillVariation = 0.5f;
float addFbmDelta = 0.0f;
float scaleFbmDelta = 0.0f;
uint32_t seed = 0xdeadbeef;

static float sandHeight = 0.12f;
static float grassHeight = 0.315f;
static float snowHeight = 0.785f;

static char landMassTxt[128] = "";
static char minZTxt[128] = "";
static char maxZTxt[128] = "";
static char seedTxt[128] = "";
float mapmin = 0.0f, mapmax = 0.0f;
static float oldmapmin = 0.0f, oldmapmax = 0.0f;

// ui
std::shared_ptr<ToolBar> params = nullptr;
std::shared_ptr<ToolBar> history = nullptr;
std::shared_ptr<ToolBar> colorMapGui = nullptr;
std::shared_ptr<Button> change_color_map_button = nullptr;

static float voronoiCoef[2] = {-1.0f, 1.0f};

static TCODColor mapGradient[256];
static constexpr auto MAX_COLOR_KEY = 10;

// TCOD's land color map
static int keyIndex[MAX_COLOR_KEY] = {
    0,
    (int)(sandHeight * 255),
    (int)(sandHeight * 255) + 4,
    (int)(grassHeight * 255),
    (int)(grassHeight * 255) + 10,
    (int)(snowHeight * 255),
    (int)(snowHeight * 255) + 10,
    255};
static TCODColor keyColor[MAX_COLOR_KEY] = {
    TCODColor(0, 0, 50),  // deep water
    TCODColor(30, 30, 170),  // water-sand transition
    TCODColor(114, 150, 71),  // sand
    TCODColor(80, 120, 10),  // sand-grass transition
    TCODColor(17, 109, 7),  // grass
    TCODColor(120, 220, 120),  // grass-snow transistion
    TCODColor(208, 208, 239),  // snow
    TCODColor(255, 255, 255)};
static std::shared_ptr<Image> keyImages[MAX_COLOR_KEY];

static constexpr int nbColorKeys = 8;

static constexpr auto BLACK = TCODColor{0, 0, 0};
static constexpr auto WHITE = TCODColor{255, 255, 255};
static constexpr auto LIGHT_BLUE = TCODColor{63, 63, 255};

static constexpr auto BLACK_ = TCOD_ColorRGB{0, 0, 0};
static constexpr auto WHITE_ = TCOD_ColorRGB{255, 255, 255};
static constexpr auto LIGHT_BLUE_ = TCOD_ColorRGB{63, 63, 255};

void initColors() { TCODColor::genMap(mapGradient, nbColorKeys, keyColor, keyIndex); }

void render(TCOD_Console& console, float delta_time) {
  TCODHeightMap backup(HM_WIDTH, HM_HEIGHT);
  isNormalized = true;
  for (auto& tile : console) tile = {' ', {255, 255, 255, 255}, {0, 0, 0, 255}};
  backup.copy(hm);
  mapmin = 1E8f;
  mapmax = -1E8f;
  for (int i = 0; i < HM_WIDTH * HM_HEIGHT; ++i) {
    if (hm->values[i] < 0.0f || hm->values[i] > 1.0f) {
      isNormalized = false;
    }
    if (hm->values[i] < mapmin) mapmin = hm->values[i];
    if (hm->values[i] > mapmax) mapmax = hm->values[i];
  }
  if (!isNormalized) backup.normalize();
  // render the TCODHeightMap
  for (int y = 0; y < HM_HEIGHT; ++y) {
    for (int x = 0; x < HM_WIDTH; ++x) {
      float z = backup.getValue(x, y);
      uint8_t val = (uint8_t)(z * 255);
      if (slope) {
        // render the slope map
        z = CLAMP(0.0f, 1.0f, hm->getSlope(x, y) * 10.0f);
        val = (uint8_t)(z * 255);
        console.at({x, y}).bg = {val, val, val, 255};
      } else if (greyscale) {
        // render the greyscale heightmap
        console.at({x, y}).bg = {val, val, val, 255};
      } else if (normal) {
        // render the normal map
        float n[3];
        hm->getNormal((float)x, (float)y, n, mapmin);
        uint8_t r = (uint8_t)((n[0] * 0.5f + 0.5f) * 255);
        uint8_t g = (uint8_t)((n[1] * 0.5f + 0.5f) * 255);
        uint8_t b = (uint8_t)((n[2] * 0.5f + 0.5f) * 255);
        console.at({x, y}).bg = {r, g, b, 255};
      } else {
        // render the colored heightmap
        console.at({x, y}).bg = {mapGradient[val].r, mapGradient[val].g, mapGradient[val].b, 255};
      }
    }
  }
  snprintf(minZTxt, sizeof(minZTxt), "min z    : %.2f", mapmin);
  snprintf(maxZTxt, sizeof(maxZTxt), "max z    : %.2f", mapmax);
  snprintf(seedTxt, sizeof(seedTxt), "seed     : %X", seed);
  const float landProportion = 100.0f - 100.0f * backup.countCells(0.0f, sandHeight) / (hm->w * hm->h);
  snprintf(landMassTxt, sizeof(landMassTxt), "landMass : %d %%", (int)landProportion);
  if (!isNormalized) {
    tcod::print(
        console, {HM_WIDTH / 2, HM_HEIGHT - 1}, "the map is not normalized !", WHITE_, std::nullopt, TCOD_CENTER);
  }
  // message
  msgDelay -= delta_time;
  if (msg.size() && msgDelay > 0.0f) {
    const int msg_height = tcod::print_rect(
        console, {HM_WIDTH / 4 + 1, HM_HEIGHT / 2 + 1, HM_WIDTH / 2 - 1, 0}, msg, WHITE_, std::nullopt, TCOD_CENTER);
    tcod::draw_rect(console, {HM_WIDTH / 4, HM_HEIGHT / 2, HM_WIDTH / 2, msg_height + 2}, 0, std::nullopt, LIGHT_BLUE_);
  }
}

template <typename... T>
void message(float delay, const char* fmt, T... args) {
  msgDelay = delay;
  msg = tcod::stringf(fmt, args...);
}

void backup() {
  // save the heightmap & RNG states
  for (int y = 0; y < HM_HEIGHT; ++y) {
    for (int x = 0; x < HM_WIDTH; ++x) {
      hm_old->setValue(x, y, hm->getValue(x, y));
    }
  }
  if (backupRnd) delete backupRnd;
  backupRnd = rnd->save();
  oldNormalized = isNormalized;
  oldmapmax = mapmax;
  oldmapmin = mapmin;
}

void restore() {
  // restore the previously saved heightmap & RNG states
  for (int y = 0; y < HM_HEIGHT; ++y) {
    for (int x = 0; x < HM_WIDTH; ++x) {
      hm->setValue(x, y, hm_old->getValue(x, y));
    }
  }
  rnd->restore(backupRnd);
  isNormalized = oldNormalized;
  mapmax = oldmapmax;
  mapmin = oldmapmin;
}

void save() {
  // TODO
  message(2.0f, "Saved.");
}

void load() {
  // TODO
}

void addHill(int nbHill, float baseRadius, float radiusVar, float height) {
  for (int i = 0; i < nbHill; ++i) {
    const float hillMinRadius = baseRadius * (1.0f - radiusVar);
    const float hillMaxRadius = baseRadius * (1.0f + radiusVar);
    const float radius = rnd->getFloat(hillMinRadius, hillMaxRadius);
    const float theta = rnd->getFloat(0.0f, 6.283185f);  // between 0 and 2Pi
    const float dist = rnd->getFloat(0.0f, (float)MIN(HM_WIDTH, HM_HEIGHT) / 2 - radius);
    const int xh = (int)(HM_WIDTH / 2 + cos(theta) * dist);
    const int yh = (int)(HM_HEIGHT / 2 + sin(theta) * dist);
    hm->addHill((float)xh, (float)yh, radius, height);
  }
}

void clearCbk() {
  hm->clear();
  Operation::clear();
  history->clear();
  params->clear();
  params->setVisible(false);
}

void reseedCbk() {
  seed = rnd->getInt(0x7FFFFFFF, 0xFFFFFFFF);
  Operation::reseed();
  message(3.0f, "Switching to seed %X", seed);
}

void cancelCbk() { Operation::cancel(); }

// operations buttons callbacks
void normalizeCbk() { (new NormalizeOperation(0.0f, 1.0f))->add(); }

void addFbmCbk() { (new AddFbmOperation(1.0f, addFbmDelta, 0.0f, 6.0f, 1.0f, 0.5f))->add(); }

void scaleFbmCbk() { (new ScaleFbmOperation(1.0f, addFbmDelta, 0.0f, 6.0f, 1.0f, 0.5f))->add(); }

void addHillCbk() {
  (new AddHillOperation(25, 10.0f, 0.5f, (mapmax == mapmin ? 0.5f : (mapmax - mapmin) * 0.1f)))->add();
}

void rainErosionCbk() { (new RainErosionOperation(1000, 0.05f, 0.05f))->add(); }

void smoothCbk() { (new SmoothOperation(mapmin, mapmax, 2))->add(); }

void voronoiCbk() { (new VoronoiOperation(100, 2, voronoiCoef))->add(); }

void noiseLerpCbk() {
  float v = (mapmax - mapmin) * 0.5f;
  if (v == 0.0f) v = 1.0f;
  (new NoiseLerpOperation(0.0f, 1.0f, addFbmDelta, 0.0f, 6.0f, v, v))->add();
}

void raiseLowerCbk() { (new AddLevelOperation(0.0f))->add(); }

// In/Out buttons callbacks

void exportCCbk() {
  const std::string code = Operation::buildCode(Operation::C);
  auto f = std::ofstream("hm.c", std::ios_base::trunc);
  f << code;
  f.close();
  message(3.0f, "The code has been exported to ./hm.c");
}

void exportPyCbk() {
  const std::string code = Operation::buildCode(Operation::PY);
  auto f = std::ofstream("hm.py", std::ios_base::trunc);
  f << code;
  f.close();
  message(3.0f, "The code has been exported to ./hm.py");
}

void exportCppCbk() {
  const std::string code = Operation::buildCode(Operation::CPP);
  auto f = std::ofstream("hm.cpp", std::ios_base::trunc);
  f << code;
  f.close();
  message(3.0f, "The code has been exported to ./hm.cpp");
}

void exportBmpCbk() {
  TCODImage img(HM_WIDTH, HM_HEIGHT);
  for (int x = 0; x < HM_WIDTH; x++) {
    for (int y = 0; y < HM_HEIGHT; y++) {
      float z = hm->getValue(x, y);
      uint8_t val = (uint8_t)(z * 255);
      if (slope) {
        // render the slope map
        z = CLAMP(0.0f, 1.0f, hm->getSlope(x, y) * 10.0f);
        val = (uint8_t)(z * 255);
        TCODColor c(val, val, val);
        img.putPixel(x, y, c);
      } else if (greyscale) {
        // render the greyscale heightmap
        TCODColor c(val, val, val);
        img.putPixel(x, y, c);
      } else if (normal) {
        // render the normal map
        float n[3];
        hm->getNormal((float)x, (float)y, n, mapmin);
        uint8_t r = (uint8_t)((n[0] * 0.5f + 0.5f) * 255);
        uint8_t g = (uint8_t)((n[1] * 0.5f + 0.5f) * 255);
        uint8_t b = (uint8_t)((n[2] * 0.5f + 0.5f) * 255);
        TCODColor c(r, g, b);
        img.putPixel(x, y, c);
      } else {
        // render the colored heightmap
        img.putPixel(x, y, mapGradient[val]);
      }
    }
  }
  img.save("hm.bmp");
  message(3.0f, "The bitmap has been exported to ./hm.bmp");
}

// Display buttons callbacks
void colorMapCbk() {
  slope = false;
  greyscale = false;
  normal = false;
}

void greyscaleCbk() {
  slope = false;
  greyscale = true;
  normal = false;
}

void slopeCbk() {
  slope = true;
  greyscale = false;
  normal = false;
}

void normalCbk() {
  slope = false;
  greyscale = false;
  normal = true;
}

void changeColorMapEndCbk() { colorMapGui->setVisible(false); }

void changeColorMapCbk() {
  colorMapGui->move(change_color_map_button->x + change_color_map_button->w + 2, change_color_map_button->y);
  colorMapGui->clear();
  for (intptr_t i = 0; i < nbColorKeys; i++) {
    colorMapGui->addSeparator(tcod::stringf("Color %d", static_cast<int>(i)).c_str());
    auto h_box = std::make_shared<HBox>(0, 0, 0);
    auto v_box = std::make_shared<VBox>(0, 0, 0);
    colorMapGui->addWidget(h_box);
    auto index_slider =
        std::make_shared<Slider>(0, 0, 3, 0.0f, 255.0f, "index", "Index of the key in the color map (0-255)");
    index_slider->setValue((float)keyIndex[i]);
    index_slider->setFormat("%.0f");
    index_slider->setCallback([i = i](float val) {
      keyIndex[i] = (int)(val);
      if (i == 1) sandHeight = (float)(i) / 255.0f;
      initColors();
    });
    v_box->addWidget(index_slider);
    keyImages[i] = std::make_shared<Image>(0, 0, 0, 2);
    keyImages[i]->setBackgroundColor(keyColor[i]);
    v_box->addWidget(keyImages[i]);
    h_box->addWidget(v_box);

    v_box = std::make_shared<VBox>(0, 0, 0);
    h_box->addWidget(v_box);
    auto redSlider = std::make_shared<Slider>(0, 0, 3, 0.0f, 255.0f, "r", "Red component of the color");
    redSlider->setValue((float)keyColor[i].r);
    redSlider->setFormat("%.0f");
    redSlider->setCallback([i = i](float val) {
      keyColor[i].r = static_cast<uint8_t>(val);
      keyImages[i]->setBackgroundColor(keyColor[i]);
      initColors();
    });
    v_box->addWidget(redSlider);
    auto greenSlider = std::make_shared<Slider>(0, 0, 3, 0.0f, 255.0f, "g", "Green component of the color");
    greenSlider->setValue((float)keyColor[i].g);
    greenSlider->setFormat("%.0f");
    greenSlider->setCallback([i = i](float val) {
      keyColor[i].g = static_cast<uint8_t>(val);
      keyImages[i]->setBackgroundColor(keyColor[i]);
      initColors();
    });
    v_box->addWidget(greenSlider);
    auto blueSlider = std::make_shared<Slider>(0, 0, 3, 0.0f, 255.0f, "b", "Blue component of the color");
    blueSlider->setValue((float)keyColor[i].b);
    blueSlider->setFormat("%.0f");
    blueSlider->setCallback([i = i](float val) {
      keyColor[i].b = static_cast<uint8_t>(val);
      keyImages[i]->setBackgroundColor(keyColor[i]);
      initColors();
    });
    v_box->addWidget(blueSlider);
  }
  colorMapGui->addWidget(std::make_shared<Button>("Ok", "", changeColorMapEndCbk));
  colorMapGui->setVisible(true);
}

// build gui

void addOperationButton(ToolBar& tools, Operation::OpType type, void (*cbk)()) {
  tools.addWidget(std::make_shared<Button>(Operation::names[type], Operation::tips[type], cbk));
}

void buildGui() {
  // status bar
  static auto status_bar = std::make_unique<StatusBar>(0, 0, HM_WIDTH, 1);

  static auto vbox = std::make_unique<VBox>(0, 2, 1);
  // stats
  auto stats = std::make_shared<ToolBar>(0, 0, 21, "Stats", "Statistics about the current map");
  stats->addWidget(std::make_shared<Label>(0, 0, landMassTxt, "Ratio of land surface / total surface"));
  stats->addWidget(std::make_shared<Label>(0, 0, minZTxt, "Minimum z value in the map"));
  stats->addWidget(std::make_shared<Label>(0, 0, maxZTxt, "Maximum z value in the map"));
  stats->addWidget(std::make_shared<Label>(0, 0, seedTxt, "Current random seed used to build the map"));
  vbox->addWidget(stats);

  // tools
  auto tools = std::make_shared<ToolBar>(0, 0, 15, "Tools", "Tools to modify the heightmap");
  tools->addWidget(std::make_shared<Button>("cancel", "Delete the selected operation", cancelCbk));
  tools->addWidget(
      std::make_shared<Button>("clear", "Remove all operations and reset all heightmap values to 0.0", clearCbk));
  tools->addWidget(std::make_shared<Button>("reseed", "Replay all operations with a new random seed", reseedCbk));

  // operations
  tools->addSeparator("Operations", "Apply a new operation to the map");
  addOperationButton(*tools, Operation::NORM, normalizeCbk);
  addOperationButton(*tools, Operation::ADD_FBM, addFbmCbk);
  addOperationButton(*tools, Operation::SCALE_FBM, scaleFbmCbk);
  addOperationButton(*tools, Operation::ADDHILL, addHillCbk);
  addOperationButton(*tools, Operation::RAIN, rainErosionCbk);
  addOperationButton(*tools, Operation::SMOOTH, smoothCbk);
  addOperationButton(*tools, Operation::VORONOI, voronoiCbk);
  addOperationButton(*tools, Operation::NOISE_LERP, noiseLerpCbk);
  addOperationButton(*tools, Operation::ADDLEVEL, raiseLowerCbk);

  // display
  tools->addSeparator("Display", "Change the type of display");
  RadioButton::setDefaultGroup(1);
  auto colormap = std::make_shared<RadioButton>("colormap", "Enable colormap mode", colorMapCbk);
  tools->addWidget(colormap);
  colormap->select();
  tools->addWidget(std::make_shared<RadioButton>("slope", "Enable slope mode", slopeCbk));
  tools->addWidget(std::make_shared<RadioButton>("greyscale", "Enable greyscale mode", greyscaleCbk));
  tools->addWidget(std::make_shared<RadioButton>("normal", "Enable normal map mode", normalCbk));
  change_color_map_button =
      std::make_shared<Button>("change colormap", "Modify the colormap used by hmtool", changeColorMapCbk);
  tools->addWidget(change_color_map_button);

  // change colormap gui
  colorMapGui =
      std::make_shared<ToolBar>(0, 0, "Colormap", "Select the color and position of the keys in the color map");
  colorMapGui->setVisible(false);

  // in/out
  tools->addSeparator("In/Out", "Import/Export stuff");
  tools->addWidget(
      std::make_shared<Button>("export C", "Generate the C code for this heightmap in ./hm.c", exportCCbk));
  tools->addWidget(
      std::make_shared<Button>("export CPP", "Generate the CPP code for this heightmap in ./hm.cpp", exportCppCbk));
  tools->addWidget(
      std::make_shared<Button>("export PY", "Generate the Python code for this heightmap in ./hm.py", exportPyCbk));
  tools->addWidget(std::make_shared<Button>("export bmp", "Save this heightmap as a bitmap in ./hm.bmp", exportBmpCbk));

  vbox->addWidget(tools);

  // params box
  params = std::make_shared<ToolBar>(0, 0, "Params", "Parameters of the current tool");
  vbox->addWidget(params);
  params->setVisible(false);

  // history
  history = std::make_shared<ToolBar>(0, tools->y + 1 + tools->h, 15, "History", "History of operations");
  vbox->addWidget(history);
}

int main(int argc, char* argv[]) {
  TCOD_ContextParams context_params{};
  context_params.tcod_version = SDL_COMPILEDVERSION;
  context_params.columns = HM_WIDTH;
  context_params.rows = HM_HEIGHT;
  context_params.argc = argc;
  context_params.argv = argv;
  context_params.vsync = true;
  context_params.window_title = "height map tool";
  context_params.sdl_window_flags = SDL_WINDOW_RESIZABLE;

  TCODConsole::initRoot(HM_WIDTH, HM_HEIGHT, "height map tool", false, TCOD_RENDERER_OPENGL2);
  guicon = new TCODConsole(HM_WIDTH, HM_HEIGHT);
  guicon->setKeyColor(TCODColor(255, 0, 255));
  Widget::setConsole(guicon);
  int desired_fps = 25;
  initColors();
  buildGui();
  hm = new TCODHeightMap(HM_WIDTH, HM_HEIGHT);
  hm_old = new TCODHeightMap(HM_WIDTH, HM_HEIGHT);
  rnd = new TCODRandom(seed);
  noise = new TCODNoise(2, rnd);
  uint8_t fade = 50;
  bool creditsEnd = false;
  auto timer = tcod::Timer();

  while (true) {
    auto context = TCOD_sys_get_internal_context();
    auto console = TCOD_sys_get_internal_console();
    const float delta_time = timer.sync(desired_fps);
    render(*console, delta_time);
    guicon->setDefaultBackground(TCODColor(255, 0, 255));
    guicon->clear();
    Widget::renderWidgets();
    if (!creditsEnd) {
      creditsEnd = TCOD_console_credits_render_ex(console, HM_WIDTH - 20, HM_HEIGHT - 7, true, delta_time);
    }
    if (Widget::focus) {
      if (fade < 200) fade += 20;
    } else {
      if (fade > 80) fade -= 20;
    }
    TCODConsole::blit(guicon, 0, 0, HM_WIDTH, HM_HEIGHT, TCODConsole::root, 0, 0, fade / 255.0f, fade / 255.0f);
    context->present(*console);

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      SDL_Event tile_event = event;
      context->convert_event_coordinates(tile_event);
      Widget::updateWidgets(tile_event, event);
      switch (event.type) {
        case SDL_QUIT:
          std::exit(EXIT_SUCCESS);
          break;
        case SDL_KEYDOWN:
          switch (event.key.keysym.sym) {
            case SDLK_MINUS:
              (new AddLevelOperation(-(mapmax - mapmin) / 50))->run();
              break;
            case SDLK_EQUALS:
              (new AddLevelOperation((mapmax - mapmin) / 50))->run();
              break;
            case SDLK_PRINTSCREEN:
              context->save_screenshot(nullptr);
              break;
            default:
              break;
          }
          break;
        default:
          break;
      }
    }
  }
  return 0;
}
