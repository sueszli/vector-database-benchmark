//
// Created by XingfengYang on 2020/7/7.
//

#include "libgui/gui_window.h"
#include "kernel/log.h"
#include "libc/stdlib.h"
#include "libgfx/gfx2d.h"
#include "libgfx/font8bits.h"
#include "libgui/gui_button.h"
#include "libgui/gui_canvas.h"
#include "libgui/gui_container.h"
#include "libgui/gui_label.h"
#include "libgui/gui_panel.h"
#include "libgui/gui_view3d.h"

extern uint32_t GFX2D_BUFFER[1024 * 768];

void gui_window_create(GUIWindow *window) {
    window->component.type = WINDOW;
    window->component.visable = true;
    window->component.colorMode = RGB;
    window->component.node.next = nullptr;
    window->component.node.prev = nullptr;
    kvector_allocate(&window->children);
    window->component.position.x = 0;
    window->component.position.y = 0;

    window->component.size.height = DEFAULT_WINDOW_HEIGHT;
    window->component.size.width = DEFAULT_WINDOW_WIDTH;

    window->component.padding.top = DEFAULT_PADDING;
    window->component.padding.bottom = DEFAULT_PADDING;
    window->component.padding.left = DEFAULT_PADDING;
    window->component.padding.right = DEFAULT_PADDING;

    window->component.margin.top = DEFAULT_MARGIN;
    window->component.margin.bottom = DEFAULT_MARGIN;
    window->component.margin.left = DEFAULT_MARGIN;
    window->component.margin.right = DEFAULT_MARGIN;

    window->component.background.a = (FLUENT_PRIMARY_BACK_COLOR >> 24) & 0xFF;
    window->component.background.r = (FLUENT_PRIMARY_BACK_COLOR >> 16) & 0xFF;
    window->component.background.g = (FLUENT_PRIMARY_BACK_COLOR >> 8) & 0xFF;
    window->component.background.b = FLUENT_PRIMARY_BACK_COLOR & 0xFF;

    window->component.foreground.a = (FLUENT_PRIMARY_FORE_COLOR >> 24) & 0xFF;
    window->component.foreground.r = (FLUENT_PRIMARY_FORE_COLOR >> 16) & 0xFF;
    window->component.foreground.g = (FLUENT_PRIMARY_FORE_COLOR >> 8) & 0xFF;
    window->component.foreground.b = FLUENT_PRIMARY_FORE_COLOR & 0xFF;

    window->component.boxShadow.color.a = (FLUENT_PRIMARY_COLOR >> 24) & 0xFF;
    window->component.boxShadow.color.r = (FLUENT_PRIMARY_COLOR >> 16) & 0xFF;
    window->component.boxShadow.color.g = (FLUENT_PRIMARY_COLOR >> 8) & 0xFF;
    window->component.boxShadow.color.b = FLUENT_PRIMARY_COLOR & 0xFF;
    window->component.boxShadow.width = 5;

    window->header.background.a = (FLUENT_PRIMARY_COLOR >> 24) & 0xFF;
    window->header.background.r = (FLUENT_PRIMARY_COLOR >> 16) & 0xFF;
    window->header.background.g = (FLUENT_PRIMARY_COLOR >> 8) & 0xFF;
    window->header.background.b = FLUENT_PRIMARY_COLOR & 0xFF;

    window->header.foreground.a = (FLUENT_PRIMARY_FORE_COLOR >> 24) & 0xFF;
    window->header.foreground.r = (FLUENT_PRIMARY_FORE_COLOR >> 16) & 0xFF;
    window->header.foreground.g = (FLUENT_PRIMARY_FORE_COLOR >> 8) & 0xFF;
    window->header.foreground.b = FLUENT_PRIMARY_FORE_COLOR & 0xFF;

    window->component.boxShadow.enable = false;
    window->isWindowNeedUpdate = true;
    window->isShadowNeedUpdate = true;

    window->title = "";

    if (window->children.data == nullptr) {
        LogError("[GUI]: window create failed, unable to allocate children vector\n");
    }

    gfx2d_create_surface(&window->surface, 1024, 768, GFX2D_BUFFER);
}

void gui_window_init(GUIWindow *window, uint32_t x, uint32_t y, const char *title) {
    window->component.position.x = x;
    window->component.position.y = y;

    window->title = title;
}

void gui_window_add_children(GUIWindow *window, GUIComponent *component) {
    if (window->children.data != nullptr) {
        window->children.operations.add(&window->children, &component->node);
    }
}

void gui_window_draw(GUIWindow *window) {
    if (window->component.visable) {
        if (window->isWindowNeedUpdate) {
            if (window->component.colorMode == RGB) {
                // 1. draw_background
                window->surface.operations.fillRect(&window->surface, window->component.position.x,
                                                    window->component.position.y,
                                                    window->component.position.x + window->component.size.width,
                                                    window->component.position.y + window->component.size.height +
                                                    DEFAULT_WINDOW_HEADER_HEIGHT,
                                                    window->header.foreground.a << 24 |
                                                    window->component.background.r << 16 |
                                                    window->component.background.g << 8 |
                                                    window->component.background.b);
            }

            // 2. draw header
            window->surface.operations.fillRect(&window->surface, window->component.position.x,
                                                window->component.position.y,
                                                window->component.position.x + window->component.size.width,
                                                window->component.position.y + DEFAULT_WINDOW_HEADER_HEIGHT,
                                                window->header.foreground.a << 24 | window->header.background.r << 16 |
                                                window->header.background.g << 8 | window->header.background.b);

            uint16_t *bitmap = win_app_16_bits();
            for (uint32_t i = 0; i < 16; i++) {
                for (uint32_t j = 0; j < 16; j++) {
                    if ((bitmap[i] & (0x1 << j)) > 0) {
                        window->surface.operations.drawPixelColor(&window->surface,
                                                                  window->component.position.x + j + DEFAULT_PADDING,
                                                                  window->component.position.y + i + DEFAULT_PADDING +
                                                                  4,
                                                                  window->header.foreground.r << 16 |
                                                                  window->header.foreground.g << 8 |
                                                                  window->header.foreground.b);
                    }
                }
            }

            // 3. draw_font
            char *tmp = window->title;
            uint32_t xOffset = 2;
            while (*tmp) {
                window->surface.operations.drawAscii(&window->surface,
                                                     window->component.position.x + xOffset * DEFAULT_FONT_SIZE +
                                                     2 * DEFAULT_PADDING,
                                                     window->component.position.y + 2 * DEFAULT_PADDING, *tmp,
                                                     window->header.foreground.r << 16 |
                                                     window->header.foreground.g << 8 | window->header.foreground.b);
                xOffset++;
                tmp++;
            }

            // 4. draw header window
            uint16_t *minBitmap = win_min_16_bits();
            for (uint32_t i = 0; i < 16; i++) {
                for (uint32_t j = 0; j < 16; j++) {
                    if ((minBitmap[i] & (0x1 << j)) > 0) {
                        window->surface.operations.drawPixelColor(&window->surface, window->component.position.x + j +
                                                                                    window->component.size.width -
                                                                                    24 * 3,
                                                                  window->component.position.y + i + DEFAULT_PADDING +
                                                                  4,
                                                                  window->header.foreground.r << 16 |
                                                                  window->header.foreground.g << 8 |
                                                                  window->header.foreground.b);
                    }
                }
            }

            uint16_t *maxBitmap = win_max_16_bits();
            for (uint32_t i = 0; i < 16; i++) {
                for (uint32_t j = 0; j < 16; j++) {
                    if ((maxBitmap[i] & (0x1 << j)) > 0) {
                        window->surface.operations.drawPixelColor(&window->surface, window->component.position.x + j +
                                                                                    window->component.size.width -
                                                                                    24 * 2,
                                                                  window->component.position.y + i + DEFAULT_PADDING +
                                                                  4,
                                                                  window->header.foreground.r << 16 |
                                                                  window->header.foreground.g << 8 |
                                                                  window->header.foreground.b);
                    }
                }
            }

            uint16_t *closeBitmap = win_close_16_bits();
            for (uint32_t i = 0; i < 16; i++) {
                for (uint32_t j = 0; j < 16; j++) {
                    if ((closeBitmap[i] & (0x1 << j)) > 0) {
                        window->surface.operations.drawPixelColor(&window->surface, window->component.position.x + j +
                                                                                    window->component.size.width - 24,
                                                                  window->component.position.y + i + DEFAULT_PADDING +
                                                                  4,
                                                                  window->header.foreground.r << 16 |
                                                                  window->header.foreground.g << 8 |
                                                                  window->header.foreground.b);
                    }
                }
            }
        }

        if (window->component.boxShadow.enable && window->isShadowNeedUpdate) {
            // left
            for (uint32_t i = 1; i < window->component.boxShadow.width; i++) {

                //         y=sqrt({250^{2}-({x-250})^{2}})
                uint32_t alpha = (0xff / window->component.boxShadow.width) * i + i * i;
                if (alpha > 0xFF) {
                    alpha = 0xFF;
                }
                window->surface.operations.fillRect(&window->surface, window->component.position.x - i,
                                                    window->component.position.y - i,
                                                    window->component.position.x - (i - 1),
                                                    window->component.position.y + window->component.size.height +
                                                    DEFAULT_WINDOW_HEADER_HEIGHT + i,
                                                    window->component.boxShadow.color.r << 16 |
                                                    window->component.boxShadow.color.g << 8 |
                                                    window->component.boxShadow.color.b | alpha << 24);
            }

            // right
            for (uint32_t i = 0; i < window->component.boxShadow.width; i++) {
                uint32_t alpha = (0xff / window->component.boxShadow.width) * i + i * i;
                if (alpha > 0xFF) {
                    alpha = 0xFF;
                }
                window->surface.operations.fillRect(&window->surface,
                                                    window->component.position.x + window->component.size.width + i,
                                                    window->component.position.y - i,
                                                    window->component.position.x + window->component.size.width + i + 1,
                                                    window->component.position.y + window->component.size.height +
                                                    DEFAULT_WINDOW_HEADER_HEIGHT + i,
                                                    window->component.boxShadow.color.r << 16 |
                                                    window->component.boxShadow.color.g << 8 |
                                                    window->component.boxShadow.color.b | alpha << 24);
            }

            // bottom
            for (uint32_t i = 0; i < window->component.boxShadow.width; i++) {
                uint32_t alpha = (0xff / window->component.boxShadow.width) * i + i * i;
                if (alpha > 0xFF) {
                    alpha = 0xFF;
                }
                window->surface.operations.fillRect(&window->surface, window->component.position.x - i,
                                                    window->component.position.y + window->component.size.height +
                                                    DEFAULT_WINDOW_HEADER_HEIGHT + i,
                                                    window->component.position.x + window->component.size.width + i,
                                                    window->component.position.y + window->component.size.height +
                                                    DEFAULT_WINDOW_HEADER_HEIGHT + i + 1,
                                                    window->component.boxShadow.color.r << 16 |
                                                    window->component.boxShadow.color.g << 8 |
                                                    window->component.boxShadow.color.b | alpha << 24);
            }

            // top
            for (uint32_t i = 1; i < window->component.boxShadow.width; i++) {
                uint32_t alpha = (0xff / window->component.boxShadow.width) * i + i * i;
                if (alpha > 0xFF) {
                    alpha = 0xFF;
                }
                window->surface.operations.fillRect(&window->surface, window->component.position.x - i,
                                                    window->component.position.y - i,
                                                    window->component.position.x + window->component.size.width + i,
                                                    window->component.position.y - (i - 1),
                                                    window->component.boxShadow.color.r << 16 |
                                                    window->component.boxShadow.color.g << 8 |
                                                    window->component.boxShadow.color.b | alpha << 24);
            }
            window->isShadowNeedUpdate = false;
        }

        // 6. draw children
        gui_window_draw_children(window);

        // 7. register click event

        // 7. register drag event
    }
}

void gui_window_draw_children(GUIWindow *window) {
    KernelVector children = window->children;
    if (children.data != nullptr) {
        for (uint32_t i = 0; i < children.size; i++) {
            ListNode *listNode = children.data[i];
            GUIComponent *component = getNode(listNode, GUIComponent, node);
            switch (component->type) {
                case BUTTON: {
                    GUIButton *button = getNode(component, GUIButton, component);
                    button->component.position.x = button->component.position.x + window->component.position.x +
                                                   window->component.padding.left;
                    button->component.position.y =
                            button->component.position.y + window->component.position.y + DEFAULT_WINDOW_HEADER_HEIGHT +
                            window->component.padding.top;
                    gui_button_draw(button);
                    break;
                }

                case LABEL: {
                    GUILabel *label = getNode(component, GUILabel, component);
                    label->component.position.x =
                            label->component.position.x + window->component.position.x + window->component.padding.left;
                    label->component.position.y =
                            label->component.position.y + window->component.position.y + DEFAULT_WINDOW_HEADER_HEIGHT +
                            window->component.padding.top;
                    gui_label_draw(label);
                    break;
                }

                case CANVAS: {
                    GUICanvas *canvas = getNode(component, GUICanvas, component);
                    canvas->component.position.x = window->component.position.x + window->component.padding.left;
                    canvas->component.position.y =
                            window->component.position.y + DEFAULT_WINDOW_HEADER_HEIGHT + window->component.padding.top;
                    gui_canvas_draw(canvas);
                    break;
                }

                case VIEW3D: {
                    GUIView3D *view = getNode(component, GUIView3D, component);
                    view->component.position.x = window->component.position.x + window->component.padding.left;
                    view->component.position.y =
                            window->component.position.y + DEFAULT_WINDOW_HEADER_HEIGHT + window->component.padding.top;
                    gui_view3d_draw(view);
                    break;
                }

                case PANEL: {
                    GUIPanel *innerPanel = getNode(component, GUIPanel, component);
                    innerPanel->component.position.x = innerPanel->component.position.x + window->component.position.x +
                                                       window->component.padding.left;
                    innerPanel->component.position.y = innerPanel->component.position.y + window->component.position.y +
                                                       DEFAULT_WINDOW_HEADER_HEIGHT + window->component.padding.top;
                    gui_panel_draw(innerPanel);
                    break;
                }

                case CONTAINER: {
                    GUIContainer *innerContainer = getNode(component, GUIContainer, component);
                    innerContainer->component.position.x =
                            innerContainer->component.position.x + window->component.position.x +
                            window->component.padding.left;
                    innerContainer->component.position.y =
                            innerContainer->component.position.y + window->component.position.y +
                            DEFAULT_WINDOW_HEADER_HEIGHT + window->component.padding.top;
                    gui_container_draw(innerContainer);
                    break;
                }
                default:
                    break;
            }
        }
    }
}
