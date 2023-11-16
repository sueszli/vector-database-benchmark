#include "emboxvccursor.h"
#include <QtCore/qdebug.h>

#include <QtGui/QImageReader>
#include <QtGui/QRgb>

QT_BEGIN_NAMESPACE

static unsigned char *__calculateCursorLocation(struct fb_info *fb, int x, int y);

QEmboxCursor::QEmboxCursor()
: mouseX(0), mouseY(0), inited(0)
{
	cursor.load(QString(":/def_cur.png"));
	cursor = cursor.convertToFormat(QImage::Format_RGB16);
	cursor_H = cursor.height();
	cursor_W = cursor.width();

	/* 4 is the upper bound for bytes per pixel */
	dirtyRect = new unsigned char[cursor_H * cursor_W * 4];
}

QEmboxCursor::~QEmboxCursor() {
	delete [] dirtyRect;
}

void QEmboxCursor::emboxCursorReset(struct fb_info *fb) {
	if (imageChanged(fb, __calculateCursorLocation(fb, mouseX, mouseY))) {
		storeDirtyRect(fb,  __calculateCursorLocation(fb, mouseX, mouseY));
	}
}

void QEmboxCursor::emboxCursorRedraw(struct fb_info *fb, int x, int y) {
	if (inited) {
		flushDirtyRect(fb, __calculateCursorLocation(fb, mouseX, mouseY));
	} else {
		inited = 1;
	}
	storeDirtyRect(fb,  __calculateCursorLocation(fb, x, y));
	drawCursor(fb, __calculateCursorLocation(fb, x, y));

	mouseX = x;
	mouseY = y;
}

static unsigned char *__calculateCursorLocation(struct fb_info *fb, int x, int y) {
	return fb->screen_base + (y * fb->var.xres + x) * 2;
}

int QEmboxCursor::imageChanged(struct fb_info *fb, unsigned char *begin) {
    int shift, i, ret = 0;

    int bpp = fb->var.bits_per_pixel / 8;

    for (i = 0, shift = 0; i < cursor_H; i++ , shift += fb->var.xres * bpp) {
    	for (int j = 0; j < cursor_W * bpp; j++) {
    		if (*(dirtyRect + i * cursor_W * bpp + j) != *(begin + shift + j)) {
    			ret = 1;
    			break;
    		}
    	}
    }

    return ret;
}


void QEmboxCursor::drawCursor(struct fb_info *fb, unsigned char *begin) {
    int shift, i;
    int bpp = fb->var.bits_per_pixel / 8;

    for (i = 0, shift = 0; i < cursor.height(); i++ , shift += fb->var.xres * bpp) {
    	memcpy(begin + shift, (const void *)cursor.constScanLine(i), cursor.bytesPerLine());
    }

     prepareCursor(fb, begin);
}

void QEmboxCursor::prepareCursor(struct fb_info *fb, unsigned char *begin) {
    int shift, i;

    int bpp = fb->var.bits_per_pixel / 8;

    for (i = 0, shift = 0; i < cursor_H; i++ , shift += fb->var.xres * bpp) {
    	for (int j = 0; j < cursor_W * bpp; j++) {
            QRgb rgb = cursor.pixel(j / bpp, i);
            if (!(qRed(rgb) + qGreen(rgb) + qBlue(rgb))) {
                *(begin + shift + j) = *(dirtyRect + i * cursor_W * bpp + j);
            }
    	}
    }

}

void QEmboxCursor::storeDirtyRect(struct fb_info *fb, unsigned char *begin) {
    int shift, i;

    int bpp = fb->var.bits_per_pixel / 8;

    for (i = 0, shift = 0; i < cursor_H; i++ , shift += fb->var.xres * bpp) {
    	memcpy(dirtyRect + i * cursor_W * bpp, (const void *)(begin + shift), cursor_W * bpp);
    }
}

void QEmboxCursor::flushDirtyRect(struct fb_info *fb, unsigned char *begin) {
    int shift, i;

    int bpp = fb->var.bits_per_pixel / 8;

    for (i = 0, shift = 0; i < cursor_H; i++, shift += fb->var.xres * bpp) {
    	memcpy(begin + shift, (const void *)(dirtyRect + i * cursor_W * bpp), cursor_W * bpp);
    }
}

QT_END_NAMESPACE
