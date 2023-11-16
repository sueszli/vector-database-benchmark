// Aseprite Document Library
// Copyright (c) 2001-2016 David Capello
//
// This file is released under the terms of the MIT license.
// Read LICENSE.txt for more information.

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <math.h>

#include "base/base.h"
#include "doc/algo.h"

namespace doc {

// Algorightm from Allegro (allegro/src/gfx.c)
// Adapted for Aseprite by David Capello.
void algo_line(int x1, int y1, int x2, int y2, void *data, AlgoPixel proc)
{
  int dx = x2-x1;
  int dy = y2-y1;
  int i1, i2;
  int x, y;
  int dd;

  /* worker macro */
#define DO_LINE(pri_sign, pri_c, pri_cond, sec_sign, sec_c, sec_cond)   \
  {                                                                     \
    if (d##pri_c == 0) {                                                \
      proc(x1, y1, data);                                               \
      return;                                                           \
    }                                                                   \
                                                                        \
    i1 = 2 * d##sec_c;                                                  \
    dd = i1 - (sec_sign (pri_sign d##pri_c));                           \
    i2 = dd - (sec_sign (pri_sign d##pri_c));                           \
                                                                        \
    x = x1;                                                             \
    y = y1;                                                             \
                                                                        \
    while (pri_c pri_cond pri_c##2) {                                   \
      proc(x, y, data);                                                 \
                                                                        \
      if (dd sec_cond 0) {                                              \
        sec_c sec_sign##= 1;                                            \
        dd += i2;                                                       \
      }                                                                 \
      else                                                              \
        dd += i1;                                                       \
                                                                        \
      pri_c pri_sign##= 1;                                              \
    }                                                                   \
  }

  if (dx >= 0) {
    if (dy >= 0) {
      if (dx >= dy) {
        /* (x1 <= x2) && (y1 <= y2) && (dx >= dy) */
        DO_LINE(+, x, <=, +, y, >=);
      }
      else {
        /* (x1 <= x2) && (y1 <= y2) && (dx < dy) */
        DO_LINE(+, y, <=, +, x, >=);
      }
    }
    else {
      if (dx >= -dy) {
        /* (x1 <= x2) && (y1 > y2) && (dx >= dy) */
        DO_LINE(+, x, <=, -, y, <=);
      }
      else {
        /* (x1 <= x2) && (y1 > y2) && (dx < dy) */
        DO_LINE(-, y, >=, +, x, >=);
      }
    }
  }
  else {
    if (dy >= 0) {
      if (-dx >= dy) {
        /* (x1 > x2) && (y1 <= y2) && (dx >= dy) */
        DO_LINE(-, x, >=, +, y, >=);
      }
      else {
        /* (x1 > x2) && (y1 <= y2) && (dx < dy) */
        DO_LINE(+, y, <=, -, x, <=);
      }
    }
    else {
      if (-dx >= -dy) {
        /* (x1 > x2) && (y1 > y2) && (dx >= dy) */
        DO_LINE(-, x, >=, -, y, <=);
      }
      else {
        /* (x1 > x2) && (y1 > y2) && (dx < dy) */
        DO_LINE(-, y, >=, -, x, <=);
      }
    }
  }
}

/* Helper function for the ellipse drawing routines. Calculates the
   points of an ellipse which fits onto the rectangle specified by x1,
   y1, x2 and y2, and calls the specified routine for each one. The
   output proc has the same format as for do_line, and if the width or
   height of the ellipse is only 1 or 2 pixels, do_line will be
   called.

   Copyright (C) 2002 by Elias Pschernig (eliaspschernig@aon.at)
   for Allegro 4.x.

   Adapted for ASEPRITE by David A. Capello. */
void algo_ellipse(int x1, int y1, int x2, int y2, void *data, AlgoPixel proc)
{
  int mx, my, rx, ry;

  int err;
  int xx, yy;
  int xa, ya;
  int x, y;

  /* Cheap hack to get elllipses with integer diameter, by just offsetting
   * some quadrants by one pixel. */
  int mx2, my2;

  mx = (x1 + x2) / 2;
  mx2 = (x1 + x2 + 1) / 2;
  my = (y1 + y2) / 2;
  my2 = (y1 + y2 + 1) / 2;
  rx = ABS(x1 - x2);
  ry = ABS(y1 - y2);

  if (rx == 1) { algo_line(x2, y1, x2, y2, data, proc); rx--; }
  if (rx == 0) { algo_line(x1, y1, x1, y2, data, proc); return; }

  if (ry == 1) { algo_line(x1, y2, x2, y2, data, proc); ry--; }
  if (ry == 0) { algo_line(x1, y1, x2, y1, data, proc); return; }

  rx /= 2;
  ry /= 2;

  /* Draw the 4 poles. */
  proc(mx, my2 + ry, data);
  proc(mx, my - ry, data);
  proc(mx2 + rx, my, data);
  proc(mx - rx, my, data);

  /* For even diameter axis, double the poles. */
  if (mx != mx2) {
    proc(mx2, my2 + ry, data);
    proc(mx2, my - ry, data);
  }

  if (my != my2) {
    proc(mx2 + rx, my2, data);
    proc(mx - rx, my2, data);
  }

  xx = rx * rx;
  yy = ry * ry;

  /* Do the 'x direction' part of the arc. */

  x = 0;
  y = ry;
  xa = 0;
  ya = xx * 2 * ry;
  err = xx / 4 - xx * ry;

  for (;;) {
    err += xa + yy;
    if (err >= 0) {
      ya -= xx * 2;
      err -= ya;
      y--;
    }
    xa += yy * 2;
    x++;
    if (xa >= ya)
      break;

    proc(mx2 + x, my - y, data);
    proc(mx - x, my - y, data);
    proc(mx2 + x, my2 + y, data);
    proc(mx - x, my2 + y, data);
  }

  /* Fill in missing pixels for very thin ellipses. (This is caused because
   * we always take 1-pixel steps above, and thus might jump past the actual
   * ellipse line.)
   */
  if (y == 0)
    while (x < rx) {
      proc(mx2 + x, my - 1, data);
      proc(mx2 + x, my2 + 1, data);
      proc(mx - x, my - 1, data);
      proc(mx - x, my2 + 1, data);
      x++;
    }

  /* Do the 'y direction' part of the arc. */

  x = rx;
  y = 0;
  xa = yy * 2 * rx;
  ya = 0;
  err = yy / 4 - yy * rx;

  for (;;) {
    err += ya + xx;
    if (err >= 0) {
      xa -= yy * 2;
      err -= xa;
      x--;
    }
    ya += xx * 2;
    y++;
    if (ya > xa)
      break;
    proc(mx2 + x, my - y, data);
    proc(mx - x, my - y, data);
    proc(mx2 + x, my2 + y, data);
    proc(mx - x, my2 + y, data);
  }

  /* See comment above. */
  if (x == 0)
    while (y < ry) {
      proc(mx - 1, my - y, data);
      proc(mx2 + 1, my - y, data);
      proc(mx - 1, my2 + y, data);
      proc(mx2 + 1, my2 + y, data);
      y++;
    }
}

void algo_ellipsefill(int x1, int y1, int x2, int y2, void *data, AlgoHLine proc)
{
  int mx, my, rx, ry;

  int err;
  int xx, yy;
  int xa, ya;
  int x, y;

  /* Cheap hack to get elllipses with integer diameter, by just offsetting
   * some quadrants by one pixel. */
  int mx2, my2;

  mx = (x1 + x2) / 2;
  mx2 = (x1 + x2 + 1) / 2;
  my = (y1 + y2) / 2;
  my2 = (y1 + y2 + 1) / 2;
  rx = ABS (x1 - x2);
  ry = ABS (y1 - y2);

  if (rx == 1) { int c; for (c=y1; c<=y2; c++) proc(x2, c, x2, data); rx--; }
  if (rx == 0) { int c; for (c=y1; c<=y2; c++) proc(x1, c, x1, data); return; }

  if (ry == 1) { proc(x1, y2, x2, data); ry--; }
  if (ry == 0) { proc(x1, y1, x2, data); return; }

  rx /= 2;
  ry /= 2;

  /* Draw the 4 poles. */
  proc(mx, my2 + ry, mx, data);
  proc(mx, my - ry, mx, data);
/*   proc(mx2 + rx, my, mx2 + rx, data); */
/*   proc(mx - rx, my, mx - rx, data); */
  proc(mx - rx, my, mx2 + rx, data);

  /* For even diameter axis, double the poles. */
  if (mx != mx2) {
    proc(mx2, my2 + ry, mx2, data);
    proc(mx2, my - ry, mx2, data);
  }

  if (my != my2) {
/*     proc(mx2 + rx, my2, data); */
/*     proc(mx - rx, my2, data); */
    proc(mx - rx, my2, mx2 + rx, data);
  }

  xx = rx * rx;
  yy = ry * ry;

  /* Do the 'x direction' part of the arc. */

  x = 0;
  y = ry;
  xa = 0;
  ya = xx * 2 * ry;
  err = xx / 4 - xx * ry;

  for (;;) {
    err += xa + yy;
    if (err >= 0) {
      ya -= xx * 2;
      err -= ya;
      y--;
    }
    xa += yy * 2;
    x++;
    if (xa >= ya)
      break;

/*     proc(mx2 + x, my - y, data); */
/*     proc(mx - x, my - y, data); */
/*     proc(mx2 + x, my2 + y, data); */
/*     proc(mx - x, my2 + y, data); */
    proc(mx - x, my - y, mx2 + x, data);
    proc(mx - x, my2 + y, mx2 + x, data);
  }

  /* Fill in missing pixels for very thin ellipses. (This is caused because
   * we always take 1-pixel steps above, and thus might jump past the actual
   * ellipse line.)
   */
  if (y == 0)
    while (x < rx) {
/*       proc(mx2 + x, my - 1, data); */
/*       proc(mx2 + x, my2 + 1, data); */
/*       proc(mx - x, my - 1, data); */
/*       proc(mx - x, my2 + 1, data); */
      x++;
    }

  /* Do the 'y direction' part of the arc. */

  x = rx;
  y = 0;
  xa = yy * 2 * rx;
  ya = 0;
  err = yy / 4 - yy * rx;

  for (;;) {
    err += ya + xx;
    if (err >= 0) {
      xa -= yy * 2;
      err -= xa;
      x--;
    }
    ya += xx * 2;
    y++;
    if (ya > xa)
      break;
/*     proc(mx2 + x, my - y, data); */
/*     proc(mx - x, my - y, data); */
/*     proc(mx2 + x, my2 + y, data); */
/*     proc(mx - x, my2 + y, data); */
    proc(mx - x, my - y, mx2 + x, data);
    proc(mx - x, my2 + y, mx2 + x, data);
  }

  /* See comment above. */
  if (x == 0)
    while (y < ry) {
/*       proc(mx - 1, my - y, data); */
/*       proc(mx2 + 1, my - y, data); */
/*       proc(mx - 1, my2 + y, data); */
/*       proc(mx2 + 1, my2 + y, data); */
      y++;
    }
}

// Algorightm from Allegro (allegro/src/spline.c)
// Adapted for Aseprite by David Capello.
void algo_spline(double x0, double y0, double x1, double y1,
                 double x2, double y2, double x3, double y3,
                 void *data, AlgoLine proc)
{
  int npts;
  int out_x1, out_x2;
  int out_y1, out_y2;

  /* Derivatives of x(t) and y(t). */
  double x, dx, ddx, dddx;
  double y, dy, ddy, dddy;
  int i;

  /* Temp variables used in the setup. */
  double dt, dt2, dt3;
  double xdt2_term, xdt3_term;
  double ydt2_term, ydt3_term;

#define MAX_POINTS   64
#undef DIST
#define DIST(x, y) (sqrt((x) * (x) + (y) * (y)))
  npts = (int)(sqrt(DIST(x1-x0, y1-y0) +
                    DIST(x2-x1, y2-y1) +
                    DIST(x3-x2, y3-y2)) * 1.2);
  if (npts > MAX_POINTS)
    npts = MAX_POINTS;
  else if (npts < 4)
    npts = 4;

  dt = 1.0 / (npts-1);
  dt2 = (dt * dt);
  dt3 = (dt2 * dt);

  xdt2_term = 3 * (x2 - 2*x1 + x0);
  ydt2_term = 3 * (y2 - 2*y1 + y0);
  xdt3_term = x3 + 3 * (-x2 + x1) - x0;
  ydt3_term = y3 + 3 * (-y2 + y1) - y0;

  xdt2_term = dt2 * xdt2_term;
  ydt2_term = dt2 * ydt2_term;
  xdt3_term = dt3 * xdt3_term;
  ydt3_term = dt3 * ydt3_term;

  dddx = 6*xdt3_term;
  dddy = 6*ydt3_term;
  ddx = -6*xdt3_term + 2*xdt2_term;
  ddy = -6*ydt3_term + 2*ydt2_term;
  dx = xdt3_term - xdt2_term + 3 * dt * (x1 - x0);
  dy = ydt3_term - ydt2_term + dt * 3 * (y1 - y0);
  x = x0;
  y = y0;

  out_x1 = (int)x0;
  out_y1 = (int)y0;

  x += .5;
  y += .5;
  for (i=1; i<npts; i++) {
    ddx += dddx;
    ddy += dddy;
    dx += ddx;
    dy += ddy;
    x += dx;
    y += dy;

    out_x2 = (int)x;
    out_y2 = (int)y;

    proc(out_x1, out_y1, out_x2, out_y2, data);

    out_x1 = out_x2;
    out_y1 = out_y2;
  }
}

double algo_spline_get_y(double x0, double y0, double x1, double y1,
                         double x2, double y2, double x3, double y3,
                         double in_x)
{
  int npts;
  double out_x, old_x;
  double out_y, old_y;

  /* Derivatives of x(t) and y(t). */
  double x, dx, ddx, dddx;
  double y, dy, ddy, dddy;
  int i;

  /* Temp variables used in the setup. */
  double dt, dt2, dt3;
  double xdt2_term, xdt3_term;
  double ydt2_term, ydt3_term;

#define MAX_POINTS   64
#undef DIST
#define DIST(x, y) (sqrt ((x) * (x) + (y) * (y)))
  npts = (int) (sqrt (DIST(x1-x0, y1-y0) +
                      DIST(x2-x1, y2-y1) +
                      DIST(x3-x2, y3-y2)) * 1.2);
  if (npts > MAX_POINTS)
    npts = MAX_POINTS;
  else if (npts < 4)
    npts = 4;

  dt = 1.0 / (npts-1);
  dt2 = (dt * dt);
  dt3 = (dt2 * dt);

  xdt2_term = 3 * (x2 - 2*x1 + x0);
  ydt2_term = 3 * (y2 - 2*y1 + y0);
  xdt3_term = x3 + 3 * (-x2 + x1) - x0;
  ydt3_term = y3 + 3 * (-y2 + y1) - y0;

  xdt2_term = dt2 * xdt2_term;
  ydt2_term = dt2 * ydt2_term;
  xdt3_term = dt3 * xdt3_term;
  ydt3_term = dt3 * ydt3_term;

  dddx = 6*xdt3_term;
  dddy = 6*ydt3_term;
  ddx = -6*xdt3_term + 2*xdt2_term;
  ddy = -6*ydt3_term + 2*ydt2_term;
  dx = xdt3_term - xdt2_term + 3 * dt * (x1 - x0);
  dy = ydt3_term - ydt2_term + dt * 3 * (y1 - y0);
  x = x0;
  y = y0;

  out_x = old_x = x0;
  out_y = old_y = y0;

  x += .5;
  y += .5;
  for (i=1; i<npts; i++) {
    ddx += dddx;
    ddy += dddy;
    dx += ddx;
    dy += ddy;
    x += dx;
    y += dy;

    out_x = x;
    out_y = y;
    if (out_x > in_x) {
      out_y = old_y + (out_y-old_y) * (in_x-old_x) / (out_x-old_x);
      break;
    }
    old_x = out_x;
    old_y = out_y;
  }

  return out_y;
}

double algo_spline_get_tan(double x0, double y0, double x1, double y1,
                           double x2, double y2, double x3, double y3,
                           double in_x)
{
  double out_x, old_x, old_dx, old_dy;
  int npts;

  /* Derivatives of x(t) and y(t). */
  double x, dx, ddx, dddx;
  double y, dy, ddy, dddy;
  int i;

  /* Temp variables used in the setup. */
  double dt, dt2, dt3;
  double xdt2_term, xdt3_term;
  double ydt2_term, ydt3_term;

#define MAX_POINTS   64
#undef DIST
#define DIST(x, y) (sqrt ((x) * (x) + (y) * (y)))
  npts = (int) (sqrt (DIST(x1-x0, y1-y0) +
                      DIST(x2-x1, y2-y1) +
                      DIST(x3-x2, y3-y2)) * 1.2);
  if (npts > MAX_POINTS)
    npts = MAX_POINTS;
  else if (npts < 4)
    npts = 4;

  dt = 1.0 / (npts-1);
  dt2 = (dt * dt);
  dt3 = (dt2 * dt);

  xdt2_term = 3 * (x2 - 2*x1 + x0);
  ydt2_term = 3 * (y2 - 2*y1 + y0);
  xdt3_term = x3 + 3 * (-x2 + x1) - x0;
  ydt3_term = y3 + 3 * (-y2 + y1) - y0;

  xdt2_term = dt2 * xdt2_term;
  ydt2_term = dt2 * ydt2_term;
  xdt3_term = dt3 * xdt3_term;
  ydt3_term = dt3 * ydt3_term;

  dddx = 6*xdt3_term;
  dddy = 6*ydt3_term;
  ddx = -6*xdt3_term + 2*xdt2_term;
  ddy = -6*ydt3_term + 2*ydt2_term;
  dx = xdt3_term - xdt2_term + 3 * dt * (x1 - x0);
  dy = ydt3_term - ydt2_term + dt * 3 * (y1 - y0);
  x = x0;
  y = y0;

  out_x = x0;

  old_x = x0;
  old_dx = dx;
  old_dy = dy;

  x += .5;
  y += .5;
  for (i=1; i<npts; i++) {
    ddx += dddx;
    ddy += dddy;
    dx += ddx;
    dy += ddy;
    x += dx;
    y += dy;

    out_x = x;
    if (out_x > in_x) {
      dx = old_dx + (dx-old_dx) * (in_x-old_x) / (out_x-old_x);
      dy = old_dy + (dy-old_dy) * (in_x-old_x) / (out_x-old_x);
      break;
    }
    old_x = out_x;
    old_dx = dx;
    old_dy = dy;
  }

  return dy / dx;
}

} // namespace doc
