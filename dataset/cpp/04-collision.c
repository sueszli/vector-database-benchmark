// original code by @vurtun (public domain)
// - rlyeh, public domain.

#include <engine.h>

int main(void) {
    // 50% sized, msaa x4 enabled
    window_create(0.50f, WINDOW_MSAA4);

    // camera that points to origin
    camera cam;
    camera_create(&cam, 0.25f, false, true);
    camera_teleport(&cam, vec3(0,30,-30), vec2(90,-45));

    int paused = 0;

    // main loop
    while(window_update()) {

        // key handler
        if (key_down('F11') || (key_down('\n') && (key('LALT') || key('RALT'))) ) { static int fs = 0;
            window_fullscreen( fs^=1 );
        }
        if (key_down('ESC')) break;

        // animation
        static float dx = 0, dy = 0;
        if (key_down(' ')) paused ^= 1;
        float delta = (0.25f / 60.f) * !paused;
        dx = dx + delta * 2.0f;
        dy = dy + delta * 0.8f;

        // camera control & view matrix
        int active = !!mouse('R') || !!mouse('L');
        set_mouse(active ? 'hide' : 'show');
        camera_enable(&cam, active);
        camera_fps(&cam, key_wasdec(), mouse_xy());

        // projection matrix
        float proj[16]; perspective44(proj, 45, window_aspect(), 0.1f, 1000);

        // projview matrix
        mat44 projview; multiply44(projview, proj, cam.view);

        // rendering
        viewport_color(vec3(0.15,0.15,0.15));
        viewport_clear(true, true);
        viewport_clip(vec2(0,0), window_size());

        // debug draw collisions
        vec3 red = {1,0,0}, green = {0,1,0}, blue = {0,0,1}, white = {1,1,1}, yellow = {1,1,0};
        ddraw_begin( projview );
        {
            // 3D
            glEnable(GL_DEPTH_TEST);

            // grid
            ddraw_color3(vec3(0.2f,0.2f,0.2f));
            ddraw_grid(10,10,10);

            // basis-axis
            ddraw_color3(red);
            ddraw_arrow(vec3(0,0,0), vec3(1,0,0));
            ddraw_color3(green);
            ddraw_arrow(vec3(0,0,0), vec3(0,1,0));
            ddraw_color3(blue);
            ddraw_arrow(vec3(0,0,0), vec3(0,0,1));

            {
                // Triangle-Ray Intersection*/
                vec3 ro, rd;
                int suc;

                triangle tri = { vec3(-9,1,28), vec3(-10,0,28), vec3(-11,1,28) };

                // ray
                ro = vec3(-10,-1,20);
                rd = vec3(-10+0.4f*sinf(dx), 2.0f*cosf(dy), 29.81023f);
                rd = sub3(rd, ro);
                rd = norm3(rd);

                ray r = ray(ro, rd);
                hit *hit = ray_hit_triangle(r, tri);
                if (hit) {
                    // point of intersection
                    ddraw_color3(red);
                    ddraw_box(hit->p, vec3(0.10f, 0.10f, 0.10f));

                    // intersection normal
                    ddraw_color3(blue);
                    vec3 v = add3(hit->p, hit->n);
                    ddraw_arrow(hit->p, v);
                }

                // line
                ddraw_color3(red);
                rd = scale3(rd,10);
                rd = add3(ro,rd);
                ddraw_line(ro, rd);

                // triangle
                if (hit) ddraw_color3(red);
                else ddraw_color3(white);
                ddraw_triangle(tri.p0,tri.p1,tri.p2);
            }
            {
                // Plane-Ray Intersection*/
                vec3 ro, rd;
                mat33 rot;

                // ray
                static float d = 0;
                d += delta * 2.0f;
                ro = vec3(0,-1,20);
                rd = vec3(0.1f, 0.5f, 9.81023f);
                rd = sub3(rd, ro);
                rd = norm3(rd);

                // rotation
                rotation33(rot, deg(d), 0,1,0);
                rd = mulv33(rot, rd);

                // intersection
                ray r = ray(ro, rd);
                plane pl = plane(vec3(0,0,28), vec3(0,0,1));
                hit *hit = ray_hit_plane(r, pl);
                if (hit) {
                    // point of intersection
                    ddraw_color3(red);
                    ddraw_box(hit->p, vec3(0.10f, 0.10f, 0.10f));

                    // intersection normal
                    ddraw_color3(blue);
                    vec3 v = add3(hit->p, hit->n);
                    ddraw_arrow(hit->p, v);
                    ddraw_color3(red);
                }
                // line
                ddraw_color3(red);
                rd = scale3(rd,9);
                rd = add3(ro,rd);
                ddraw_line(ro, rd);

                // plane
                if (hit) ddraw_color3(red);
                else ddraw_color3(white);
                ddraw_plane(vec3(0,0,28), vec3(0,0,1), 3.0f);
            }
            {
                // Sphere-Ray Intersection*/
                vec3 ro, rd;
                sphere s;

                // ray
                ro = vec3(0,-1,0);
                rd = vec3(0.4f*sinf(dx), 2.0f*cosf(dy), 9.81023f);
                rd = sub3(rd, ro);
                rd = norm3(rd);

                ray r = ray(ro, rd);
                s = sphere(vec3(0,0,8), 1);
                hit *hit = ray_hit_sphere(r, s);
                if(hit) {
                    // points of intersection
                    vec3 in = add3(ro,scale3(rd,hit->t0));

                    ddraw_color3(green);
                    ddraw_box(in, vec3(0.05f, 0.05f, 0.05f));

                    in = add3(ro,scale3(rd,hit->t1));

                    ddraw_color3(yellow);
                    ddraw_box(in, vec3(0.05f, 0.05f, 0.05f));

                    // intersection normal
                    ddraw_color3(blue);
                    vec3 v = add3(hit->p, hit->n);
                    ddraw_arrow(hit->p, v);
                    ddraw_color3(red);
                }
                // line
                ddraw_color3(red);
                rd = scale3(rd,10);
                rd = add3(ro,rd);
                ddraw_line(ro, rd);

                // sphere
                if (hit) ddraw_color3(red);
                else ddraw_color3(white);
                ddraw_sphere(vec3(0,0,8), 1);
            }
            {   // ray-aabb
                aabb bounds = aabb(vec3(10-0.5f,-0.5f,7.5f), vec3(10.5f,0.5f,8.5f));

                vec3 ro = vec3(10,-1,0);
                vec3 rd = vec3(10+0.4f*sinf(dx), 2.0f*cosf(dy), 9.81023f);
                rd = norm3(sub3(rd, ro));
                ray r = ray(ro, rd);

                hit *hit = ray_hit_aabb(r, bounds);
                if(hit) {
                    // points of intersection
                    vec3 in;
                    in = scale3(rd,hit->t0);
                    in = add3(ro,in);

                    ddraw_color3(red);
                    ddraw_box(in, vec3(0.05f, 0.05f, 0.05f));

                    in = scale3(rd,hit->t1);
                    in = add3(ro,in);

                    ddraw_color3(red);
                    ddraw_box(in, vec3(0.05f, 0.05f, 0.05f));

                    // intersection normal
                    ddraw_color3(blue);
                    vec3 v = add3(hit->p, hit->n);
                    ddraw_arrow(hit->p, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);
                ddraw_box(vec3(10,0,8), vec3(1,1,1));

                // line
                ddraw_color3(red);
                rd = scale3(rd,10);
                rd = add3(ro,rd);
                ddraw_line(ro, rd);
            }
            {
                // Sphere-Sphere intersection*/
                sphere a = sphere(vec3(-10,0,8), 1);
                sphere b = sphere(vec3(-10+0.6f*sinf(dx), 3.0f*cosf(dy),8), 1);
                hit *m = sphere_hit_sphere(a, b);
                if (m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);

                ddraw_sphere(a.c, 1);
                ddraw_sphere(b.c, 1);
            }
            {
                // AABB-AABB intersection*/
                const float x = 10+0.6f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = 20.0f;

                aabb a = aabb(vec3(10-0.5f,-0.5f,20-0.5f), vec3(10+0.5f,0.5f,20.5f));
                aabb b = aabb(vec3(x-0.5f,y-0.5f,z-0.5f), vec3(x+0.5f,y+0.5f,z+0.5f));
                hit *m = aabb_hit_aabb(a, b);
                if(m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);

                ddraw_box(vec3(10,0,20), vec3(1,1,1));
                ddraw_box(vec3(x,y,z), vec3(1,1,1));
            }
            {
                // Capsule-Capsule intersection*/
                const float x = 20+0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = 28.5f;

                capsule a = capsule(vec3(20.0f,-1.0f,28.0f), vec3(20.0f,1.0f,28.0f), 0.2f);
                capsule b = capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z-1.0f), 0.2f);
                hit *m = capsule_hit_capsule(a, b);
                if( m ) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);
                ddraw_capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z-1.0f), 0.2f);
                ddraw_capsule(vec3(20.0f,-1.0f,28.0f), vec3(20.0f,1.0f,28.0f), 0.2f);
            }
            {
                // AABB-Sphere intersection*/
                aabb a = aabb(vec3(20-0.5f,-0.5f,7.5f), vec3(20.5f,0.5f,8.5f));
                sphere s = sphere(vec3(20+0.6f*sinf(dx), 3.0f*cosf(dy),8), 1);
                hit *m  = aabb_hit_sphere(a, s);
                if(m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);

                ddraw_box(vec3(20,0,8), vec3(1,1,1));
                ddraw_sphere(s.c, 1);
            }
            {
                // Sphere-AABB intersection*/
                const float x = 10+0.6f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -8.0f;

                sphere s = sphere(vec3(10,0,-8), 1);
                aabb a = aabb(vec3(x-0.5f,y-0.5f,z-0.5f), vec3(x+0.5f,y+0.5f,z+0.5f));
                hit *m = sphere_hit_aabb(s, a);
                if(m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);

                ddraw_box(vec3(x,y,z), vec3(1,1,1));
                ddraw_sphere(s.c, 1);
            }
            {
                // Capsule-Sphere intersection*/
                capsule c = capsule(vec3(-20.5f,-1.0f,7.5f), vec3(-20+0.5f,1.0f,8.5f), 0.2f);
                sphere b = sphere(vec3(-20+0.6f*sinf(dx), 3.0f*cosf(dy),8), 1);
                hit *m = capsule_hit_sphere(c, b);
                if(m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);
                ddraw_sphere(b.c, 1);
                ddraw_capsule(vec3(-20.5f,-1.0f,7.5f), vec3(-20+0.5f,1.0f,8.5f), 0.2f);
            }
            {
                // Sphere-Capsule intersection*/
                const float x = 20+0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -8;

                sphere s = sphere(vec3(20,0,-8), 1);
                capsule c = capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z-1.0f), 0.2f);
                hit *m = sphere_hit_capsule(s, c);
                if(m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);

                ddraw_capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z-1.0f), 0.2f);
                ddraw_sphere(s.c, 1);
            }
            {
                // Capsule-AABB intersection*/
                const float x = -20+0.6f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = 28.0f;

                capsule c = capsule(vec3(-20.5f,-1.0f,27.5f), vec3(-20+0.5f,1.0f,28.5f), 0.2f);
                aabb b = aabb(vec3(x-0.5f,y-0.5f,z-0.5f), vec3(x+0.5f,y+0.5f,z+0.5f));
                hit *m = capsule_hit_aabb(c, b);
                if(m) {
                    vec3 v;
                    ddraw_color3(blue);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    v = add3(m->contact_point, m->normal);
                    ddraw_arrow(m->contact_point, v);
                    ddraw_color3(red);
                } else ddraw_color3(white);
                ddraw_box(vec3(x,y,z), vec3(1,1,1));
                ddraw_capsule(vec3(-20.5f,-1.0f,27.5f), vec3(-20+0.5f,1.0f,28.5f), 0.2f);
            }
            {
                // AABB-Capsule intersection*/
                const float x = 0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -8;

                aabb a = aabb(vec3(-0.5f,-0.5f,-8.5f), vec3(0.5f,0.5f,-7.5f));
                capsule c = capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z-1.0f), 0.2f);
                hit *m = aabb_hit_capsule(a, c);
                if(m) {
                    ddraw_color3(red);
                    ddraw_box(m->contact_point, vec3(0.05f, 0.05f, 0.05f));
                    ddraw_arrow(m->contact_point, add3(m->contact_point, m->normal));
                } else ddraw_color3(white);

                ddraw_capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z-1.0f), 0.2f);
                ddraw_box(vec3(0,0,-8.0f), vec3(1,1,1));
            }
            {
                // poly(Pyramid)-Sphere (GJK) intersection*/
                sphere s = sphere(vec3(-10+0.6f*sinf(dx), 3.0f*cosf(dy),-8), 1);
                poly pyr = pyramid(vec3(-10.5f,-0.5f,-7.5f), vec3(-10.5f,1.0f,-7.5f), 1.0f);

                gjk_result gjk;
                if (poly_hit_sphere(&gjk, pyr, s))
                    ddraw_color3(red);
                else ddraw_color3(white);

                ddraw_sphere(s.c, 1);
                ddraw_pyramid(vec3(-10.5f,-0.5f,-7.5f), vec3(-10.5f,1.0f,-7.5f), 1.0f);

                poly_free(&pyr);

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }
            {
                // poly(Diamond)-Sphere (GJK) intersection*/

                sphere s = sphere(vec3(-20+0.6f*sinf(dx), 3.0f*cosf(dy),-8), 1);
                poly dmd = diamond(vec3(-20.5f,-0.5f,-7.5f), vec3(-20.5f,1.0f,-7.5f), 0.5f);

                gjk_result gjk;
                if (poly_hit_sphere(&gjk, dmd, s))
                    ddraw_color3(red);
                else ddraw_color3(white);

                ddraw_sphere(s.c, 1);
                ddraw_diamond(vec3(-20.5f,-0.5f,-7.5f), vec3(-20.5f,1.0f,-7.5f), 0.5f);

                poly_free(&dmd);

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }
            {
                // poly(Pyramid)-Capsule (GJK) intersection*/

                const float x = 0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -15;

                capsule c = capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z), 0.2f);
                poly pyr = pyramid(vec3(-0.5f,-0.5f,-15.5f), vec3(-0.5f,1.0f,-15.5f), 1.0f);

                gjk_result gjk;
                if (poly_hit_capsule(&gjk, pyr, c))
                    ddraw_color3(red);
                else ddraw_color3(white);

                ddraw_capsule(c.a, c.b, c.r);
                ddraw_pyramid(vec3(-0.5f,-0.5f,-15.5f), vec3(-0.5f,1.0f,-15.5f), 1.0f);

                poly_free(&pyr);

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }

            {
                // poly(Diamond)-Capsule (GJK) intersection*/

                const float x = -10 + 0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -15;

                capsule c = capsule(vec3(x,y-1.0f,z), vec3(x,y+1.0f,z), 0.2f);
                poly dmd = diamond(vec3(-10.5f,-0.5f,-15.5f), vec3(-10.5f,1.0f,-15.5f), 0.5f);

                gjk_result gjk;
                if (poly_hit_capsule(&gjk, dmd, c))
                    ddraw_color3(red);
                else ddraw_color3(white);

                ddraw_capsule(c.a, c.b, c.r);
                ddraw_diamond(vec3(-10.5f,-0.5f,-15.5f), vec3(-10.5f,1.0f,-15.5f), 0.5f);

                poly_free(&dmd);

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }

            {
                // poly(Diamond)-poly(Pyramid) (GJK) intersection*/

                const float x = -20 + 0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -15;

                poly pyr = pyramid(vec3(x,y-0.5f,z), vec3(x,y+1,z), 0.8f);
                poly dmd = diamond(vec3(-20.5f,-0.5f,-15.5f), vec3(-20.5f,1.0f,-15.5f), 0.5f);

                gjk_result gjk;
                if (poly_hit_poly(&gjk, dmd, pyr))
                    ddraw_color3(red);
                else ddraw_color3(white);

                ddraw_pyramid(vec3(x,y-0.5f,z), vec3(x,y+1,z), 0.8f);
                ddraw_diamond(vec3(-20.5f,-0.5f,-15.5f), vec3(-20.5f,1.0f,-15.5f), 0.5f);

                poly_free(&dmd);
                poly_free(&pyr);

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }
            {
                // poly(Pyramid)-poly(Diamond) (GJK) intersection*/

                const float x = 10 + 0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -15;

                poly dmd = diamond(vec3(x,y-0.5f,z), vec3(x,y+1,z), 0.5f);
                poly pyr = pyramid(vec3(10.5f,-0.5f,-15.5f), vec3(10.5f,1.0f,-15.5f), 1.0f);

                gjk_result gjk;
                if (poly_hit_poly(&gjk, dmd, pyr))
                    ddraw_color3(red);
                else ddraw_color3(white);

                ddraw_diamond(vec3(x,y-0.5f,z), vec3(x,y+1,z), 0.5f);
                ddraw_pyramid(vec3(10.5f,-0.5f,-15.5f), vec3(10.5f,1.0f,-15.5f), 1.0f);

                poly_free(&dmd);
                poly_free(&pyr);

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }
            {
                // poly(Diamond)-AABB (GJK) intersection*/

                const float x = 20 + 0.4f*sinf(dx);
                const float y = 3.0f*cosf(dy);
                const float z = -15;

                poly dmd = diamond(vec3(x,y-0.5f,z), vec3(x,y+1,z), 0.5f);
                aabb a = aabb(vec3(19.5f,-0.5f,-14.5f), vec3(20.5f,0.5f,-15.5f));

                gjk_result gjk;
                if (poly_hit_aabb(&gjk, dmd, a))
                    ddraw_color3(red);
                else ddraw_color3(white);

                poly_free(&dmd);

                ddraw_diamond(vec3(x,y-0.5f,z), vec3(x,y+1,z), 0.5f);
                ddraw_box(vec3(20,0,-15), vec3(1,1,1));

                ddraw_box(gjk.p0, vec3(0.05f, 0.05f, 0.05f));
                ddraw_box(gjk.p1, vec3(0.05f, 0.05f, 0.05f));
                ddraw_line(gjk.p0, gjk.p1);
            }
        }

        ddraw_end();

        ddraw_printf( window_stats() );
        ddraw_printf( "space - pause simulation");
        window_swap(0);
    }
}
