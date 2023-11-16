#include "../../platform/agp/glfun.h"

static void surf_destroy(struct wl_client* cl, struct wl_resource* res)
{
	trace(TRACE_ALLOC, "destroy:surf(%"PRIxPTR")", (uintptr_t) res);
	struct comp_surf* surf = wl_resource_get_user_data(res);
	if (!surf){
		trace(TRACE_ALLOC, "destroy:lost-surface");
		return;
	}

/* check pending subsurfaces? */
	destroy_comp_surf(surf, true);
}

static void buffer_destroy(struct wl_listener* list, void* data)
{
	struct comp_surf* surf = NULL;
	surf = wl_container_of(list, surf, l_bufrem);
	if (!surf)
		return;

	trace(TRACE_SURF, "(event) destroy:buffer(%"PRIxPTR")", (uintptr_t) data);

	if (surf->buf){
		surf->cbuf = (uintptr_t) NULL;
		surf->buf = NULL;
	}

	if (surf->l_bufrem_a){
		surf->l_bufrem_a = false;
		wl_list_remove(&surf->l_bufrem.link);
	}
}

/*
 * Buffer now belongs to surface, but it is useless until there's a commit
 */
static void surf_attach(struct wl_client* cl,
	struct wl_resource* res, struct wl_resource* buf, int32_t x, int32_t y)
{
	struct comp_surf* surf = wl_resource_get_user_data(res);
	if (!surf){
		trace(TRACE_SURF, "attempted attach to missing surface\n");
		return;
	}

	if (surf->l_bufrem_a){
		surf->l_bufrem_a = false;
		wl_list_remove(&surf->l_bufrem.link);
	}

	trace(TRACE_SURF, "attach to: %s, @x,y: %d, %d - buf: %"
		PRIxPTR, surf->tracetag, (int)x, (int)y, (uintptr_t)buf);

	bool changed = false;
	if (surf->buf && !buf){
		trace(TRACE_SURF, "mark visible: %s", surf->tracetag);
		surf->viewport.ext.viewport.invisible = true;
		arcan_shmif_enqueue(&surf->acon, &surf->viewport);
		changed = true;
	}

/* a note for subsurfaces here - so these can have hierarchies that are mixed:
 *
 *  a (surf)
 *    b (subsurf, null buffer)
 *      c (subsurf, buffer)
 *
 *  some toolkits use this for intermediate positioning (guess OO + rel anchors)
 *  in principle the code here is correct, i.e. the subsurface becomes invisible
 *  but the idiomatic way of implementing on wm side is by linking the subsurfaces
 *  to the surfaces, and hide on viewport marking them invisible.
 *
 *  the other option is to suck it up as we pay the allocation price anyhow to
 *  get the 'right' size and commit full-translucent non-decorated in the case
 *  of non-buf on subsurface (if we do it on a normal surface we get problems
 *  with clients that keep toplevels around and hide them ..
 */
	else if (surf->viewport.ext.viewport.invisible){
		trace(TRACE_SURF, "mark visible: %s", surf->tracetag);
		surf->viewport.ext.viewport.invisible = false;
		arcan_shmif_enqueue(&surf->acon, &surf->viewport);
		changed = true;
	}

	if (buf){
		surf->l_bufrem_a = true;
		surf->l_bufrem.notify = buffer_destroy;
		wl_resource_add_destroy_listener(buf, &surf->l_bufrem);
	}
/* follow up on the explanation above, push a fully translucent buffer */
	else if (surf->is_subsurface && changed){
		surf->acon.hints |= SHMIF_RHINT_IGNORE_ALPHA;
		for (size_t y = 0; y < surf->acon.h; y++)
			memset(&surf->acon.vidb[y * surf->acon.stride], '\0', surf->acon.stride);
		arcan_shmif_dirty(&surf->acon, 0, 0, surf->acon.w, surf->acon.h, 0);
		arcan_shmif_signal(&surf->acon, SHMIF_SIGVID | SHMIF_SIGBLK_NONE);
	}

/* buf XOR cookie == cbuf in commit */
	surf->cbuf = (uintptr_t) buf;
	surf->buf = (void*) ((uintptr_t) buf ^ ((uintptr_t) 0xfeedface));
}

/*
 * Similar to the X damage stuff, just grow the synch region for shm repacking
 * but there's more to this (of course there is) as there's the whole buffer
 * isn't necessarily 1:1 of surface.
 */
static void surf_damage(struct wl_client* cl,
	struct wl_resource* res, int32_t x, int32_t y, int32_t w, int32_t h)
{
	struct comp_surf* surf = wl_resource_get_user_data(res);

	x *= surf->scale;
	y *= surf->scale;
	w *= surf->scale;
	h *= surf->scale;

	if (x < 0)
		x = 0;

	if (y < 0)
		y = 0;

	if (w < 0)
		w = surf->acon.w;

	if (h < 0)
		h = surf->acon.h;

	trace(TRACE_SURF,"%s:(%"PRIxPTR") @x,y+w,h(%d+%d, %d+%d)",
		surf->tracetag, (uintptr_t)res, (int)x, (int)w, (int)y, (int)h);

	arcan_shmif_dirty(&surf->acon, x, y, x+w, y+h, 0);
}

/*
 * The client wants this object to be signalled when it is time to produce a
 * new frame. There's a few options:
 * - CLOCKREQ and attach it to frame
 * - signal immediately, but defer if we're invisible and wait for DISPLAYHINT.
 * - set a FUTEX/KQUEUE to monitor the segment vready flag, and when
 *   that triggers, send the signal.
 * - enable the frame-feedback mode on shmif.
 */
static void surf_frame(
	struct wl_client* cl, struct wl_resource* res, uint32_t cb)
{
	struct comp_surf* surf = wl_resource_get_user_data(res);
	trace(TRACE_SURF, "req-cb, %s(%"PRIu32")", surf->tracetag, cb);

	if (surf->frames_pending + surf->subsurf_pending > COUNT_OF(surf->scratch)){
		trace(TRACE_ALLOC, "too many pending surface ops");
		wl_resource_post_no_memory(res);
		return;
	}

	struct wl_resource* cbres =
		wl_resource_create(cl, &wl_callback_interface, 1, cb);

	if (!cbres){
	trace(TRACE_ALLOC, "frame callback allocation failed");
		wl_resource_post_no_memory(res);
		return;
	}

/* special case, if the surface has not yet been promoted to a usable type
 * and the client requests a callback, ack:it immediately */
	if (!surf->shell_res){
		trace(TRACE_SURF, "preemptive-cb-ack");
		wl_callback_send_done(cbres, arcan_timemillis());
		wl_resource_destroy(cbres);
		return;
	}

/* should just bitmap this .. */
	for (size_t i = 0; i < COUNT_OF(surf->scratch); i++){
		if (surf->scratch[i].type == 1){
			wl_callback_send_done(surf->scratch[i].res, surf->scratch[i].id);
			wl_resource_destroy(surf->scratch[i].res);
			surf->frames_pending--;
			surf->scratch[i].res = NULL;
			surf->scratch[i].id = 0;
			surf->scratch[i].type = 0;
		}

		if (surf->scratch[i].type == 0){
			surf->frames_pending++;
			surf->scratch[i].res = cbres;
			surf->scratch[i].id = cb;
			surf->scratch[i].type = 1;
			break;
		}
	}
}

static bool shm_to_gl(
	struct arcan_shmif_cont* acon, struct comp_surf* surf,
	int w, int h, int fmt, void* data, int stride)
{
/* globally rejected or per-window rejected or no GL or it has failed before */
	if (!arcan_shmif_handle_permitted(&wl.control) ||
		!arcan_shmif_handle_permitted(acon) ||
		arcan_shmifext_isext(&wl.control) != 1 ||
		surf->shm_gl_fail)
		return false;

	int gl_fmt = -1;
	int px_fmt = GL_UNSIGNED_BYTE;
	int gl_int_fmt = -1;
	int pitch = 0;

	switch(fmt){
	case WL_SHM_FORMAT_ARGB8888:
	case WL_SHM_FORMAT_XRGB8888:
		gl_fmt = GL_BGRA_EXT;
/* only gles/gles2 supports int_fmt as GL_BGRA */
		gl_int_fmt = GL_RGBA8;
//		pitch = stride ? stride / 4 : 0;
		pitch = w;
	break;
	case WL_SHM_FORMAT_ABGR8888:
	case WL_SHM_FORMAT_XBGR8888:
		gl_fmt = GL_RGBA;
		gl_int_fmt = GL_RGBA8;
		pitch = stride ? stride / 4 : 0;
	break;
	case WL_SHM_FORMAT_RGB565:
		gl_fmt = GL_RGB;
		gl_int_fmt = GL_RGBA8;
		px_fmt = GL_UNSIGNED_SHORT_5_6_5;
		pitch = stride ? stride / 2 : 0;
	break;
/* for WL_SHM_FORMAT_YUV***, NV12, we need a really complicated dance here with
 * multiple planes, GL_UNSIGNED_BYTE, ... as well as custom unpack shaders and
 * a conversion pass through FBO */
	default:
		return false;
	break;
	}

/* used the shared primary context for allocation, this is also where we can do
 * our color conversion, currently just use the teximage- format */
	struct agp_fenv* fenv = arcan_shmifext_getfenv(&wl.control);
	GLuint glid;
	fenv->gen_textures(1, &glid);
	fenv->bind_texture(GL_TEXTURE_2D, glid);
	fenv->pixel_storei(GL_UNPACK_ROW_LENGTH, pitch);
	fenv->tex_image_2d(GL_TEXTURE_2D,
		0, gl_int_fmt, w, h, 0, gl_fmt, px_fmt, data);
	fenv->pixel_storei(GL_UNPACK_ROW_LENGTH, 0);
	fenv->bind_texture(GL_TEXTURE_2D, 0);

/* this seems to be needed still or the texture contents will be invalid,
 * better still is to have an explicit fence and queue the release of the
 * buffer until the upload is finished */
	fenv->flush();

/* build descriptors */
	int fd;
	size_t stride_out;
	int out_fmt;

	uintptr_t gl_display;
	arcan_shmifext_egl_meta(&wl.control, &gl_display, NULL, NULL);

/* can still fail for mysterious reasons, force-fallback to normal shm */
	if (!arcan_shmifext_gltex_handle(&wl.control,
		gl_display, glid, &fd, &stride_out, &out_fmt)){
		trace(TRACE_SURF, "shm->glhandle failed");
		fenv->delete_textures(1, &glid);
		surf->shm_gl_fail = true;
		return false;
	}

	trace(TRACE_SURF, "shm->gl(%d, %d)", glid, fd);
	arcan_shmif_signalhandle(acon,
		SHMIF_SIGVID | SHMIF_SIGBLK_NONE,
		fd, stride_out, out_fmt
	);

	fenv->delete_textures(1, &glid);
	return true;
}

/*
 * IGNORE, shmif doesn't split up into regions like this, though
 * we can forward it as messages and let the script-side decide.
 */
static void surf_opaque(struct wl_client* cl,
	struct wl_resource* res, struct wl_resource* reg)
{
	trace(TRACE_REGION, "opaque_region");
}

static void surf_inputreg(struct wl_client* cl,
	struct wl_resource* res, struct wl_resource* reg)
{
	trace(TRACE_REGION, "input_region");
/*
 * INCOMPLETE:
 * Should either send this onward for the wm scripts to mask/forward
 * events that fall outside the region, or annotate the surface resource
 * and route the input in the bridge. This becomes important with complex
 * hierarchies (from popups and subsurfaces).
 */
}

static bool fmt_has_alpha(int fmt, struct comp_surf* surf)
{
/* should possible check for the special case if the entire region is marked
 * as opaque as well or if there are translucent portions */
	return
		fmt == WL_SHM_FORMAT_XRGB8888 ||
		fmt == WL_DRM_FORMAT_XRGB4444 ||
		fmt == WL_DRM_FORMAT_XBGR4444 ||
		fmt == WL_DRM_FORMAT_RGBX4444 ||
		fmt == WL_DRM_FORMAT_BGRX4444 ||
		fmt == WL_DRM_FORMAT_XRGB1555 ||
		fmt == WL_DRM_FORMAT_XBGR1555 ||
		fmt == WL_DRM_FORMAT_RGBX5551 ||
		fmt == WL_DRM_FORMAT_BGRX5551 ||
		fmt == WL_DRM_FORMAT_XRGB8888 ||
		fmt == WL_DRM_FORMAT_XBGR8888 ||
		fmt == WL_DRM_FORMAT_RGBX8888 ||
		fmt == WL_DRM_FORMAT_BGRX8888 ||
		fmt == WL_DRM_FORMAT_XRGB2101010 ||
		fmt == WL_DRM_FORMAT_XBGR2101010 ||
		fmt == WL_DRM_FORMAT_RGBX1010102 ||
		fmt == WL_DRM_FORMAT_BGRX1010102;
}

static void synch_acon_alpha(struct arcan_shmif_cont* acon, bool has_alpha)
{
	if (has_alpha){
		if (acon->hints & SHMIF_RHINT_IGNORE_ALPHA){
			/* NOP */
		}
		else {
			acon->hints |= SHMIF_RHINT_IGNORE_ALPHA;
		}
	}
	else {
		if (acon->hints & SHMIF_RHINT_IGNORE_ALPHA){
			acon->hints &= ~SHMIF_RHINT_IGNORE_ALPHA;
		}
		else {
			/* NOP */
		}
	}
}

static bool push_drm(struct wl_client* cl,
	struct arcan_shmif_cont* acon, struct wl_resource* buf, struct comp_surf* surf)
{
	struct wl_drm_buffer* drm_buf = wayland_drm_buffer_get(wl.drm, buf);
	if (!drm_buf)
		return false;

	trace(TRACE_SURF, "surf_commit(egl:%s)", surf->tracetag);
	synch_acon_alpha(acon,
		fmt_has_alpha(wayland_drm_buffer_get_format(drm_buf), surf));
	wayland_drm_commit(surf, drm_buf, acon);
	return true;
}

static bool push_dma(struct wl_client* cl,
	struct arcan_shmif_cont* acon, struct wl_resource* buf, struct comp_surf* surf)
{
	struct dma_buf* dmabuf = dmabuf_buffer_get(buf);
	if (!dmabuf)
		return false;

	if (dmabuf->w != acon->w || dmabuf->h != acon->h){
		arcan_shmif_resize(acon, dmabuf->w, dmabuf->h);
	}

/* same dance as in wayland_drm, if the receiving side doesn't want dma bufs,
 * attach them to the context (and extend to accelerated) then force a CPU
 * readback - could be leveraged to perform other transforms at the same time,
 * one candidate being subsurface composition and colorspace conversion */
	if (!arcan_shmif_handle_permitted(acon) ||
			!arcan_shmif_handle_permitted(&wl.control)){
		if (!arcan_shmifext_isext(acon)){
			struct arcan_shmifext_setup defs = arcan_shmifext_defaults(acon);
			defs.no_context = true;
			arcan_shmifext_setup(acon, defs);
		}

		int n_planes = 0;
		struct shmifext_buffer_plane planes[4];
		for (size_t i = 0; i < 4; i++){
			planes[i] = dmabuf->planes[i];
			if (planes[i].fd <= 0)
				break;
			planes[i].fd = arcan_shmif_dupfd(planes[i].fd, -1, false);
			n_planes++;
		}

		if (arcan_shmifext_import_buffer(acon,
			SHMIFEXT_BUFFER_GBM, planes, n_planes, sizeof(planes[0]))){

		}
		else {
			for (size_t i = 0; i < n_planes; i++)
				if (planes[i].fd > 0)
					close(planes[i].fd);
		}

	return true;
	}

/* right now this only supports a single transfered buffer, the real support
 * is close by in another branch, but for the sake of bringup just block those
 * now */
	for (size_t i = 0; i < COUNT_OF(dmabuf->planes); i++){
		if (i == 0){
			arcan_shmif_signalhandle(acon, SHMIF_SIGVID | SHMIF_SIGBLK_NONE,
				dmabuf->planes[i].fd, dmabuf->planes[i].gbm.stride, dmabuf->fmt);
		}
	}

	synch_acon_alpha(acon, fmt_has_alpha(dmabuf->fmt, surf));

	trace(TRACE_SURF, "surf_commit(dmabuf:%s)", surf->tracetag);
	return true;
}

/*
 * since if we have GL already going if the .egl toggle is set, we can pull
 * in agp and use those functions raw
 */
#include "../../platform/video_platform.h"
static bool push_shm(struct wl_client* cl,
	struct arcan_shmif_cont* acon, struct wl_resource* buf, struct comp_surf* surf)
{
	struct wl_shm_buffer* shm_buf = wl_shm_buffer_get(buf);
	if (!shm_buf)
		return false;

	trace(TRACE_SURF, "surf_commit(shm:%s)", surf->tracetag);

	uint32_t w = wl_shm_buffer_get_width(shm_buf);
	uint32_t h = wl_shm_buffer_get_height(shm_buf);
	int fmt = wl_shm_buffer_get_format(shm_buf);
	void* data = wl_shm_buffer_get_data(shm_buf);
	int stride = wl_shm_buffer_get_stride(shm_buf);

	if (acon->w != w || acon->h != h){
		trace(TRACE_SURF,
			"surf_commit(shm, resize to: %zu, %zu)", (size_t)w, (size_t)h);
		arcan_shmif_resize(acon, w, h);
	}

/* resize failed, this will only happen when growing, thus we can crop */
	if (acon->w != w || acon->h != h){
		w = acon->w;
		h = acon->h;
	}

/* alpha state changed? only changing this flag does not require a resynch
 * as the hint is checked on each frame */
	synch_acon_alpha(acon, fmt_has_alpha(fmt, surf));
	wl_shm_buffer_begin_access(shm_buf);
	if (shm_to_gl(acon, surf, w, h, fmt, data, stride))
		goto out;

/* two other options to avoid repacking, one is to actually use this signal-
 * handle facility to send a descriptor, and mark the type as the WL shared
 * buffer with the metadata in vidp[] in order for offset and other bits to
 * make sense.
 * This is currently not supported in arcan/shmif.
 *
 * The other is to actually allow the shmif server to ptrace into us (wut)
 * and use a rare linuxism known as process_vm_writev and process_vm_readv
 * and send the pointers that way. One might call that one exotic.
 */
	if (stride != acon->stride){
		trace(TRACE_SURF,"surf_commit(stride-mismatch)");
		for (size_t row = 0; row < h; row++){
			memcpy(&acon->vidp[row * acon->pitch],
				&((uint8_t*)data)[row * stride],
				w * sizeof(shmif_pixel)
			);
		}
	}
	else
		memcpy(acon->vidp, data, w * h * sizeof(shmif_pixel));

	arcan_shmif_signal(acon, SHMIF_SIGVID | SHMIF_SIGBLK_NONE);

out:
	wl_shm_buffer_end_access(shm_buf);
	return true;
}

/*
 * Practically there is another thing to consider here and that is the trash
 * fire of subsurfaces. Mapping each to a shmif segment is costly, and
 * snowballs into a bunch of extra work in the WM, making otherwise trivial
 * features nightmarish. The other possible option here would be to do the
 * composition ourself into one shmif segment, masking that the thing exists at
 * all.
 */
static void surf_commit(struct wl_client* cl, struct wl_resource* res)
{
	struct comp_surf* surf = wl_resource_get_user_data(res);
	trace(TRACE_SURF, "%s (@%"PRIxPTR")->commit", surf->tracetag, (uintptr_t)surf->cbuf);
	struct arcan_shmif_cont* acon = &surf->acon;

	if (!surf){
		trace(TRACE_SURF, "no surface in resource (severe)");
		return;
	}

/* xwayland surface that is unpaired? */
	if (surf->cookie != 0xfeedface && res != surf->client->cursor){
		if (!xwl_pair_surface(cl, surf, res)){
			trace(TRACE_SURF, "defer commit until paired");
			return;
		}
	}

	if (!surf->cbuf){
		trace(TRACE_SURF, "no buffer");
		if (surf->internal){
			surf->internal(surf, CMD_RECONFIGURE);
			surf->internal(surf, CMD_FLUSH_CALLBACKS);
		}

		return;
	}

	if (!surf->client){
		trace(TRACE_SURF, "no bridge");
		return;
	}

	struct wl_resource* buf = (struct wl_resource*)(
		(uintptr_t) surf->buf ^ ((uintptr_t) 0xfeedface));
	if ((uintptr_t) buf != surf->cbuf){
		trace(TRACE_SURF, "corrupted or unknown buf "
			"(%"PRIxPTR" vs %"PRIxPTR") (severe)", (uintptr_t) buf, surf->cbuf);
		return;
	}

/*
 * special case, if the surface we should synch is the currently set
 * pointer resource, then draw that to the special segment.
 */
	if (surf->cookie != 0xfeedface){
		if (res == surf->client->cursor){
			acon = &surf->client->acursor;
/* synch hot-spot changes at this stage */
			if (surf->client->dirty_hot){
				struct arcan_event ev = {
					.ext.kind = ARCAN_EVENT(MESSAGE)
				};
				snprintf((char*)ev.ext.message.data,
					COUNT_OF(ev.ext.message.data), "hot:%d:%d",
					(int)(surf->client->hot_x * surf->scale),
					(int)(surf->client->hot_y * surf->scale)
				);
				arcan_shmif_enqueue(&surf->client->acursor, &ev);
				surf->client->dirty_hot = false;
			}
			trace(TRACE_SURF, "cursor updated");
		}
/* In general, a surface without a role means that the client is in the wrong
 * OR that there is a rootless Xwayland surface going - for the latter, we'd
 * like to figure out if this is correct or not - so wrap around a query
 * function. Since there can be stalls etc. in the corresponding wm, we need to
 * tag this surface as pending on failure */
}

	if (!acon || !acon->addr){
		trace(TRACE_SURF, "couldn't map to arcan connection");
		return;
	}

/*
 * Safeguard due to the SIGBLK_NONE, used for signalling, below.
 */
	while(arcan_shmif_signalstatus(acon) > 0){}

/*
 * So it seems that the buffer- protocol actually don't give us
 * a type of the buffer, so the canonical way is to just try them in
 * order shm -> drm -> dma-buf.
 */

	if (
		!push_shm(cl, acon, buf, surf) &&
		!push_drm(cl, acon, buf, surf) &&
		!push_dma(cl, acon, buf, surf)){
		trace(TRACE_SURF, "surf_commit(unknown:%s)", surf->tracetag);
	}

/* might be that this should be moved to the buffer types as well,
 * since we might need double-triple buffering, uncertain how mesa
 * actually handles this */
	wl_buffer_send_release(buf);

	trace(TRACE_SURF,
		"surf_commit(%zu,%zu-%zu,%zu)accel_fail=%d",
			(size_t)acon->dirty.x1, (size_t)acon->dirty.y1,
			(size_t)acon->dirty.x2, (size_t)acon->dirty.y2,
			(int)surf->shm_gl_fail);

/* reset the dirty rectangle */
	acon->dirty.x1 = acon->w;
	acon->dirty.x2 = 0;
	acon->dirty.y1 = acon->h;
	acon->dirty.y2 = 0;
}

static void surf_transform(struct wl_client* cl,
	struct wl_resource* res, int32_t transform)
{
	trace(TRACE_SURF, "surf_transform(%d)", (int) transform);
	struct comp_surf* surf = wl_resource_get_user_data(res);
	if (!surf || !surf->acon.addr)
		return;

	struct arcan_event ev = {
		.ext.kind = ARCAN_EVENT(MESSAGE),
	};
	snprintf((char*)ev.ext.message.data,
		COUNT_OF(ev.ext.message.data), "transform:%"PRId32, transform);

	arcan_shmif_enqueue(&surf->acon, &ev);
}

static void surf_scale(struct wl_client* cl,
	struct wl_resource* res, int32_t scale)
{
	trace(TRACE_SURF, "surf_scale(%d)", (int) scale);
	struct comp_surf* surf = wl_resource_get_user_data(res);
	if (!surf || !surf->acon.addr)
		return;

	surf->scale = scale > 0 ? scale : 1;
	struct arcan_event ev = {
		.ext.kind = ARCAN_EVENT(MESSAGE)
	};
	snprintf((char*)ev.ext.message.data,
		COUNT_OF(ev.ext.message.data), "scale:%"PRId32, scale);
	arcan_shmif_enqueue(&surf->acon, &ev);
}
