/*
 * Web Audio output backend for the wasm (emscripten) build.
 *
 * There is no native audio device here, so this backend pulls PCM from the
 * stream's data_callback on a dedicated fill thread, writes it into a lock-free
 * single-producer/single-consumer ring buffer that lives in the shared wasm heap,
 * and a host-side AudioWorklet (build/audio-out.js) drains the ring into a Web
 * Audio AudioContext on the page. Output only (no capture). The JS side
 * (emaudio_*) runs proxied on the main browser thread, since AudioContext is only
 * available there.
 */
#include "cubeb-internal.h"
#include "cubeb/cubeb.h"
#include <emscripten.h>
#include <emscripten/threading.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* JS bridge (build/audio-out.js, linked via --js-library). */
extern int emaudio_get_rate(void);
extern int emaudio_start(void * ring, int cap_frames, int channels, int rate);
extern void emaudio_stop(void);

/* Ring capacity in frames. MUST be a power of two so the frame index wraps
 * cleanly across the uint32 counter wrap (2^32 % cap == 0). 16384 frames is
 * ~341ms at 48kHz. */
#define EM_RING_CAP 16384
#define EM_FILL_CHUNK 4096

/* Header MUST match audio-out.js's view: int32 write, int32 read, then float
 * data[cap*channels]. write/read are absolute frame counters using uint32
 * modular arithmetic (avail = (uint32_t)(write - read)). */
struct em_ring {
  _Atomic int32_t write;
  _Atomic int32_t read;
  float data[1];
};

struct cubeb {
  struct cubeb_ops const * ops;
  int rate;
};

struct cubeb_stream {
  cubeb * context;
  cubeb_data_callback data_cb;
  cubeb_state_callback state_cb;
  void * user_ptr;
  uint32_t rate;
  uint32_t channels;
  cubeb_sample_format format;
  struct em_ring * ring;
  void * tmp; /* data_callback scratch, in the stream's sample format */
  _Atomic int running;
  _Atomic int draining;
  _Atomic float volume;
  pthread_t thread;
};

extern struct cubeb_ops const wasmaudio_ops;

static int
is_float_format(cubeb_sample_format f)
{
  return !(f == CUBEB_SAMPLE_S16LE || f == CUBEB_SAMPLE_S16BE);
}

int
wasmaudio_init(cubeb ** context, char const * context_name)
{
  (void)context_name;
  cubeb * ctx = (cubeb *)calloc(1, sizeof(cubeb));
  if (!ctx) {
    return CUBEB_ERROR;
  }
  ctx->ops = &wasmaudio_ops;
  ctx->rate = emaudio_get_rate(); /* AudioContext.sampleRate (proxied to main) */
  if (ctx->rate <= 0) {
    free(ctx);
    return CUBEB_ERROR; /* no Web Audio available */
  }
  *context = ctx;
  return CUBEB_OK;
}

static char const *
wasmaudio_get_backend_id(cubeb * ctx)
{
  (void)ctx;
  return "wasmaudio";
}

static int
wasmaudio_get_max_channel_count(cubeb * ctx, uint32_t * max_channels)
{
  (void)ctx;
  *max_channels = 2;
  return CUBEB_OK;
}

static int
wasmaudio_get_min_latency(cubeb * ctx, cubeb_stream_params params,
                          uint32_t * latency_frames)
{
  (void)ctx;
  (void)params;
  *latency_frames = 2048; /* frames; ring holds far more */
  return CUBEB_OK;
}

static int
wasmaudio_get_preferred_sample_rate(cubeb * ctx, uint32_t * rate)
{
  *rate = ctx->rate > 0 ? (uint32_t)ctx->rate : 48000;
  return CUBEB_OK;
}

static void
wasmaudio_destroy(cubeb * ctx)
{
  free(ctx);
}

/* Fill thread: pull from data_callback and write into the ring (converting to
 * float32 and applying volume). Sleeps when the ring is full. */
static void *
em_fill_thread(void * arg)
{
  cubeb_stream * s = (cubeb_stream *)arg;
  const int ch = (int)s->channels;
  const int is_float = is_float_format(s->format);

  while (atomic_load(&s->running)) {
    int32_t w = atomic_load_explicit(&s->ring->write, memory_order_relaxed);
    int32_t r = atomic_load_explicit(&s->ring->read, memory_order_acquire);
    uint32_t used = (uint32_t)(w - r);
    if (used > EM_RING_CAP) {
      used = EM_RING_CAP;
    }
    uint32_t space = EM_RING_CAP - used;

    if (!atomic_load(&s->draining) && space >= 1024) {
      long want = (long)space;
      if (want > EM_FILL_CHUNK) {
        want = EM_FILL_CHUNK;
      }
      long got = s->data_cb(s, s->user_ptr, NULL, s->tmp, want);
      if (got < 0) {
        s->state_cb(s, s->user_ptr, CUBEB_STATE_ERROR);
        break;
      }
      float vol = atomic_load(&s->volume);
      for (long i = 0; i < got; i++) {
        int pos = (int)((uint32_t)(w + i) & (EM_RING_CAP - 1)) * ch;
        for (int c = 0; c < ch; c++) {
          float v;
          if (is_float) {
            v = ((const float *)s->tmp)[i * ch + c];
          } else {
            v = ((const int16_t *)s->tmp)[i * ch + c] / 32768.0f;
          }
          s->ring->data[pos + c] = v * vol;
        }
      }
      atomic_store_explicit(&s->ring->write, w + (int32_t)got,
                            memory_order_release);
      if (got < want) {
        atomic_store(&s->draining, 1); /* short read => no more data */
      }
    } else if (atomic_load(&s->draining)) {
      int32_t w2 = atomic_load(&s->ring->write);
      int32_t r2 = atomic_load_explicit(&s->ring->read, memory_order_acquire);
      if ((uint32_t)(w2 - r2) == 0) {
        s->state_cb(s, s->user_ptr, CUBEB_STATE_DRAINED);
        break;
      }
      emscripten_thread_sleep(10);
    } else {
      emscripten_thread_sleep(10);
    }
  }
  atomic_store(&s->running, 0);
  return NULL;
}

static int
wasmaudio_stream_init(cubeb * context, cubeb_stream ** stream,
                      char const * stream_name, cubeb_devid input_device,
                      cubeb_stream_params * input_stream_params,
                      cubeb_devid output_device,
                      cubeb_stream_params * output_stream_params,
                      unsigned int latency, cubeb_data_callback data_callback,
                      cubeb_state_callback state_callback, void * user_ptr)
{
  (void)stream_name;
  (void)input_device;
  (void)output_device;
  (void)latency;

  if (input_stream_params) {
    return CUBEB_ERROR_NOT_SUPPORTED; /* output only */
  }
  if (!output_stream_params) {
    return CUBEB_ERROR_INVALID_PARAMETER;
  }

  uint32_t ch = output_stream_params->channels;
  if (ch < 1) {
    ch = 1;
  }
  if (ch > 2) {
    ch = 2; /* the worklet outputs at most stereo */
  }

  cubeb_stream * s = (cubeb_stream *)calloc(1, sizeof(cubeb_stream));
  if (!s) {
    return CUBEB_ERROR;
  }
  s->context = context;
  s->data_cb = data_callback;
  s->state_cb = state_callback;
  s->user_ptr = user_ptr;
  s->rate = output_stream_params->rate;
  s->channels = ch;
  s->format = output_stream_params->format;
  atomic_store(&s->volume, 1.0f);

  size_t ring_bytes =
      sizeof(struct em_ring) + (size_t)EM_RING_CAP * ch * sizeof(float);
  s->ring = (struct em_ring *)calloc(1, ring_bytes);
  size_t sample_size = is_float_format(s->format) ? sizeof(float) : sizeof(int16_t);
  s->tmp = calloc((size_t)EM_FILL_CHUNK * ch, sample_size);
  if (!s->ring || !s->tmp) {
    free(s->ring);
    free(s->tmp);
    free(s);
    return CUBEB_ERROR;
  }

  *stream = s;
  return CUBEB_OK;
}

static int
wasmaudio_stream_stop(cubeb_stream * s)
{
  if (atomic_exchange(&s->running, 0)) {
    pthread_join(s->thread, NULL);
    emaudio_stop();
    s->state_cb(s, s->user_ptr, CUBEB_STATE_STOPPED);
  }
  return CUBEB_OK;
}

static void
wasmaudio_stream_destroy(cubeb_stream * s)
{
  wasmaudio_stream_stop(s);
  free(s->ring);
  free(s->tmp);
  free(s);
}

static int
wasmaudio_stream_start(cubeb_stream * s)
{
  if (atomic_load(&s->running)) {
    return CUBEB_OK;
  }
  if (emaudio_start(s->ring, EM_RING_CAP, (int)s->channels, (int)s->rate) != 0) {
    return CUBEB_ERROR;
  }
  atomic_store(&s->draining, 0);
  atomic_store(&s->running, 1);
  if (pthread_create(&s->thread, NULL, em_fill_thread, s) != 0) {
    atomic_store(&s->running, 0);
    emaudio_stop();
    return CUBEB_ERROR;
  }
  s->state_cb(s, s->user_ptr, CUBEB_STATE_STARTED);
  return CUBEB_OK;
}

static int
wasmaudio_stream_get_position(cubeb_stream * s, uint64_t * position)
{
  /* Frames the worklet has consumed. uint32 counter (wraps ~12h at 48kHz). */
  *position = (uint32_t)atomic_load(&s->ring->read);
  return CUBEB_OK;
}

static int
wasmaudio_stream_get_latency(cubeb_stream * s, uint32_t * latency)
{
  int32_t w = atomic_load(&s->ring->write);
  int32_t r = atomic_load(&s->ring->read);
  *latency = (uint32_t)(w - r);
  return CUBEB_OK;
}

static int
wasmaudio_stream_set_volume(cubeb_stream * s, float volume)
{
  atomic_store(&s->volume, volume);
  return CUBEB_OK;
}

struct cubeb_ops const wasmaudio_ops = {
    .init = wasmaudio_init,
    .get_backend_id = wasmaudio_get_backend_id,
    .get_max_channel_count = wasmaudio_get_max_channel_count,
    .get_min_latency = wasmaudio_get_min_latency,
    .get_preferred_sample_rate = wasmaudio_get_preferred_sample_rate,
    .get_supported_input_processing_params = NULL,
    .enumerate_devices = NULL,
    .device_collection_destroy = NULL,
    .destroy = wasmaudio_destroy,
    .stream_init = wasmaudio_stream_init,
    .stream_destroy = wasmaudio_stream_destroy,
    .stream_start = wasmaudio_stream_start,
    .stream_stop = wasmaudio_stream_stop,
    .stream_get_position = wasmaudio_stream_get_position,
    .stream_get_latency = wasmaudio_stream_get_latency,
    .stream_get_input_latency = NULL,
    .stream_set_volume = wasmaudio_stream_set_volume,
    .stream_set_name = NULL,
    .stream_get_current_device = NULL,
    .stream_set_input_mute = NULL,
    .stream_set_input_processing_params = NULL,
    .stream_device_destroy = NULL,
    .stream_register_device_changed_callback = NULL,
    .register_device_collection_changed = NULL};
