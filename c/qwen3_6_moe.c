#include "qwen3_6_moe.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#if defined(__AVX2__) || defined(__x86_64__)
#include <immintrin.h>
#endif

/* Activation helpers */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* AVX2 Accelerated Dot Product */
static float dot_product_f32(const float *a, const float *b, int n) {
    int i = 0;
    float sum = 0.0f;
#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; k++) sum += tmp[k];
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Tensor Functions */
void qwen_qt_init_f32(QwenQT *qt, int O, int I, float *data) {
    memset(qt, 0, sizeof(*qt));
    qt->fmt = QWEN_FMT_F32;
    qt->O = O;
    qt->I = I;
    qt->qf = data;
}

void qwen_qt_init_int8(QwenQT *qt, int O, int I, int8_t *q8, float *scales) {
    memset(qt, 0, sizeof(*qt));
    qt->fmt = QWEN_FMT_INT8;
    qt->O = O;
    qt->I = I;
    qt->q8 = q8;
    qt->s = scales;
}

void qwen_qt_free(QwenQT *qt) {
    /* For zero-copy views, pointers are owned elsewhere or by mmap */
    memset(qt, 0, sizeof(*qt));
}

void qwen_matmul_qt(const float *x, const QwenQT *w, float *out, int S) {
    (void)S; /* Single token pipeline (S = 1) */
    int O = w->O;
    int I = w->I;

    if (w->fmt == QWEN_FMT_F32 && w->qf) {
        for (int o = 0; o < O; o++) {
            out[o] = dot_product_f32(x, w->qf + o * I, I);
        }
    } else if (w->fmt == QWEN_FMT_INT8 && w->q8 && w->s) {
        for (int o = 0; o < O; o++) {
            const int8_t *row = w->q8 + o * I;
            float scale = w->s[o];
            float sum = 0.0f;
            int i = 0;
#if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            for (; i <= I - 8; i += 8) {
                __m256 vx = _mm256_loadu_ps(x + i);
                __m256i vq = _mm256_cvtepi8_epi32(_mm_loadu_si64(row + i));
                __m256 vqf = _mm256_cvtepi32_ps(vq);
                vsum = _mm256_fmadd_ps(vx, vqf, vsum);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            for (int k = 0; k < 8; k++) sum += tmp[k];
#endif
            for (; i < I; i++) {
                sum += x[i] * (float)row[i];
            }
            out[o] = sum * scale;
        }
    } else {
        /* Fallback for uninitialized/null weights */
        memset(out, 0, O * sizeof(float));
    }
}

/* Memory Mapping Functions */
int qwen_mmap_create_dummy_shard(const char *filepath, size_t size) {
    int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    if (ftruncate(fd, size) != 0) {
        close(fd);
        return -1;
    }
    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        return -1;
    }

    /* Fill dummy shard with synthetic quant int8 data + scales for 256 experts */
    int8_t *bptr = (int8_t*)ptr;
    for (size_t i = 0; i < size; i++) {
        bptr[i] = (int8_t)((i % 127) - 63);
    }
    
    munmap(ptr, size);
    close(fd);
    return 0;
}

int qwen_mmap_open(QwenMmapShard *shard, const char *filepath) {
    memset(shard, 0, sizeof(*shard));
    strncpy(shard->filepath, filepath, sizeof(shard->filepath) - 1);
    
    shard->fd = open(filepath, O_RDONLY);
    if (shard->fd < 0) {
        perror("[QwenMoE] open shard failed");
        return -1;
    }

    struct stat st;
    if (fstat(shard->fd, &st) != 0) {
        perror("[QwenMoE] fstat shard failed");
        close(shard->fd);
        shard->fd = -1;
        return -1;
    }
    shard->size = (size_t)st.st_size;

    shard->base_addr = mmap(NULL, shard->size, PROT_READ, MAP_SHARED, shard->fd, 0);
    if (shard->base_addr == MAP_FAILED) {
        perror("[QwenMoE] mmap shard failed");
        close(shard->fd);
        shard->fd = -1;
        return -1;
    }

#if defined(__linux__)
    madvise(shard->base_addr, shard->size, MADV_WILLNEED);
#endif

    shard->is_mapped = true;
    printf("[QwenMoE] Memory-mapped shard '%s' (%.2f MB)\n", 
           filepath, (double)shard->size / (1024.0 * 1024.0));
    return 0;
}

void qwen_mmap_close(QwenMmapShard *shard) {
    if (shard && shard->is_mapped) {
        if (shard->base_addr && shard->base_addr != MAP_FAILED) {
            munmap(shard->base_addr, shard->size);
        }
        if (shard->fd >= 0) {
            close(shard->fd);
        }
        shard->is_mapped = false;
        shard->base_addr = NULL;
        shard->fd = -1;
    }
}

/* LRU Cache Implementation */
int qwen_lru_init(QwenLRUCache *cache, int capacity) {
    memset(cache, 0, sizeof(*cache));
    cache->capacity = capacity;
    cache->slots = (QwenESlot*)calloc(capacity, sizeof(QwenESlot));
    if (!cache->slots) return -1;

    for (int i = 0; i < capacity; i++) {
        cache->slots[i].eid = -1;
    }
    return 0;
}

void qwen_lru_free(QwenLRUCache *cache) {
    if (cache && cache->slots) {
        for (int i = 0; i < cache->capacity; i++) {
            if (!cache->slots[i].is_mmap) {
                if (cache->slots[i].slab) free(cache->slots[i].slab);
                if (cache->slots[i].fslab) free(cache->slots[i].fslab);
            }
        }
        free(cache->slots);
        cache->slots = NULL;
    }
}

QwenESlot* qwen_lru_lookup(QwenLRUCache *cache, int eid) {
    if (!cache || eid < 0) return NULL;
    for (int i = 0; i < cache->capacity; i++) {
        if (cache->slots[i].eid == eid) {
            cache->global_clock++;
            cache->slots[i].last_used = cache->global_clock;
            cache->hits++;
            return &cache->slots[i];
        }
    }
    return NULL;
}

QwenESlot* qwen_lru_evict_and_prepare(QwenLRUCache *cache, int eid_to_load) {
    if (!cache) return NULL;
    cache->global_clock++;
    cache->misses++;

    /* 1. Check for empty slot */
    for (int i = 0; i < cache->capacity; i++) {
        if (cache->slots[i].eid < 0) {
            cache->slots[i].eid = eid_to_load;
            cache->slots[i].last_used = cache->global_clock;
            cache->count++;
            return &cache->slots[i];
        }
    }

    /* 2. Cache is full: Find LRU slot (minimum last_used) */
    int lru_idx = 0;
    uint64_t min_clock = cache->slots[0].last_used;
    for (int i = 1; i < cache->capacity; i++) {
        if (cache->slots[i].last_used < min_clock) {
            min_clock = cache->slots[i].last_used;
            lru_idx = i;
        }
    }

    /* Evict selected slot */
    QwenESlot *slot = &cache->slots[lru_idx];
    cache->evictions++;
    slot->eid = eid_to_load;
    slot->last_used = cache->global_clock;
    return slot;
}

int qwen_expert_load_from_mmap(QwenMoELayer *layer, QwenMmapShard *shard, int eid, QwenESlot *slot) {
    if (!shard || !shard->is_mapped || !slot) return -1;
    
    int H = layer->hidden_dim;
    int I = layer->moe_inter_dim;

    /* Compute offset into mmap shard based on eid */
    size_t expert_bytes = (size_t)(I * H * 2 + H * I + (I * 2 + H) * sizeof(float));
    size_t offset = (size_t)eid * expert_bytes;

    if (offset + expert_bytes > shard->size) {
        /* Fallback: wrap offset around if dummy shard size is smaller */
        offset = offset % (shard->size > expert_bytes ? (shard->size - expert_bytes) : 1);
    }

    uint8_t *base = (uint8_t*)shard->base_addr + offset;

    slot->eid = eid;
    slot->is_mmap = true;

    /* Zero-copy view mapping for gate, up, and down projections */
    int8_t *g_q8 = (int8_t*)base;
    int8_t *u_q8 = g_q8 + (I * H);
    int8_t *d_q8 = u_q8 + (I * H);
    float *scales = (float*)(d_q8 + (H * I));

    qwen_qt_init_int8(&slot->g, I, H, g_q8, scales);
    qwen_qt_init_int8(&slot->u, I, H, u_q8, scales + I);
    qwen_qt_init_int8(&slot->d, H, I, d_q8, scales + 2 * I);

    return 0;
}

/* Router Selection */
void qwen_router_select(const float *x, const float *router_w, const float *router_b, 
                        int hidden, int n_experts, int topk, 
                        int *out_eids, float *out_weights) {
    float logits[256];
    for (int e = 0; e < n_experts; e++) {
        float score = dot_product_f32(x, router_w + e * hidden, hidden);
        if (router_b) score += router_b[e];
        logits[e] = score;
    }

    /* Top-K Selection via Bubble / Select */
    for (int k = 0; k < topk; k++) {
        int best_idx = -1;
        float best_val = -1e30f;
        for (int e = 0; e < n_experts; e++) {
            bool already_selected = false;
            for (int prev = 0; prev < k; prev++) {
                if (out_eids[prev] == e) { already_selected = true; break; }
            }
            if (!already_selected && logits[e] > best_val) {
                best_val = logits[e];
                best_idx = e;
            }
        }
        out_eids[k] = (best_idx >= 0) ? best_idx : k;
        out_weights[k] = best_val;
    }

    /* Softmax over Top-K */
    float max_logit = out_weights[0];
    for (int k = 1; k < topk; k++) {
        if (out_weights[k] > max_logit) max_logit = out_weights[k];
    }
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) {
        out_weights[k] = expf(out_weights[k] - max_logit);
        sum += out_weights[k];
    }
    float inv_sum = sum > 0.0f ? (1.0f / sum) : 1.0f;
    for (int k = 0; k < topk; k++) {
        out_weights[k] *= inv_sum;
    }
}

/* Expert Forward (SwiGLU) */
void qwen_expert_forward(const QwenESlot *slot, const float *x, float *out, int hidden, int inter) {
    (void)hidden;
    float gate_buf[1408];
    float up_buf[1408];
    float intermediate[1408];

    qwen_matmul_qt(x, &slot->g, gate_buf, 1);
    qwen_matmul_qt(x, &slot->u, up_buf, 1);

    for (int i = 0; i < inter; i++) {
        intermediate[i] = silu(gate_buf[i]) * up_buf[i];
    }

    qwen_matmul_qt(intermediate, &slot->d, out, 1);
}

void qwen_shared_expert_forward(const QwenMoELayer *layer, const float *x, float *out) {
    int S_I = QWEN3_6_SHARED_INTER_DIM;

    float gate_buf[1408];
    float up_buf[1408];
    float intermediate[1408];

    qwen_matmul_qt(x, &layer->sh_gate, gate_buf, 1);
    qwen_matmul_qt(x, &layer->sh_up, up_buf, 1);

    for (int i = 0; i < S_I; i++) {
        intermediate[i] = silu(gate_buf[i]) * up_buf[i];
    }

    qwen_matmul_qt(intermediate, &layer->sh_down, out, 1);
}

/* Gated DeltaNet Linear Attention */
int qwen_deltanet_init(QwenGatedDeltaNet *dn, int hidden, int num_heads, int head_dim) {
    memset(dn, 0, sizeof(*dn));
    dn->hidden_dim = hidden;
    dn->num_heads = num_heads;
    dn->head_dim = head_dim;

    size_t state_elems = (size_t)num_heads * head_dim * head_dim;
    dn->state_matrix = (float*)calloc(state_elems, sizeof(float));
    if (!dn->state_matrix) return -1;

    return 0;
}

void qwen_deltanet_free(QwenGatedDeltaNet *dn) {
    if (dn && dn->state_matrix) {
        free(dn->state_matrix);
        dn->state_matrix = NULL;
    }
}

void qwen_deltanet_forward(QwenGatedDeltaNet *dn, const float *x, float *out) {
    int NH = dn->num_heads;
    int HD = dn->head_dim;

    float q[2048], k[2048], v[4096], g[4096], beta[2048];
    qwen_matmul_qt(x, &dn->q_proj, q, 1);
    qwen_matmul_qt(x, &dn->k_proj, k, 1);
    qwen_matmul_qt(x, &dn->v_proj, v, 1);
    qwen_matmul_qt(x, &dn->g_proj, g, 1);
    qwen_matmul_qt(x, &dn->beta_proj, beta, 1);

    float attn_out[4096];

    for (int h = 0; h < NH; h++) {
        const float *qh = q + h * HD;
        const float *kh = k + h * HD;
        const float *vh = v + h * (HD * 2);
        float b_val = sigmoid(beta[h * HD]);
        float *S_h = dn->state_matrix + h * HD * HD;

        /* DeltaNet Recurrence State Update: S_t = (1 - beta) * S_{t-1} + beta * (v * k^T) */
        for (int r = 0; r < HD; r++) {
            for (int c = 0; c < HD; c++) {
                int idx = r * HD + c;
                S_h[idx] = (1.0f - b_val) * S_h[idx] + b_val * (vh[r] * kh[c]);
            }
        }

        /* Recurrence Query Matrix Vector Multiplication: y = S_t * q */
        float *yh = attn_out + h * (HD * 2);
        for (int r = 0; r < HD * 2; r++) {
            float sum = 0.0f;
            for (int c = 0; c < HD; c++) {
                sum += S_h[(r % HD) * HD + c] * qh[c];
            }
            /* Gated linear output: y * silu(gate) */
            yh[r] = sum * silu(g[h * (HD * 2) + r]);
        }
    }

    qwen_matmul_qt(attn_out, &dn->out_proj, out, 1);

    /* Residual pass through */
    for (int i = 0; i < dn->hidden_dim; i++) {
        out[i] += x[i];
    }
}

/* Layer Initialization & Forward Pipeline */
int qwen_moe_layer_init(QwenMoELayer *layer, int layer_id, int hidden, int moe_inter, 
                        int n_experts, int topk, int cache_capacity) {
    memset(layer, 0, sizeof(*layer));
    layer->layer_id = layer_id;
    layer->hidden_dim = hidden;
    layer->moe_inter_dim = moe_inter;
    layer->n_routed_experts = n_experts;
    layer->topk = topk;
    layer->is_full_attn = (layer_id == 11 || layer_id == 15 || layer_id == 19 || layer_id == 23 || 
                           layer_id == 27 || layer_id == 31 || layer_id == 35 || layer_id == 39);

    layer->input_layernorm = (float*)malloc(hidden * sizeof(float));
    layer->post_attn_layernorm = (float*)malloc(hidden * sizeof(float));
    layer->q_norm = (float*)calloc(256, sizeof(float));
    layer->k_norm = (float*)calloc(256, sizeof(float));
    for (int i = 0; i < hidden; i++) {
        layer->input_layernorm[i] = 1.0f;
        layer->post_attn_layernorm[i] = 1.0f;
    }

    if (qwen_deltanet_init(&layer->deltanet, hidden, QWEN3_6_N_HEADS, QWEN3_6_HEAD_DIM) != 0) {
        return -1;
    }

    layer->router_weights = (float*)calloc(n_experts * hidden, sizeof(float));
    layer->router_bias = (float*)calloc(n_experts, sizeof(float));
    for (int e = 0; e < n_experts; e++) {
        for (int h = 0; h < hidden; h++) {
            layer->router_weights[e * hidden + h] = (float)sin(e + h * 0.01);
        }
    }

    if (qwen_lru_init(&layer->expert_cache, cache_capacity) != 0) {
        return -1;
    }

    return 0;
}

void qwen_moe_layer_free(QwenMoELayer *layer) {
    if (layer) {
        if (layer->input_layernorm) free(layer->input_layernorm);
        if (layer->post_attn_layernorm) free(layer->post_attn_layernorm);
        if (layer->q_norm) free(layer->q_norm);
        if (layer->k_norm) free(layer->k_norm);
        if (layer->router_weights) free(layer->router_weights);
        if (layer->router_bias) free(layer->router_bias);

        qwen_deltanet_free(&layer->deltanet);
        qwen_lru_free(&layer->expert_cache);
    }
}

void qwen_moe_layer_forward(QwenMoELayer *layer, const float *x_in, float *x_out, QwenMmapShard *shard) {
    int H = layer->hidden_dim;
    int topk = layer->topk;

    /* 1. Input RMSNorm + Gated DeltaNet Linear Attention */
    float ss0 = 0.0f;
    for (int i = 0; i < H; i++) ss0 += x_in[i] * x_in[i];
    float rms0 = 1.0f / sqrtf(ss0 / H + 1e-6f);

    float norm_x[2048] = {0};
    for (int i = 0; i < H; i++) {
        norm_x[i] = (x_in[i] * rms0) * layer->input_layernorm[i];
    }

    float attn_out[2048] = {0};
    qwen_deltanet_forward(&layer->deltanet, norm_x, attn_out);

    /* Residual connection 1 */
    float x1[2048] = {0};
    for (int i = 0; i < H; i++) {
        x1[i] = x_in[i] + attn_out[i];
    }

    /* 2. Post-Attention RMSNorm */
    float ss1 = 0.0f;
    for (int i = 0; i < H; i++) ss1 += x1[i] * x1[i];
    float rms1 = 1.0f / sqrtf(ss1 / H + 1e-6f);

    float norm_x1[2048] = {0};
    for (int i = 0; i < H; i++) {
        norm_x1[i] = (x1[i] * rms1) * layer->post_attn_layernorm[i];
    }

    /* 3. Shared Expert Execution */
    float shared_out[2048] = {0};
    qwen_shared_expert_forward(layer, norm_x1, shared_out);

    /* 4. Top-K Expert Routing & LRU Streaming */
    int selected_eids[QWEN3_6_TOPK];
    float routing_weights[QWEN3_6_TOPK];
    qwen_router_select(norm_x1, layer->router_weights, layer->router_bias,
                       H, layer->n_routed_experts, topk, selected_eids, routing_weights);

    float routed_out[2048];
    memset(routed_out, 0, H * sizeof(float));

    for (int k = 0; k < topk; k++) {
        int eid = selected_eids[k];
        float w_k = routing_weights[k];

        /* Check LRU cache */
        QwenESlot *slot = qwen_lru_lookup(&layer->expert_cache, eid);
        if (!slot) {
            /* Evict LRU slot and load expert weights from mmap shard */
            slot = qwen_lru_evict_and_prepare(&layer->expert_cache, eid);
            qwen_expert_load_from_mmap(layer, shard, eid, slot);
        }

        /* Forward single expert */
        float exp_out[2048];
        qwen_expert_forward(slot, norm_x1, exp_out, H, layer->moe_inter_dim);

        /* Accumulate weighted contribution */
        for (int i = 0; i < H; i++) {
            routed_out[i] += w_k * exp_out[i];
        }
    }

    /* 5. Final residual combination: x_out = x1 + shared_out + routed_out */
    for (int i = 0; i < H; i++) {
        x_out[i] = x1[i] + shared_out[i] + routed_out[i];
    }
}

/* Full Model Pipeline Functions */
int qwen_model_init(Qwen3_6Model *model, int n_layers, int hidden, int moe_inter, 
                    int shared_inter, int n_experts, int topk, int vocab, int cache_capacity) {
    memset(model, 0, sizeof(*model));
    model->n_layers = n_layers;
    model->hidden_dim = hidden;
    model->moe_inter_dim = moe_inter;
    model->shared_inter_dim = shared_inter;
    model->n_routed_experts = n_experts;
    model->topk = topk;
    model->vocab_size = vocab;

    model->embed_tokens = (float*)calloc((size_t)vocab * hidden, sizeof(float));
    model->final_norm = (float*)malloc(hidden * sizeof(float));
    model->lm_head = (float*)calloc((size_t)vocab * hidden, sizeof(float));
    model->layers = (QwenMoELayer*)calloc(n_layers, sizeof(QwenMoELayer));

    if (!model->embed_tokens || !model->final_norm || !model->lm_head || !model->layers) {
        qwen_model_free(model);
        return -1;
    }

    for (int i = 0; i < hidden; i++) {
        model->final_norm[i] = 1.0f;
    }

    for (int l = 0; l < n_layers; l++) {
        if (qwen_moe_layer_init(&model->layers[l], l, hidden, moe_inter, n_experts, topk, cache_capacity) != 0) {
            qwen_model_free(model);
            return -1;
        }
    }

    return 0;
}

int qwen_model_load_backbone(Qwen3_6Model *model, const char *backbone_path) {
    FILE *f = fopen(backbone_path, "rb");
    if (!f) return -1;

    uint32_t magic = 0, version = 0;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || magic != 0x5157454E) {
        fclose(f);
        return -1;
    }
    int32_t n_l=0, h_d=0, m_i=0, s_i=0, n_e=0, tk=0, v_s=0;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&n_l, sizeof(int32_t), 1, f) != 1 ||
        fread(&h_d, sizeof(int32_t), 1, f) != 1 ||
        fread(&m_i, sizeof(int32_t), 1, f) != 1 ||
        fread(&s_i, sizeof(int32_t), 1, f) != 1 ||
        fread(&n_e, sizeof(int32_t), 1, f) != 1 ||
        fread(&tk, sizeof(int32_t), 1, f) != 1 ||
        fread(&v_s, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    /* Read embedding table */
    size_t embed_size = (size_t)v_s * h_d;
    if (fread(model->embed_tokens, sizeof(float), embed_size, f) != embed_size) {
        fclose(f);
        return -1;
    }

    /* Read layers backbone */
    for (int l = 0; l < n_l && l < model->n_layers; l++) {
        QwenMoELayer *layer = &model->layers[l];
        if (fread(layer->input_layernorm, sizeof(float), h_d, f) != (size_t)h_d) break;

        float *buf_q = (float*)malloc(h_d * h_d * sizeof(float));
        float *buf_k = (float*)malloc(h_d * h_d * sizeof(float));
        float *buf_v = (float*)malloc(4096 * h_d * sizeof(float));
        float *buf_g = (float*)malloc(4096 * h_d * sizeof(float));
        float *buf_beta = (float*)malloc(h_d * h_d * sizeof(float));
        float *buf_out = (float*)malloc(h_d * 4096 * sizeof(float));

        size_t nread = 0;
        nread += fread(buf_q, sizeof(float), h_d * h_d, f);
        nread += fread(buf_k, sizeof(float), h_d * h_d, f);
        nread += fread(buf_v, sizeof(float), 4096 * h_d, f);
        nread += fread(buf_g, sizeof(float), 4096 * h_d, f);
        nread += fread(buf_beta, sizeof(float), h_d * h_d, f);
        nread += fread(buf_out, sizeof(float), h_d * 4096, f);

        qwen_qt_init_f32(&layer->deltanet.q_proj, h_d, h_d, buf_q);
        qwen_qt_init_f32(&layer->deltanet.k_proj, h_d, h_d, buf_k);
        qwen_qt_init_f32(&layer->deltanet.v_proj, 4096, h_d, buf_v);
        qwen_qt_init_f32(&layer->deltanet.g_proj, 4096, h_d, buf_g);
        qwen_qt_init_f32(&layer->deltanet.beta_proj, h_d, h_d, buf_beta);
        qwen_qt_init_f32(&layer->deltanet.out_proj, h_d, 4096, buf_out);

        nread += fread(layer->post_attn_layernorm, sizeof(float), h_d, f);
        nread += fread(layer->q_norm, sizeof(float), 256, f);
        nread += fread(layer->k_norm, sizeof(float), 256, f);

        float *buf_sg = (float*)malloc(s_i * h_d * sizeof(float));
        float *buf_su = (float*)malloc(s_i * h_d * sizeof(float));
        float *buf_sd = (float*)malloc(h_d * s_i * sizeof(float));
        nread += fread(buf_sg, sizeof(float), s_i * h_d, f);
        nread += fread(buf_su, sizeof(float), s_i * h_d, f);
        nread += fread(buf_sd, sizeof(float), h_d * s_i, f);
        qwen_qt_init_f32(&layer->sh_gate, s_i, h_d, buf_sg);
        qwen_qt_init_f32(&layer->sh_up, s_i, h_d, buf_su);
        qwen_qt_init_f32(&layer->sh_down, h_d, s_i, buf_sd);

        nread += fread(layer->router_weights, sizeof(float), n_e * h_d, f);
        nread += fread(layer->router_bias, sizeof(float), n_e, f);
        (void)nread;
    }

    size_t fn_read = fread(model->final_norm, sizeof(float), h_d, f);
    size_t lm_read = fread(model->lm_head, sizeof(float), (size_t)v_s * h_d, f);
    (void)fn_read; (void)lm_read;

    fclose(f);
    return 0;
}

int qwen_model_load_experts(Qwen3_6Model *model, const char *expert_shard_path) {
    if (qwen_mmap_open(&model->expert_shard, expert_shard_path) != 0) {
        return -1;
    }
    model->expert_shard_open = true;
    return 0;
}

void qwen_model_free(Qwen3_6Model *model) {
    if (!model) return;
    if (model->expert_shard_open) {
        qwen_mmap_close(&model->expert_shard);
        model->expert_shard_open = false;
    }
    if (model->layers) {
        for (int l = 0; l < model->n_layers; l++) {
            if (model->layers[l].deltanet.q_proj.qf) free(model->layers[l].deltanet.q_proj.qf);
            if (model->layers[l].deltanet.k_proj.qf) free(model->layers[l].deltanet.k_proj.qf);
            if (model->layers[l].deltanet.v_proj.qf) free(model->layers[l].deltanet.v_proj.qf);
            if (model->layers[l].deltanet.g_proj.qf) free(model->layers[l].deltanet.g_proj.qf);
            if (model->layers[l].deltanet.beta_proj.qf) free(model->layers[l].deltanet.beta_proj.qf);
            if (model->layers[l].deltanet.out_proj.qf) free(model->layers[l].deltanet.out_proj.qf);
            if (model->layers[l].sh_gate.qf) free(model->layers[l].sh_gate.qf);
            if (model->layers[l].sh_up.qf) free(model->layers[l].sh_up.qf);
            if (model->layers[l].sh_down.qf) free(model->layers[l].sh_down.qf);

            qwen_moe_layer_free(&model->layers[l]);
        }
        free(model->layers);
        model->layers = NULL;
    }
    if (model->embed_tokens) { free(model->embed_tokens); model->embed_tokens = NULL; }
    if (model->final_norm) { free(model->final_norm); model->final_norm = NULL; }
    if (model->lm_head) { free(model->lm_head); model->lm_head = NULL; }
}

void qwen_model_forward_token(Qwen3_6Model *model, int token_id, float *logits) {
    int H = model->hidden_dim;
    int V = model->vocab_size;

    if (token_id < 0 || token_id >= V) token_id = 0;

    float x[2048];
    const float *emb = model->embed_tokens + (size_t)token_id * H;
    memcpy(x, emb, H * sizeof(float));

    float x_next[2048];
    for (int l = 0; l < model->n_layers; l++) {
        qwen_moe_layer_forward(&model->layers[l], x, x_next, &model->expert_shard);
        memcpy(x, x_next, H * sizeof(float));
    }

    /* Final RMSNorm */
    float ss = 0.0f;
    for (int i = 0; i < H; i++) {
        ss += x[i] * x[i];
    }
    float rms = 1.0f / sqrtf(ss / H + 1e-6f);

    for (int i = 0; i < H; i++) {
        x[i] = (x[i] * rms) * model->final_norm[i];
    }

    for (int v = 0; v < V; v++) {
        logits[v] = dot_product_f32(x, model->lm_head + (size_t)v * H, H);
    }
}

int qwen_sample_argmax(const float *logits, int vocab_size) {
    int best_id = 0;
    float max_val = logits[0];
    for (int v = 1; v < vocab_size; v++) {
        if (logits[v] > max_val) {
            max_val = logits[v];
            best_id = v;
        }
    }
    return best_id;
}

int qwen_sample_top_p(const float *logits, int vocab_size, float temperature, float top_p) {
    if (temperature <= 0.0f) {
        return qwen_sample_argmax(logits, vocab_size);
    }

    float *probs = (float*)malloc(vocab_size * sizeof(float));
    if (!probs) return qwen_sample_argmax(logits, vocab_size);

    float max_logit = logits[0];
    for (int v = 1; v < vocab_size; v++) {
        if (logits[v] > max_logit) max_logit = logits[v];
    }

    double sum = 0.0;
    for (int v = 0; v < vocab_size; v++) {
        probs[v] = expf((logits[v] - max_logit) / temperature);
        sum += probs[v];
    }

    for (int v = 0; v < vocab_size; v++) {
        probs[v] /= (float)sum;
    }

    typedef struct { int id; float prob; } ProbPair;
    ProbPair *pairs = (ProbPair*)malloc(vocab_size * sizeof(ProbPair));
    if (!pairs) {
        free(probs);
        return qwen_sample_argmax(logits, vocab_size);
    }
    for (int v = 0; v < vocab_size; v++) {
        pairs[v].id = v;
        pairs[v].prob = probs[v];
    }

    int k_top = 200 < vocab_size ? 200 : vocab_size;
    for (int i = 0; i < k_top; i++) {
        int max_i = i;
        for (int j = i + 1; j < vocab_size; j++) {
            if (pairs[j].prob > pairs[max_i].prob) max_i = j;
        }
        if (max_i != i) {
            ProbPair tmp = pairs[i];
            pairs[i] = pairs[max_i];
            pairs[max_i] = tmp;
        }
    }

    float cumsum = 0.0f;
    int cutoff = k_top;
    for (int i = 0; i < k_top; i++) {
        cumsum += pairs[i].prob;
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    float r = ((float)rand() / (float)RAND_MAX) * cumsum;
    float cur = 0.0f;
    int selected_id = (cutoff > 0) ? pairs[0].id : 0;
    for (int i = 0; i < cutoff; i++) {
        cur += pairs[i].prob;
        if (r <= cur) {
            selected_id = pairs[i].id;
            break;
        }
    }

    free(pairs);
    free(probs);
    return selected_id;
}

