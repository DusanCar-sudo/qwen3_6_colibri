#ifndef QWEN3_6_MOE_H
#define QWEN3_6_MOE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Architecture constants for Qwen 3.6 35B MoE */
#define QWEN3_6_HIDDEN_DIM          2048
#define QWEN3_6_MOE_INTER_DIM       1408
#define QWEN3_6_SHARED_INTER_DIM    512
#define QWEN3_6_N_ROUTED_EXPERTS    256
#define QWEN3_6_N_SHARED_EXPERTS    1
#define QWEN3_6_TOPK                8
#define QWEN3_6_N_HEADS             16
#define QWEN3_6_HEAD_DIM            128

/* Tensor Format Types */
typedef enum {
    QWEN_FMT_F32  = 0,
    QWEN_FMT_INT8 = 1,
    QWEN_FMT_INT4 = 2
} QwenTensorFmt;

/* Quantized/Dense Tensor Representation */
typedef struct {
    QwenTensorFmt fmt;
    int O;          /* Output dim (rows) */
    int I;          /* Input dim (cols) */
    int gs;         /* Group size for quant (0 = per-row) */
    float *qf;      /* F32 pointer */
    int8_t *q8;     /* INT8 weights */
    uint8_t *q4;    /* Packed INT4 weights */
    float *s;       /* Scale factors */
} QwenQT;

/* Expert Cache Slot (Stores 1 Routed Expert) */
typedef struct {
    int eid;             /* Expert ID (0 .. 255), -1 if empty */
    QwenQT g;            /* Gate projection */
    QwenQT u;            /* Up projection */
    QwenQT d;            /* Down projection */
    uint8_t *slab;       /* Memory slab backing (for mmap view or heap allocation) */
    float *fslab;        /* Scales slab backing */
    size_t slab_cap;     /* Slab capacity in bytes */
    size_t fslab_cap;    /* Float slab capacity */
    uint64_t last_used;  /* Access clock timestamp for LRU eviction */
    bool is_mmap;        /* True if zero-copy mmap view */
} QwenESlot;

/* LRU Cache Structure for MoE Layer */
typedef struct {
    QwenESlot *slots;    /* Dynamic array of cache slots */
    int capacity;        /* Maximum number of cached experts in RAM */
    int count;           /* Currently loaded slots */
    uint64_t global_clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
} QwenLRUCache;

/* Memory Mapping Shard Structure */
typedef struct {
    int fd;
    void *base_addr;
    size_t size;
    char filepath[512];
    bool is_mapped;
} QwenMmapShard;

/* Gated DeltaNet Linear Attention Structure */
typedef struct {
    int hidden_dim;
    int num_heads;
    int head_dim;
    
    /* Linear attention recurrence state [num_heads][head_dim][head_dim] */
    float *state_matrix; 
    
    /* Projection weights */
    QwenQT q_proj;
    QwenQT k_proj;
    QwenQT v_proj;
    QwenQT g_proj;      /* Output gate projection */
    QwenQT beta_proj;   /* DeltaNet update gate beta */
    QwenQT out_proj;    /* Final output projection */
} QwenGatedDeltaNet;

/* Qwen 3.6 MoE Layer Pipeline */
typedef struct {
    int layer_id;
    int hidden_dim;
    int moe_inter_dim;
    int n_routed_experts;
    int topk;
    bool is_full_attn;         /* True for self_attn layers (11, 15, 19, 23, 27, 31, 35, 39) */

    /* LayerNorm weights */
    float *input_layernorm;
    float *post_attn_layernorm;
    float *q_norm;             /* For self_attn layers */
    float *k_norm;             /* For self_attn layers */
    
    /* Gated DeltaNet linear attention module */
    QwenGatedDeltaNet deltanet;
    
    /* MoE Router weights & bias */
    float *router_weights;     /* [n_routed_experts, hidden_dim] */
    float *router_bias;        /* [n_routed_experts] */
    
    /* 1 Shared Expert (Always resident in memory) */
    QwenQT sh_gate;            /* [shared_inter, hidden] */
    QwenQT sh_up;              /* [shared_inter, hidden] */
    QwenQT sh_down;            /* [hidden, shared_inter] */
    
    /* LRU Cache for the 256 routed experts */
    QwenLRUCache expert_cache;
} QwenMoELayer;

/* Qwen 3.6 Full MoE Model Structure */
typedef struct {
    int n_layers;
    int hidden_dim;
    int moe_inter_dim;
    int shared_inter_dim;
    int n_routed_experts;
    int topk;
    int vocab_size;

    float *embed_tokens;       /* [vocab_size, hidden_dim] */
    QwenMoELayer *layers;      /* [n_layers] */
    float *final_norm;         /* [hidden_dim] */
    float *lm_head;            /* [vocab_size, hidden_dim] */

    QwenMmapShard expert_shard;
    bool expert_shard_open;
    void *backbone_mmap_ptr;   /* Backing memory pointer if mmapped */
    size_t backbone_mmap_size;
} Qwen3_6Model;

/* ========================================================================= */
/* Function Declarations                                                    */
/* ========================================================================= */

/* Tensor Helpers */
void qwen_qt_init_f32(QwenQT *qt, int O, int I, float *data);
void qwen_qt_init_int8(QwenQT *qt, int O, int I, int8_t *q8, float *scales);
void qwen_qt_free(QwenQT *qt);
void qwen_matmul_qt(const float *x, const QwenQT *w, float *out, int S);

/* Memory Mapping Functions */
int qwen_mmap_create_dummy_shard(const char *filepath, size_t size);
int qwen_mmap_open(QwenMmapShard *shard, const char *filepath);
void qwen_mmap_close(QwenMmapShard *shard);

/* LRU Cache & Expert Streaming Functions */
int qwen_lru_init(QwenLRUCache *cache, int capacity);
void qwen_lru_free(QwenLRUCache *cache);
QwenESlot* qwen_lru_lookup(QwenLRUCache *cache, int eid);
QwenESlot* qwen_lru_evict_and_prepare(QwenLRUCache *cache, int eid_to_load);
int qwen_expert_load_from_mmap(QwenMoELayer *layer, QwenMmapShard *shard, int eid, QwenESlot *slot);

/* Router & Expert Execution */
void qwen_router_select(const float *x, const float *router_w, const float *router_b, 
                        int hidden, int n_experts, int topk, 
                        int *out_eids, float *out_weights);

void qwen_expert_forward(const QwenESlot *slot, const float *x, float *out, int hidden, int inter);
void qwen_shared_expert_forward(const QwenMoELayer *layer, const float *x, float *out);

/* Gated DeltaNet Linear Attention */
int qwen_deltanet_init(QwenGatedDeltaNet *dn, int hidden, int num_heads, int head_dim);
void qwen_deltanet_free(QwenGatedDeltaNet *dn);
void qwen_deltanet_forward(QwenGatedDeltaNet *dn, const float *x, float *out);

/* Full MoE Layer Pipeline */
int qwen_moe_layer_init(QwenMoELayer *layer, int layer_id, int hidden, int moe_inter, 
                        int n_experts, int topk, int cache_capacity);
void qwen_moe_layer_free(QwenMoELayer *layer);
void qwen_moe_layer_forward(QwenMoELayer *layer, const float *x_in, float *x_out, QwenMmapShard *shard);

/* Full Model Pipeline Functions */
int qwen_model_init(Qwen3_6Model *model, int n_layers, int hidden, int moe_inter, 
                    int shared_inter, int n_experts, int topk, int vocab, int cache_capacity);
int qwen_model_load_backbone(Qwen3_6Model *model, const char *backbone_path);
int qwen_model_load_experts(Qwen3_6Model *model, const char *expert_shard_path);
void qwen_model_free(Qwen3_6Model *model);
void qwen_model_forward_token(Qwen3_6Model *model, int token_id, float *logits);

/* Sampling Helpers */
int qwen_sample_argmax(const float *logits, int vocab_size);
int qwen_sample_top_p(const float *logits, int vocab_size, float temperature, float top_p);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3_6_MOE_H */

