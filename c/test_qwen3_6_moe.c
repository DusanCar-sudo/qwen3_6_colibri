#include "qwen3_6_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

int main(void) {
    printf("=====================================================\n");
    printf(" Qwen 3.6 35B MoE Layer & Expert Streaming Test Suite \n");
    printf("=====================================================\n\n");

    const char *shard_path = "/tmp/qwen3_6_experts.shard";
    size_t shard_size = 128 * 1024 * 1024; /* 128 MB dummy shard */

    /* 1. Test Dummy Shard Creation & Memory Mapping */
    printf("[1/5] Creating dummy expert shard & memory mapping...\n");
    if (qwen_mmap_create_dummy_shard(shard_path, shard_size) != 0) {
        fprintf(stderr, "FAIL: Failed to create dummy expert shard at %s\n", shard_path);
        return 1;
    }

    QwenMmapShard shard;
    if (qwen_mmap_open(&shard, shard_path) != 0) {
        fprintf(stderr, "FAIL: Failed to mmap shard %s\n", shard_path);
        return 1;
    }
    assert(shard.is_mapped == true);
    assert(shard.base_addr != NULL);
    printf(" -> SUCCESS: Memory mapped 128 MB shard successfully.\n\n");

    /* 2. Test LRU Cache Mechanics & Eviction */
    printf("[2/5] Testing LRU Cache structure (Capacity = 16 experts)...\n");
    QwenLRUCache cache;
    int cache_cap = 16;
    assert(qwen_lru_init(&cache, cache_cap) == 0);

    /* Fill cache with experts 0..15 */
    for (int i = 0; i < 16; i++) {
        QwenESlot *slot = qwen_lru_lookup(&cache, i);
        assert(slot == NULL); /* Must be miss initially */
        slot = qwen_lru_evict_and_prepare(&cache, i);
        assert(slot != NULL);
        assert(slot->eid == i);
    }
    assert(cache.count == 16);
    assert(cache.misses == 16);
    assert(cache.hits == 0);

    /* Access expert 0 to make it MRU (Most Recently Used) */
    QwenESlot *slot_0 = qwen_lru_lookup(&cache, 0);
    assert(slot_0 != NULL && slot_0->eid == 0);
    assert(cache.hits == 1);

    /* Now request expert 16 (Cache Full -> Evicts expert 1, since 0 was touched) */
    QwenESlot *slot_16 = qwen_lru_lookup(&cache, 16);
    assert(slot_16 == NULL);
    slot_16 = qwen_lru_evict_and_prepare(&cache, 16);
    assert(slot_16 != NULL);
    assert(slot_16->eid == 16);
    assert(cache.evictions == 1);

    /* Verify expert 0 is STILL in cache (hit) */
    QwenESlot *hit_0 = qwen_lru_lookup(&cache, 0);
    assert(hit_0 != NULL && hit_0->eid == 0);
    assert(cache.hits == 2);

    /* Verify expert 1 was evicted (miss) */
    QwenESlot *miss_1 = qwen_lru_lookup(&cache, 1);
    assert(miss_1 == NULL);

    printf(" -> SUCCESS: LRU Cache hits (%lu), misses (%lu), evictions (%lu) verified.\n\n",
           (unsigned long)cache.hits, (unsigned long)cache.misses, (unsigned long)cache.evictions);
    qwen_lru_free(&cache);

    /* 3. Test Gated DeltaNet Linear Attention */
    printf("[3/5] Testing Gated DeltaNet Linear Attention module...\n");
    QwenGatedDeltaNet deltanet;
    int hidden = QWEN3_6_HIDDEN_DIM;
    assert(qwen_deltanet_init(&deltanet, hidden, QWEN3_6_N_HEADS, QWEN3_6_HEAD_DIM) == 0);

    float x_in[QWEN3_6_HIDDEN_DIM];
    float attn_out[QWEN3_6_HIDDEN_DIM];
    for (int i = 0; i < hidden; i++) x_in[i] = (float)(i % 100) * 0.01f;

    qwen_deltanet_forward(&deltanet, x_in, attn_out);
    printf(" -> SUCCESS: Gated DeltaNet forward completed.\n\n");
    qwen_deltanet_free(&deltanet);

    /* 4. Test Router Top-K Selection (256 Experts) */
    printf("[4/5] Testing Top-K Routing over 256 experts...\n");
    float router_w[256 * QWEN3_6_HIDDEN_DIM];
    float router_b[256];
    for (int i = 0; i < 256 * hidden; i++) router_w[i] = (float)sin(i * 0.001);
    for (int i = 0; i < 256; i++) router_b[i] = 0.0f;

    int selected_eids[QWEN3_6_TOPK];
    float selected_weights[QWEN3_6_TOPK];
    qwen_router_select(x_in, router_w, router_b, hidden, 256, QWEN3_6_TOPK, selected_eids, selected_weights);

    float sum_weights = 0.0f;
    printf("Selected Experts: ");
    for (int k = 0; k < QWEN3_6_TOPK; k++) {
        printf("E#%d (w=%.4f) ", selected_eids[k], selected_weights[k]);
        sum_weights += selected_weights[k];
    }
    printf("\n");
    assert(fabsf(sum_weights - 1.0f) < 1e-4f);
    printf(" -> SUCCESS: Router selected top-%d experts with normalized probabilities (sum = %.4f).\n\n",
           QWEN3_6_TOPK, sum_weights);

    /* 5. End-to-End Pipeline Multi-Token Streaming Forward Test */
    printf("[5/5] Testing Full MoE Layer Pipeline forward over 10 streaming tokens...\n");
    QwenMoELayer layer;
    assert(qwen_moe_layer_init(&layer, 0, hidden, QWEN3_6_MOE_INTER_DIM, 256, QWEN3_6_TOPK, 32) == 0);

    float x_out[QWEN3_6_HIDDEN_DIM];
    for (int t = 0; t < 10; t++) {
        for (int i = 0; i < hidden; i++) {
            x_in[i] = (float)(i + t) * 0.005f;
        }
        qwen_moe_layer_forward(&layer, x_in, x_out, &shard);
    }

    printf(" -> LRU Cache Stats: Hits = %lu, Misses = %lu, Evictions = %lu\n",
           (unsigned long)layer.expert_cache.hits,
           (unsigned long)layer.expert_cache.misses,
           (unsigned long)layer.expert_cache.evictions);
    assert(layer.expert_cache.hits > 0 || layer.expert_cache.misses > 0);

    qwen_moe_layer_free(&layer);
    qwen_mmap_close(&shard);
    remove(shard_path);

    printf("\n=====================================================\n");
    printf(" ALL Qwen 3.6 MoE Tests Passed Successfully!          \n");
    printf("=====================================================\n");
    return 0;
}
