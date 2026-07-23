#include "qwen3_6_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

/* Timer Helper */
static double get_time_sec(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
#endif
}

/* Simple Byte / Tokenizer Helpers for Demo CLI */
typedef struct {
    char **tokens;
    int vocab_size;
} QwenVocab;

static QwenVocab g_vocab = {NULL, 0};

static void load_vocab_bin(const char *vocab_path) {
    FILE *f = fopen(vocab_path, "rb");
    if (!f) return;
    uint32_t count = 0;
    if (fread(&count, 4, 1, f) != 1) { fclose(f); return; }
    g_vocab.vocab_size = (int)count;
    g_vocab.tokens = (char**)calloc(count, sizeof(char*));
    for (uint32_t i = 0; i < count; i++) {
        uint16_t len = 0;
        if (fread(&len, 2, 1, f) != 1) break;
        g_vocab.tokens[i] = (char*)malloc(len + 1);
        if (len > 0) {
            if (fread(g_vocab.tokens[i], 1, len, f) != len) {}
        }
        g_vocab.tokens[i][len] = '\0';
    }
    fclose(f);
    printf("[QwenVocab] Loaded %d real tokens from '%s'\n", g_vocab.vocab_size, vocab_path);
}

static void encode_prompt(const char *prompt, int *tokens, int *n_tokens, int vocab_size) {
    int count = 0;
    tokens[count++] = 151644; // <|im_start|>
    tokens[count++] = 872;    // user
    tokens[count++] = 198;    // \n

    if (g_vocab.tokens && g_vocab.vocab_size > 0) {
        int len = (int)strlen(prompt);
        int pos = 0;
        while (pos < len) {
            int best_match_len = 0;
            int best_token_id = -1;

            // Simple greedy longest-prefix subword matching
            for (int id = 0; id < g_vocab.vocab_size; id++) {
                if (!g_vocab.tokens[id]) continue;
                const char *vtok = g_vocab.tokens[id];
                int vlen = (int)strlen(vtok);
                if (vlen > 0 && vlen <= (len - pos)) {
                    if (strncmp(prompt + pos, vtok, vlen) == 0) {
                        if (vlen > best_match_len) {
                            best_match_len = vlen;
                            best_token_id = id;
                        }
                    }
                }
            }

            if (best_token_id >= 0 && best_match_len > 0) {
                tokens[count++] = best_token_id;
                pos += best_match_len;
            } else {
                tokens[count++] = (unsigned char)prompt[pos];
                pos++;
            }
            if (count >= 1000) break;
        }
    } else {
        int len = (int)strlen(prompt);
        for (int i = 0; i < len; i++) {
            tokens[count++] = (unsigned char)prompt[i];
        }
    }

    tokens[count++] = 151645; // <|im_end|>
    tokens[count++] = 198;    // \n
    tokens[count++] = 151644; // <|im_start|>
    tokens[count++] = 77091;  // assistant
    tokens[count++] = 198;    // \n
    *n_tokens = count;
}

static void print_token(int token_id) {
    if (g_vocab.tokens && token_id >= 0 && token_id < g_vocab.vocab_size && g_vocab.tokens[token_id]) {
        const char *t = g_vocab.tokens[token_id];
        for (int i = 0; t[i] != '\0'; i++) {
            // Handle BPE space marker \u0120 (0xc4 0xa0)
            if ((unsigned char)t[i] == 0xc4 && (unsigned char)t[i+1] == 0xa0) {
                putchar(' ');
                i++;
            } else if ((unsigned char)t[i] == 0xc4 && (unsigned char)t[i+1] == 0x8a) {
                putchar('\n');
                i++;
            } else {
                putchar((unsigned char)t[i]);
            }
        }
    } else {
        printf("[%d]", token_id);
    }
    fflush(stdout);
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --backbone <file>  Path to backbone binary file (default: qwen3_6_backbone.bin)\n");
    printf("  --experts <file>   Path to expert shard file (default: qwen3_6_experts.shard)\n");
    printf("  --prompt <str>     Single prompt generation mode\n");
    printf("  --chat, -i         Interactive chat mode (default if no prompt provided)\n");
    printf("  --temp <float>     Sampling temperature (default: 0.7)\n");
    printf("  --top-p <float>    Top-p sampling threshold (default: 0.9)\n");
    printf("  --max-tokens <int> Maximum tokens to generate (default: 128)\n");
    printf("  --cache-cap <int>  LRU Expert Cache capacity per layer (default: 32)\n");
    printf("  --layers <int>     Number of layers (default: 4)\n");
    printf("  --vocab <int>      Vocab size (default: 1000)\n");
    printf("  --help, -h         Display this help message\n");
}

/* Auto-generate synthetic dummy files if not found */
static void ensure_dummy_files_exist(const char *backbone_path, const char *experts_path, int n_layers, int vocab_size) {
    FILE *fb = fopen(backbone_path, "rb");
    if (!fb) {
        printf("[Qwen3.6 CLI] Backbone file '%s' not found. Creating synthetic backbone file...\n", backbone_path);
        FILE *fw = fopen(backbone_path, "wb");
        if (fw) {
            uint32_t magic = 0x5157454E;
            uint32_t version = 1;
            int32_t n_l = n_layers, h_d = QWEN3_6_HIDDEN_DIM, m_i = QWEN3_6_MOE_INTER_DIM;
            int32_t s_i = QWEN3_6_SHARED_INTER_DIM, n_e = QWEN3_6_N_ROUTED_EXPERTS, tk = QWEN3_6_TOPK, v_s = vocab_size;

            fwrite(&magic, sizeof(uint32_t), 1, fw);
            fwrite(&version, sizeof(uint32_t), 1, fw);
            fwrite(&n_l, sizeof(int32_t), 1, fw);
            fwrite(&h_d, sizeof(int32_t), 1, fw);
            fwrite(&m_i, sizeof(int32_t), 1, fw);
            fwrite(&s_i, sizeof(int32_t), 1, fw);
            fwrite(&n_e, sizeof(int32_t), 1, fw);
            fwrite(&tk, sizeof(int32_t), 1, fw);
            fwrite(&v_s, sizeof(int32_t), 1, fw);

            float *zero_buf = (float*)calloc((size_t)v_s * h_d, sizeof(float));
            fwrite(zero_buf, sizeof(float), (size_t)v_s * h_d, fw);

            for (int l = 0; l < n_l; l++) {
                float *ln = (float*)malloc(h_d * sizeof(float));
                for (int i = 0; i < h_d; i++) ln[i] = 1.0f;
                fwrite(ln, sizeof(float), h_d, fw);
                free(ln);

                float *dn = (float*)calloc(h_d * h_d, sizeof(float));
                for (int i = 0; i < 6; i++) fwrite(dn, sizeof(float), h_d * h_d, fw);
                free(dn);

                float *post_ln = (float*)malloc(h_d * sizeof(float));
                for (int i = 0; i < h_d; i++) post_ln[i] = 1.0f;
                fwrite(post_ln, sizeof(float), h_d, fw);
                free(post_ln);

                float *sh_up = (float*)calloc(s_i * h_d, sizeof(float));
                float *sh_d = (float*)calloc(h_d * s_i, sizeof(float));
                fwrite(sh_up, sizeof(float), s_i * h_d, fw);
                fwrite(sh_up, sizeof(float), s_i * h_d, fw);
                fwrite(sh_d, sizeof(float), h_d * s_i, fw);
                free(sh_up); free(sh_d);

                float *rw = (float*)calloc(n_e * h_d, sizeof(float));
                float *rb = (float*)calloc(n_e, sizeof(float));
                fwrite(rw, sizeof(float), n_e * h_d, fw);
                fwrite(rb, sizeof(float), n_e, fw);
                free(rw); free(rb);
            }

            float *fn = (float*)malloc(h_d * sizeof(float));
            for (int i = 0; i < h_d; i++) fn[i] = 1.0f;
            fwrite(fn, sizeof(float), h_d, fw);
            free(fn);

            fwrite(zero_buf, sizeof(float), (size_t)v_s * h_d, fw);
            free(zero_buf);

            fclose(fw);
        }
    } else {
        fclose(fb);
    }

    FILE *fe = fopen(experts_path, "rb");
    if (!fe) {
        printf("[Qwen3.6 CLI] Expert shard '%s' not found. Creating synthetic expert shard...\n", experts_path);
        size_t expert_bytes = (size_t)(QWEN3_6_MOE_INTER_DIM * QWEN3_6_HIDDEN_DIM * 2 + QWEN3_6_HIDDEN_DIM * QWEN3_6_MOE_INTER_DIM + (QWEN3_6_MOE_INTER_DIM * 2 + QWEN3_6_HIDDEN_DIM) * sizeof(float));
        size_t total_size = (size_t)n_layers * QWEN3_6_N_ROUTED_EXPERTS * expert_bytes;
        qwen_mmap_create_dummy_shard(experts_path, total_size);
    } else {
        fclose(fe);
    }
}

static void run_generation(Qwen3_6Model *model, const char *prompt, int max_tokens, float temp, float top_p) {
    int prompt_tokens[1024];
    int n_prompt = 0;
    encode_prompt(prompt, prompt_tokens, &n_prompt, model->vocab_size);

    printf("\nAssistant > ");
    fflush(stdout);

    float *logits = (float*)malloc(model->vocab_size * sizeof(float));
    if (!logits) return;

    double t0 = get_time_sec();
    int current_token = 0;

    /* Prefill Prompt Tokens */
    for (int i = 0; i < n_prompt; i++) {
        current_token = prompt_tokens[i];
        qwen_model_forward_token(model, current_token, logits);
    }

    /* Generation Loop */
    int generated_count = 0;
    uint64_t initial_hits = 0, initial_misses = 0, initial_evictions = 0;
    for (int l = 0; l < model->n_layers; l++) {
        initial_hits += model->layers[l].expert_cache.hits;
        initial_misses += model->layers[l].expert_cache.misses;
        initial_evictions += model->layers[l].expert_cache.evictions;
    }

    for (int step = 0; step < max_tokens; step++) {
        int next_token = qwen_sample_top_p(logits, model->vocab_size, temp, top_p);
        print_token(next_token);
        generated_count++;

        if (next_token == 151643 || next_token == 151645) { /* EOS token */
            break;
        }

        current_token = next_token;
        qwen_model_forward_token(model, current_token, logits);
    }

    double t1 = get_time_sec();
    double dt = t1 - t0;
    double tok_per_sec = (dt > 0.0) ? ((double)generated_count / dt) : 0.0;

    uint64_t total_hits = 0, total_misses = 0, total_evictions = 0;
    for (int l = 0; l < model->n_layers; l++) {
        total_hits += model->layers[l].expert_cache.hits;
        total_misses += model->layers[l].expert_cache.misses;
        total_evictions += model->layers[l].expert_cache.evictions;
    }

    uint64_t hits = total_hits - initial_hits;
    uint64_t misses = total_misses - initial_misses;
    uint64_t evictions = total_evictions - initial_evictions;

    printf("\n\n-----------------------------------------------------\n");
    printf(" [Stats] %d tokens generated in %.3fs (%.1f tok/s)\n", generated_count, dt, tok_per_sec);
    printf(" [LRU Cache] Hits: %lu | Misses: %lu | Evictions: %lu\n",
           (unsigned long)hits, (unsigned long)misses, (unsigned long)evictions);
    printf("-----------------------------------------------------\n\n");

    free(logits);
}

int main(int argc, char **argv) {
    srand((unsigned int)time(NULL));

    const char *backbone_path = "qwen3_6_backbone.bin";
    const char *experts_path = "qwen3_6_experts.shard";
    const char *vocab_path = "qwen3_6_vocab.bin";
    const char *prompt = NULL;
    bool interactive = false;
    float temp = 0.7f;
    float top_p = 0.9f;
    int max_tokens = 128;
    int cache_cap = 32;
    int n_layers = 40;
    int vocab_size = 248320;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backbone") == 0 && i + 1 < argc) {
            backbone_path = argv[++i];
        } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            experts_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab-bin") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--chat") == 0 || strcmp(argv[i], "-i") == 0) {
            interactive = true;
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache-cap") == 0 && i + 1 < argc) {
            cache_cap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (!prompt) {
        interactive = true;
    }

    printf("=====================================================\n");
    printf(" Qwen 3.6 35B MoE Inference Engine CLI (C Pure)      \n");
    printf("=====================================================\n");
    printf(" Backbone Path : %s\n", backbone_path);
    printf(" Experts Shard : %s\n", experts_path);
    printf(" Layers Count  : %d\n", n_layers);
    printf(" Cache Capacity: %d experts / layer\n", cache_cap);
    printf(" Temperature   : %.2f | Top-P: %.2f\n", temp, top_p);
    printf("=====================================================\n\n");

    load_vocab_bin(vocab_path);

    ensure_dummy_files_exist(backbone_path, experts_path, n_layers, vocab_size);

    Qwen3_6Model model;
    if (qwen_model_init(&model, n_layers, QWEN3_6_HIDDEN_DIM, QWEN3_6_MOE_INTER_DIM, QWEN3_6_SHARED_INTER_DIM, QWEN3_6_N_ROUTED_EXPERTS, QWEN3_6_TOPK, vocab_size, cache_cap) != 0) {
        fprintf(stderr, "Error: Failed to initialize Qwen 3.6 MoE model.\n");
        return 1;
    }

    qwen_model_load_backbone(&model, backbone_path);
    if (qwen_model_load_experts(&model, experts_path) != 0) {
        fprintf(stderr, "Error: Failed to mmap expert shard file '%s'.\n", experts_path);
        qwen_model_free(&model);
        return 1;
    }

    if (prompt) {
        printf("User > %s\n", prompt);
        run_generation(&model, prompt, max_tokens, temp, top_p);
    }

    if (interactive) {
        char input_buf[1024];
        printf("Type '/exit' or '/quit' to stop.\n\n");
        while (1) {
            printf("User > ");
            fflush(stdout);
            if (!fgets(input_buf, sizeof(input_buf), stdin)) break;
            
            /* Strip trailing newline */
            size_t len = strlen(input_buf);
            if (len > 0 && input_buf[len - 1] == '\n') input_buf[len - 1] = '\0';
            if (strlen(input_buf) == 0) continue;

            if (strcmp(input_buf, "/exit") == 0 || strcmp(input_buf, "/quit") == 0) {
                printf("Exiting chat session.\n");
                break;
            }

            run_generation(&model, input_buf, max_tokens, temp, top_p);
        }
    }

    qwen_model_free(&model);
    return 0;
}
