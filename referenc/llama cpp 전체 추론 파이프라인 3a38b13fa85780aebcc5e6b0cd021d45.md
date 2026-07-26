# llama.cpp 전체 추론 파이프라인

# 목차

1. 전체 한 장 요약
2. llama.cpp 전체 파이프라인
3. 모델 파일과 실행 Graph의 차이
4. 실행 진입점과 모델 로드 요청
5. `llama_model_loader`: GGUF를 Runtime 정보로 바꾸는 단계
6. Metadata, Hyperparameter, Vocabulary 로드
7. `create_tensor`: Weight 값을 만드는 함수가 아니다
8. Tensor Data 연결과 Backend Buffer 배치
9. CPU_Mapped, Backend Buffer, Repack의 차이
10. Context와 KV Cache 초기화
11. 문자열이 Token ID가 되는 과정
12. `llama_batch`와 `llama_ubatch`
13. `llama_decode`: 한 번의 Forward 요청이 시작되는 지점
14. Graph Input Tensor 준비
15. Input Embedding과 `ggml_get_rows`
16. `ggml_mul_mat`: 계산이 아니라 계산 Node 생성
17. Attention Graph Build
18. KV Cache Write와 Read
19. FFN Graph Build
20. Final Norm과 Output Projection
21. `ggml_build_forward_expand`: 실행 Graph 확정
22. Graph Dump로 함수의 Input/Output 확인하기
23. Backend Scheduler와 Graph Split
24. `supports_op`와 `supports_buft`
25. Backend 경계의 Tensor Copy
26. Backend `graph_compute`와 실제 계산
27. Qualcomm HTP Backend 사례
28. HTP0-REPACK과 Q4 Weight
29. HMX, HVX, Fallback 경로
30. Logits, Sampling, Detokenize
31. Prefill과 Decode가 같은 모델인데도 다르게 동작하는 이유
32. Decode 1 Step 전체 함수·Tensor 흐름
33. 성능 원인 분석 시 지켜야 할 증명 순서
34. 권장 Instrumentation 산출물
35. 코드 탐색 순서
36. 결론

---

# 1. 전체 한 장 요약

llama.cpp의 추론 과정은 GGUF 파일을 열어 그 안의 연산 Graph를 그대로 실행하는 구조가 아니다. **GGUF에는 모델의 구조를 설명하는 Metadata, Tokenizer 정보, Weight Tensor의 이름·Shape·Type·Offset, 그리고 실제 Weight Byte가 저장되어 있다.** llama.cpp는 이 파일을 읽어 Runtime에서 사용할 Weight Tensor 객체를 만들고, 각 Tensor의 데이터를 CPU 또는 가속기 Backend가 접근할 수 있는 Buffer에 연결한다.

**실제 추론이 시작되면 문자열은 Tokenizer를 거쳐 Token ID 배열이 된다**. 이 Token ID와 Position, Attention Mask, KV Cache Index가 `llama_batch` 또는 내부의 `llama_ubatch` 형태로 정리된다. 그다음 Architecture별 Graph Builder가 Input Embedding, Attention, KV Cache, FFN, Final Norm, Output Projection에 필요한 ggml Node를 만든다. 이 시점의 `ggml_get_rows`, `ggml_mul_mat`, `ggml_rms_norm` 호출은 계산을 수행하는 것이 아니라 “어떤 입력을 사용해 어떤 출력을 만들어야 하는가”를 기록한 계산 Node를 생성하는 단계다.

**Graph가 완성되면 Backend Scheduler가 각 Node를 CPU, GPU, HTP 같은 실행 장치에 배정**한다. Scheduler는 Operation 종류만 보지 않는다. **Weight와 Activation의 Type, Shape, Stride, 현재 Buffer 위치, Backend가 그 Buffer를 읽을 수 있는지, Backend 사이에 Copy가 가능한지를 함께 판단**한다. Backend가 달라지는 경계에는 Tensor Copy와 Synchronization이 추가된다.

마지막으로 각 Backend의 `graph_compute`가 **자신에게 배정된 Graph Split을 실행**한다. CPU Backend는 CPU Kernel을 호출하고, GPU나 HTP Backend는 범용 ggml Operation을 장치 전용 Kernel 또는 Command로 바꿔 실행한다. **모든 Layer가 끝나면 Output Projection이 Logits를 만들고, Sampling 로직이 다음 Token을 선택하며, Tokenizer가 그 ID를 문자열 조각으로 복원**한다.

따라서 “모델이 NPU에서 실행된다”는 말은 하나의 단일 상태가 아니다. 정확한 의미는 어떤 Weight가 어느 Buffer에 놓였는지, 어떤 Graph Node가 어느 Backend에 배정됐는지, Backend 경계에서 Copy가 몇 번 발생했는지, 실제로 어떤 Kernel 경로가 실행됐는지가 모두 확인되었다는 뜻이다.

---

# 2. llama.cpp 전체 파이프라인

아래 흐름은 특정 모델이나 특정 가속기에 종속되지 않는 llama.cpp의 전체 Runtime 구조를 단순화한 것이다.

```
[01] CLI / Application API
     모델 경로, Context, Batch, Backend 옵션 전달
        ↓
[02] Model Load Entry
     llama_model_load_from_file 계열
        ↓
[03] llama_model_loader
     GGUF Header, Metadata, Tensor Descriptor 파싱
        ↓
[04] Model Structure 생성
     Hyperparameter, Vocabulary, Weight Tensor 객체 정의
        ↓
[05] Tensor Data Load
     GGUF Weight Byte 연결, mmap/read, Backend Buffer 배치, 필요 시 Repack
        ↓
[06] Context 초기화
     KV Cache, Backend Scheduler, Runtime Buffer, Thread 설정
        ↓
[07] Tokenize / Batch 준비
     문자열 → Token ID → llama_batch / llama_ubatch
        ↓
[08] llama_decode / llama_encode
     Forward 요청 시작
        ↓
[09] Graph Input 설정
     Token, Position, Mask, KV Index를 Input Tensor에 기록
        ↓
[10] Graph Build
     Embedding → Attention → KV Cache → FFN → Output Projection Node 생성
        ↓
[11] Graph Finalize
     ggml_build_forward_expand로 Dependency Graph 확정
        ↓
[12] Backend Scheduler
     Node별 Backend 배정, Split 생성, Copy 경계 결정
        ↓
[13] Backend graph_compute
     CPU / GPU / HTP 전용 실행 경로 진입
        ↓
[14] Logits / Sampling
     다음 Token 선택
        ↓
[15] Detokenize
     Token ID → 문자열 조각
```

이 파이프라인은 크게 모델 로드 구간, Context 초기화 구간, Forward Graph 생성 구간, Backend 실행 구간, Sampling 구간으로 나눌 수 있다. 성능 문제가 발생했을 때는 어느 구간에서 문제가 생겼는지를 먼저 분리해야 한다. 모델 변환이나 Weight Type 문제인지, Tensor Buffer 배치 문제인지, Graph Scheduling 문제인지, 장치 Kernel 문제인지, Sampling 이후의 CPU 처리 문제인지가 서로 다른 원인이기 때문이다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 3. 모델 파일과 실행 Graph의 차이

GGUF는 실행 Graph 파일이 아니라 모델 Container다. GGUF 안에는 `general.architecture`, Embedding Dimension, Layer 수, Attention Head 수, Tokenizer Vocabulary 같은 Metadata가 들어 있다. 각 Weight Tensor에 대해서는 이름, 차원 수, Shape, Data Type, 실제 Byte가 시작되는 Offset이 기록된다.

반면 이번 Forward에서 실제로 실행할 Operation의 순서는 GGUF 안에 고정되어 있지 않다. 같은 모델이라도 Prefill인지 Decode인지, 입력 Token 수가 몇 개인지, KV Cache가 어느 위치까지 채워졌는지, 전체 Logits가 필요한지 마지막 Token의 Logits만 필요한지에 따라 만들어지는 Tensor Shape와 Graph 일부가 달라질 수 있다.

이 차이를 이해하지 못하면 “GGUF가 Q4이므로 Q4 Kernel이 실행된다”거나 “모델이 HTP에 올라갔으므로 전체 Graph가 HTP에서 실행된다”는 식의 잘못된 결론을 내리기 쉽다. GGUF의 Tensor Type은 저장 상태를 설명한다. 실제 실행 상태는 Graph Build, Buffer Placement, Scheduler, Backend Kernel까지 확인해야 한다.

---

# 4. 실행 진입점과 모델 로드 요청

**일반적인 Application은 `llama-cli`, 별도 Android Native Application, JNI Wrapper 또는 자체 서비스 코드에서 llama.cpp API를 호출**한다. 모델 로드의 대표 진입점은 `llama_model_load_from_file` 계열 함수다. 버전에 따라 내부 구현 함수가 하나 더 존재하거나 Parameter 구조가 달라질 수 있지만, 역할은 동일하다.

이 함수가 받는 주요 Input은 GGUF 모델 경로와 `llama_model_params`다. `llama_model_params`에는 어떤 Device를 사용할지, 몇 개 Layer를 가속기에 Offload할지, mmap을 사용할지, Tensor Split이나 Device 목록을 어떻게 설정할지 같은 모델 로드 정책이 들어간다. Output은 성공 시 `llama_model*`이며, 이 객체 안에는 Architecture 정보, Vocabulary, Weight Tensor, Backend Buffer 관련 상태가 연결된다.

```
Input
model path
llama_model_params
device / offload configuration
mmap 또는 direct I/O 설정
optional tensor override

Output
llama_model*
또는 오류 상태
```

여기서 중요한 점은 `llama_model*`이 곧 추론 Context는 아니라는 것이다. 모델 객체는 고정 Weight와 모델 구조를 보관한다. 실제 Sequence 상태, KV Cache, Scheduler, Runtime Activation Buffer는 이후 `llama_init_from_model` 계열 단계에서 만들어지는 `llama_context*`가 관리한다.

실제 코드 추적에서는 Application Entry에서 모델 로드 함수까지의 Caller Chain을 먼저 기록해야 한다. 팀 문서에는 함수명만 적는 것보다 “누가 이 함수를 호출하며, 어떤 설정값이 여기서 확정되는가”를 함께 적는 편이 유용하다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 5. `llama_model_loader`: GGUF를 Runtime 정보로 바꾸는 단계

`llama_model_loader`는 **GGUF 파일을 열어 llama.cpp 내부에서 사용할 수 있는 형태로 정리하는 중심 객체**다. 이 단계의 Input은 파일 경로, mmap 사용 여부, Tensor 검증 여부, Split GGUF 파일 목록, Metadata Override 같은 로드 옵션이다.

```cpp
metadata_ptr.reset(gguf_init_from_file(fname.c_str(), params));
metadata = metadata_ptr.get();
```

- **gguf_init_from_file 는 어디에 정의된 함수이며, 어떤 동작을 하는지?**
    
    `ggml/src/gguf.cpp` 에 정의되어 있으며, 다음 작업을 수행함
    
    - GGUF 파일을 바이너리 읽기 모드로 연다.
    - 실제 파싱을 `gguf_init_from_file_ptr()`에 위임한다.
    - 파일을 닫는다.
    - 파싱 결과인 `gguf_context*`를 반환한다.
    
    ```cpp
    struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
        FILE * file = ggml_fopen(fname, "rb"); // ggml 파일을 binary로 open
    
        if (!file) {
            GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
            return nullptr;
        }
    
        struct gguf_context * result = gguf_init_from_file_ptr(file, params);
        fclose(file);
        return result;
    }
    ```
    
    ```cpp
    struct gguf_context * gguf_init_from_file_ptr(FILE * file, struct gguf_init_params params) {
        if (!file) {
            return nullptr;
        }
    
        const int64_t cur = gguf_ftell(file);
        if (cur < 0) {
            return nullptr;
        }
    
        gguf_file_reader reader = {
            /*.file   = */ file,
            /*.offset = */ static_cast<uint64_t>(cur),
        };
        const struct gguf_reader gr(gguf_file_reader_callback, &reader, SIZE_MAX, reader.offset, gguf_reader::file_remain(file));
        return gguf_init_from_reader(gr, params);
    }
    
    struct gguf_buffer_reader {
        const uint8_t * data;
        size_t          size;
    };
    ```
    
    실제 GGUF 해석은 더 아래의 `gguf_init_from_reader()`에서 수행합니다.
    
    - GGUF Header 및 Version 확인
    - Metadata Key-Value 읽기
    - Tensor 이름 읽기
    - Tensor Shape 읽기
    - Tensor Type 읽기
    - Tensor Data Offset 읽기
    - 중복 Tensor 이름 검사
    - Alignment, 크기, Type 유효성 검사
    - `params.no_alloc` 설정에 따라 Tensor 메모리 할당 여부 결정
    
    ```cpp
    static struct gguf_context * gguf_init_from_reader(const struct gguf_reader & gr, struct gguf_init_params params) {
        struct gguf_context * ctx = new gguf_context;
    
        bool ok = true;
    
        // file magic
        {
            std::vector<char> magic;
            ok = ok && gr.read(magic, 4);
    
            if (!ok) {
                GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
    
            for (uint32_t i = 0; i < magic.size(); i++) {
                if (magic[i] != GGUF_MAGIC[i]) {
                    char c0 = isprint(magic[0]) ? magic[0] : '?';
                    char c1 = isprint(magic[1]) ? magic[1] : '?';
                    char c2 = isprint(magic[2]) ? magic[2] : '?';
                    char c3 = isprint(magic[3]) ? magic[3] : '?';
                    GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                    gguf_free(ctx);
                    return nullptr;
                }
            }
        }
    
        // header
        int64_t n_kv      = 0;
        int64_t n_tensors = 0;
    
        if (ok && gr.read(ctx->version)) {
            if (ok && ctx->version == 0) {
                GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
                ok = false;
            }
    
            /*
             * bit layout is different when reading non-native endian models.
             * assuming that the GGUF version is 3, the non-native endian model
             * would read it as 0x30000000. we can use the AND operation against
             * the last 4 hexadecimal digits to check if the model is the same
             * endianness as the host system.
            */
            if (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
                GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
                ok = false;
            }
    
            if (ok && ctx->version == 1) {
                GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
                ok = false;
            }
            if (ok && ctx->version > GGUF_VERSION) {
                GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                    __func__, ctx->version, GGUF_VERSION);
                ok = false;
            }
        } else {
            ok = false;
        }
    
        if (ok && gr.read(n_tensors)) {
            static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
            if (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/sizeof(gguf_tensor_info))) {
                GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_tensors, SIZE_MAX/sizeof(gguf_tensor_info));
                ok = false;
            }
        } else {
            ok = false;
        }
    
        if (ok && gr.read(n_kv)) {
            static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
            if (n_kv < 0 || n_kv > int64_t(SIZE_MAX/sizeof(gguf_kv))) {
                GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                        __func__, n_kv, SIZE_MAX/sizeof(gguf_kv));
                ok = false;
            }
        } else {
            ok = false;
        }
    
        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read header\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }
    
        // KV pairs
        {
            for (int64_t i = 0; ok && i < n_kv; ++i) {
                std::string key;
                gguf_type   type     = gguf_type(-1);
                bool        is_array = false;
                uint64_t    n        = 1;
    
                try {
                    ok = ok && gr.read(key);
                } catch (std::length_error &) {
                    GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                    ok = false;
                } catch (std::bad_alloc &) {
                    GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                    ok = false;
                }
                for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                    if (key == ctx->kv[j].key) {
                        GGML_LOG_ERROR("%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                        ok = false;
                    }
                }
                if (!ok) {
                    break;
                }
    
                ok = ok && gr.read(type);
                if (type == GGUF_TYPE_ARRAY) {
                    is_array = true;
                    ok = ok && gr.read(type);
                    ok = ok && gr.read(n);
                }
                if (!ok) {
                    break;
                }
    
                switch (type) {
                    case GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<float>      (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<bool>       (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<double>     (gr, ctx->kv, key, is_array, n); break;
                    case GGUF_TYPE_ARRAY:
                    default:
                        {
                            GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                            ok = false;
                        } break;
                }
            }
    
            if (!ok) {
                GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);
    
            const int alignment_idx = gguf_find_key(ctx, GGUF_KEY_GENERAL_ALIGNMENT);
            ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);
    
            if (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
                GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
                gguf_free(ctx);
                return nullptr;
            }
        }
    
        // read the tensor info
        for (int64_t i = 0; ok && i < n_tensors; ++i) {
            struct gguf_tensor_info info;
    
            // tensor name
            {
                std::string name;
                try {
                    ok = ok && gr.read(name);
                } catch (std::length_error &) {
                    GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                    ok = false;
                } catch (std::bad_alloc &) {
                    GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                    ok = false;
                }
                if (name.length() >= GGML_MAX_NAME) {
                    GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                    ok = false;
                    break;
                }
                ggml_set_name(&info.t, name.c_str());
    
                // make sure there are no duplicate tensor names
                for (int64_t j = 0; ok && j < i; ++j) {
                    if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                        GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__, info.t.name, j, i);
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok) {
                break;
            }
    
            // tensor shape
            {
                uint32_t n_dims = 0;
                ok = ok && gr.read(n_dims);
                if (n_dims > GGML_MAX_DIMS) {
                    GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                        __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                    ok = false;
                    break;
                }
                for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                    info.t.ne[j] = 1;
                    if (j < n_dims) {
                        ok = ok && gr.read(info.t.ne[j]);
                    }
    
                    // check that all ne are non-negative
                    if (info.t.ne[j] < 0) {
                        GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                            __func__, info.t.name, j, info.t.ne[j]);
                        ok = false;
                        break;
                    }
                }
    
                // check that the total number of elements is representable
                if (ok && ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                           (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                           (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2]))) {
    
                    GGML_LOG_ERROR("%s: total number of elements in tensor '%s' with shape "
                        "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                        __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }
    
            // tensor type
            {
                ok = ok && gr.read(info.t.type);
    
                // check that tensor type is within defined range
                if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                    GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                        __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                    ok = false;
                    break;
                }
                const size_t  type_size = ggml_type_size(info.t.type);
                const int64_t blck_size = ggml_blck_size(info.t.type);
    
                // check that row size is divisible by block size
                if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                        "not a multiple of block size (%" PRId64 ")\n",
                        __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                    ok = false;
                    break;
                }
    
                // check that the size of the tensor in bytes is representable
                if (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                    GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                        __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                    ok = false;
                    break;
                }
    
                // calculate byte offsets given the tensor shape and type
                info.t.nb[0] = type_size;
                info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
                }
            }
            if (!ok) {
                break;
            }
    
            // tensor data offset within buffer
            ok = ok && gr.read(info.offset);
    
            ctx->info.push_back(info);
        }
    
        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }
        GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);
    
        // we require the data section to be aligned, so take into account any padding
        if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment))) {
            GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }
    
        // store the current file offset - this is where the data section starts
        ctx->offset = gr.tell();
    
        // compute the total size of the data section, taking into account the alignment
        {
            ctx->size = 0;
            for (size_t i = 0; i < ctx->info.size(); ++i) {
                const gguf_tensor_info & ti = ctx->info[i];
                if (ti.offset != ctx->size) {
                    GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                        __func__, ti.t.name, ti.offset, ctx->size);
                    GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                    gguf_free(ctx);
                    return nullptr;
                }
                size_t padded_size = GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
                if (SIZE_MAX - ctx->size < padded_size) {
                    GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                        __func__, ti.t.name, ctx->size, padded_size);
                    gguf_free(ctx);
                    return nullptr;
                }
                ctx->size += padded_size;
            }
        }
    
        // load the tensor data only if requested
        if (params.ctx != nullptr) {
            // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
            // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
            //   the ggml_tensor structs to the appropriate locations in the binary blob
    
            // compute the exact size needed for the new ggml_context
            size_t mem_size = 0;
            if (params.no_alloc) {
                if (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                    GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                    gguf_free(ctx);
                    return nullptr;
                }
    
                const size_t overhead = n_tensors * ggml_tensor_overhead();
    
                mem_size = overhead;
            } else {
                if ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                    GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                    gguf_free(ctx);
                    return nullptr;
                }
    
                const size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();
    
                if (SIZE_MAX - overhead < ctx->size) {
                    GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                    gguf_free(ctx);
                    return nullptr;
                }
    
                mem_size = overhead + ctx->size;
            }
    
            struct ggml_init_params pdata = {
                /*mem_size   =*/ mem_size,
                /*mem_buffer =*/ nullptr,
                /*no_alloc   =*/ params.no_alloc,
            };
    
            *params.ctx = ggml_init(pdata);
            if (*params.ctx == nullptr) {
                GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
    
            struct ggml_context * ctx_data = *params.ctx;
    
            struct ggml_tensor * data = nullptr;
    
            if (!params.no_alloc) {
                data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);
    
                ok = ok && data != nullptr;
    
                if (ok) {
                    ggml_set_name(data, "GGUF tensor data binary blob");
                }
    
                // read the binary blob with the tensor data
                ok = ok && gr.read(data->data, ctx->size);
    
                if (!ok) {
                    GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                    ggml_free(ctx_data);
                    *params.ctx = nullptr;
                    gguf_free(ctx);
                    return nullptr;
                }
    
                ctx->data = data->data;
            }
    
            ggml_set_no_alloc(ctx_data, true);
    
            // create the tensors
            for (size_t i = 0; i < ctx->info.size(); ++i) {
                const struct gguf_tensor_info & info = ctx->info[i];
    
                struct ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);
    
                ok = ok && cur != nullptr;
    
                if (!ok) {
                    break;
                }
    
                ggml_set_name(cur, info.t.name);
    
                // point the data member to the appropriate location in the binary blob using the tensor info
                if (!params.no_alloc) {
                    cur->data = (char *) data->data + info.offset;
                }
            }
    
            if (!ok) {
                GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                return nullptr;
            }
    
            ggml_set_no_alloc(ctx_data, params.no_alloc);
        }
    
        return ctx;
    }
    
    ```
    

```cpp
struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(nullptr, nullptr, nullptr, path_model, splits, /*file*/ nullptr, params);
}
```

```cpp
static struct llama_model * llama_model_load_from_file_impl(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & path_model,
        std::vector<std::string> & splits,
        FILE * file,
        struct llama_model_params params) {
    {
        int n_sources_defined = 0;
        if (metadata != nullptr) {
            n_sources_defined++;
        }
        if (!path_model.empty()) {
            n_sources_defined++;
        }
        if (file != nullptr) {
            n_sources_defined++;
        }
        if (n_sources_defined != 1) {
            LLAMA_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
            return nullptr;
        }
    }
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }
		//model metadata, 형식 등에 문제가 없으면 llama_model을 load 함
    const auto [status, model] = llama_model_load(metadata, set_tensor_data, set_tensor_data_ud, path_model, splits, file, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        if (model) {
            llama_model_free(model);
        }
        return nullptr;
    }

    return model;
}
```

```cpp

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static std::pair<int, llama_model *> llama_model_load(struct gguf_context * metadata, llama_model_set_tensor_data_t set_tensor_data, void * set_tensor_data_ud,
        const std::string & fname, std::vector<std::string> & splits, FILE * file, llama_model_params & params) {
    try {
        llama_model_loader ml(metadata, set_tensor_data, set_tensor_data_ud, fname, splits, file, params.use_mmap, params.use_direct_io,
            params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();
        std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, params));

        bool ok = llama_prepare_model_devices(params, model_ptr.get()); 
        // 어떤 함수인지 파악해야함
        if (!ok) {
            return {-1, nullptr};
        }

        auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
        if (model == nullptr) {
            GGML_ABORT("fatal error: model does not implement llama_model_base");
        }

        // loading time will be recalculated after the first eval, so
        // we take page faults deferred by mmap() into consideration
        model->t_load_us = 0;
        time_meas tm(model->t_load_us);

        model->t_start_us = tm.t_start_us;

        model->hparams.vocab_only = params.vocab_only;
        model->hparams.no_alloc   = params.no_alloc;

        try {
            model->load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model->arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model->load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model->load_stats(ml);
        model->print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return {0, model_ptr.release()};
        }

        if (!model->load_tensors(ml)) {
            return {-2, nullptr};
        }

        return {0, model_ptr.release()};
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return {-1, nullptr};
    }
}

```

- **hparams 실제 어떻게 생겼는지**
    
    `src/llama-hparams.h` 내에 구조체 형태로 정의되어 있음
    
    ```cpp
    llama_hparams
    ├─ 모델 구조
    │  ├─ Embedding Dimension
    │  ├─ Layer 수
    │  ├─ Attention Head 수
    │  ├─ KV Head 수
    │  └─ FFN Dimension
    ├─ Attention / RoPE 설정
    ├─ Norm 설정
    ├─ MoE 설정
    └─ Runtime Load 정책
       ├─ vocab_only
       └─ no_alloc
    ```
    

Loader는 먼저 GGUF Header와 Metadata Key-Value Table을 읽는다. 그다음 Tensor Descriptor를 순회하면서 각 Tensor의 이름, Type, Shape, Byte 수, 파일 내 Offset을 확인한다. 모델이 여러 GGUF 파일로 분할되어 있다면 Tensor가 어느 Split 파일에 있는지도 함께 관리한다.

이 단계의 Output은 계산된 Activation이나 Logits가 아니다. Loader가 만드는 것은 “어떤 Weight가 어느 파일의 어느 Offset에 어떤 Type과 Shape로 들어 있는가”를 조회할 수 있는 Runtime Map이다.

```
GGUF file
→ gguf_context
→ metadata map
→ tensor descriptor map
→ tensor name과 file/offset의 연결
```

예를 들어 `blk.0.attn_q.weight`라는 Tensor가 발견되면 Loader는 그 Tensor의 Type이 Q4_0인지 F16인지, Shape가 무엇인지, 실제 Byte가 어느 위치에서 시작하는지를 기록한다. 이후 모델 구조를 만드는 코드가 이 정보를 이용해 `ggml_tensor` 객체를 정의하고 실제 Weight Byte를 연결한다.

팀에서 Loader를 분석할 때는 다음과 같은 로그가 가장 유용하다.

```
[MODEL-LOADER]
file=model.gguf
architecture=...
tensor_count=...
metadata_count=...
total_elements=...
total_bytes=...
use_mmap=true

[MODEL-TENSOR]
name=blk.0.attn_q.weight
type=Q4_0
shape=[...]
file_index=0
offset=...
nbytes=...
```

이 로그가 있으면 모델 파일 문제와 Runtime Graph 문제를 초기에 분리할 수 있다. Tensor가 애초에 기대한 이름과 Type으로 로드되지 않았다면 Scheduler나 Kernel을 조사하기 전에 GGUF 변환 및 Loader 단계부터 수정해야 한다.

- **GGUF를 Runtime 정보로 바꾸는 단계의 Input/Output 파이프라인**
    
    ```
    GGUF 파일 경로 + llama_model_params
    → llama_model_loader::llama_model_loader()
    → gguf_init_from_file()
    → gguf_init_from_file_ptr()
    → gguf_init_from_reader()
    → gguf_context
    → Metadata·Tensor Descriptor 순회 및 검증
    → weights_map·Split 정보·모델 크기 통계 생성
    → 초기화가 완료된 llama_model_loader
    → load_hparams() / load_vocab() / load_stats()
    → llama_model 내부 Runtime 정보
    ```
    
    ### 1. 최초 Input
    
    이 단계의 최초 Input은 GGUF 파일 자체와 모델 로드 정책이다. 코드에서는 주로 GGUF 파일 경로인 `fname`과 `llama_model_params`가 전달된다.
    
    ```
    Input Data
    
    fname
    = "/data/local/tmp/models/model.gguf"
    
    llama_model_params
    = {
        use_mmap,
        use_direct_io,
        check_tensors,
        vocab_only,
        no_alloc,
        kv_overrides,
        tensor_buft_overrides
    }
    ```
    
    `fname`은 첫 번째 GGUF 파일의 위치를 나타낸다. `llama_model_params`는 파일을 mmap으로 연결할지, Tensor 정보를 검증할지, Vocabulary만 읽을지, 실제 Tensor Buffer를 할당하지 않을지 같은 로드 정책을 담는다.
    
    이 Input은 `llama_model_loader::llama_model_loader()` 생성자로 전달된다.
    
    ```
    GGUF 파일 경로 + 로드 옵션
    → llama_model_loader::llama_model_loader()
    ```
    
    ### 2. GGUF 파일 파싱 요청
    
    `llama_model_loader` 생성자는 GGUF 파일을 직접 Byte 단위로 모두 해석하지 않고, 하위 GGUF Parser인 `gguf_init_from_file()`을 호출한다.
    
    ```
    fname + gguf_init_params
    → gguf_init_from_file()
    → gguf_context*
    ```
    
    이때 전달되는 `gguf_init_params`에는 Tensor 검증 여부와 실제 Tensor Data를 할당할지 여부 등이 포함된다.
    
    ```
    gguf_init_params
    = {
        no_alloc,
        check_tensors,
        ctx
    }
    ```
    
    `gguf_init_from_file()`은 파일을 바이너리 모드로 연 뒤 `gguf_init_from_file_ptr()`에 파일 포인터를 전달한다.
    
    ```
    GGUF 파일 경로
    → gguf_init_from_file()
    → FILE*
    → gguf_init_from_file_ptr()
    ```
    
    ### 3. GGUF Header와 Metadata 파싱
    
    `gguf_init_from_file_ptr()`은 실제 파싱 로직인 `gguf_init_from_reader()`로 처리를 넘긴다.
    
    ```
    FILE*
    → gguf_init_from_file_ptr()
    → GGUF Reader
    → gguf_init_from_reader()
    ```
    
    `gguf_init_from_reader()`는 파일 앞부분부터 GGUF 구조를 읽는다. 먼저 Magic Number, GGUF Version, Metadata 개수, Tensor 개수를 읽고 파일이 정상적인 GGUF인지 확인한다.
    
    ```
    Raw GGUF Header Byte
    → Header 파싱
    → {
        magic,
        version,
        metadata_count,
        tensor_count
    }
    ```
    
    예를 들어 중간 결과는 개념적으로 다음과 같은 형태다.
    
    ```
    GGUF Header
    
    magic          = "GGUF"
    version        = 3
    metadata_count = 35
    tensor_count   = 412
    ```
    
    Header 확인이 끝나면 Metadata Key-Value 영역을 읽는다.
    
    ```
    GGUF Metadata Byte
    → Key·Type·Value 파싱
    → Metadata Map
    ```
    
    Metadata Map은 개념적으로 다음과 같이 만들어진다.
    
    ```
    Metadata Map
    
    "general.architecture"              → "gemma"
    "general.name"                      → "model-name"
    "gemma.embedding_length"            → 2048
    "gemma.block_count"                 → 26
    "gemma.attention.head_count"        → 8
    "tokenizer.ggml.tokens"             → Token 문자열 배열
    "tokenizer.ggml.bos_token_id"       → 2
    "tokenizer.ggml.eos_token_id"       → 1
    ```
    
    이 단계의 Output은 아직 `llama_hparams`나 `llama_vocab`이 아니다. **GGUF에 저장된 Key와 Value를 그대로 조회할 수 있는 `gguf_context` 내부 Metadata 구조**다.
    
    ### 4. Tensor Descriptor 파싱
    
    Metadata 뒤에는 각 Weight Tensor의 Descriptor가 저장되어 있다. `gguf_init_from_reader()`는 Tensor마다 이름, 차원 수, Shape, Type, 파일 내 Offset을 읽는다.
    
    ```
    GGUF Tensor Descriptor Byte
    → Tensor별 Descriptor 파싱
    → Tensor Descriptor 배열
    ```
    
    Tensor 하나의 중간 결과는 개념적으로 다음과 같다.
    
    ```
    Tensor Descriptor
    
    name       = "blk.0.attn_q.weight"
    n_dims     = 2
    shape      = [2048, 2048]
    type       = GGML_TYPE_Q4_0
    offset     = 12345678
    n_elements = 4194304
    n_bytes    = 2359296
    ```
    
    이 Descriptor는 Weight 값 자체가 아니다. 해당 Weight가 어떤 이름과 Shape, Type을 가지며 실제 Weight Byte가 파일의 어느 위치에 저장되어 있는지를 나타내는 위치표다.
    
    ```
    Tensor Descriptor
    = Tensor 이름 + Shape + Type + File Offset
    ```
    
    모든 Descriptor가 파싱되면 `gguf_context` 안에는 Metadata와 Tensor Descriptor가 함께 정리된다.
    
    ```
    gguf_context
    
    metadata
    = Key-Value Map
    
    tensor_infos
    = [
        {
            name,
            shape,
            type,
            offset,
            size
        },
        ...
    ]
    ```
    
    ### 5. `gguf_context` 생성
    
    `gguf_init_from_reader()`가 성공하면 최종적으로 `gguf_context*`가 반환된다.
    
    ```
    GGUF Raw Byte
    → gguf_init_from_reader()
    → gguf_context*
    ```
    
    이 `gguf_context`는 GGUF 파일을 Runtime 코드가 조회할 수 있는 구조로 바꾼 첫 번째 Output이다.
    
    ```
    Intermediate Output 1
    
    gguf_context*
    = {
        GGUF Header,
        Metadata Map,
        Tensor Descriptor 배열,
        Tensor Data 시작 Offset,
        Alignment 정보
    }
    ```
    
    이 시점에는 아직 Tensor가 CPU나 HTP Buffer에 배치되지 않았고, Attention이나 FFN Graph도 생성되지 않았다.
    
    ### 6. Split GGUF 확인 및 추가 파일 로드
    
    모델이 여러 GGUF 파일로 나뉘어 있으면 `llama_model_loader`는 첫 번째 파일의 Metadata를 통해 전체 Split 개수를 확인한다.
    
    ```
    첫 번째 gguf_context
    → Split Metadata 확인
    → 전체 Split 파일 경로 결정
    ```
    
    예를 들면 다음과 같다.
    
    ```
    model-00001-of-00003.gguf
    model-00002-of-00003.gguf
    model-00003-of-00003.gguf
    ```
    
    각 Split 파일도 동일하게 `gguf_init_from_file()`을 통해 파싱된다.
    
    ```
    Split GGUF 파일 목록
    → 각 파일별 gguf_init_from_file()
    → gguf_context 배열
    ```
    
    중간 결과는 다음과 같은 형태다.
    
    ```
    files
    
    files[0]
    = {
        path = "model-00001-of-00003.gguf",
        gguf_context,
        tensor_data_offset
    }
    
    files[1]
    = {
        path = "model-00002-of-00003.gguf",
        gguf_context,
        tensor_data_offset
    }
    
    files[2]
    = {
        path = "model-00003-of-00003.gguf",
        gguf_context,
        tensor_data_offset
    }
    ```
    
    ### 7. Metadata Override 적용
    
    **`llama_model_params.kv_overrides`가 설정되어 있으면 GGUF Metadata 값을 Runtime 설정값으로 덮어쓴다.**
    
    ```
    GGUF Metadata
    + kv_overrides
    → Override가 반영된 Runtime Metadata
    ```
    
    예를 들어 GGUF의 Metadata 값과 Override 값이 다음과 같다면:
    
    ```
    GGUF Metadata
    "context_length" = 8192
    
    kv_overrides
    "context_length" = 4096
    ```
    
    이후 Loader를 통해 조회되는 Runtime 값은 다음처럼 처리된다.
    
    ```
    Runtime Metadata
    "context_length" = 4096
    ```
    
    원본 GGUF 파일이 변경되는 것은 아니며, 현재 실행에서 Loader가 반환하는 Metadata 조회 결과만 변경된다.
    
    ### 8. Tensor 위치표 생성
    
    `llama_model_loader`는 각 `gguf_context`의 Tensor Descriptor를 순회하면서 Tensor 이름으로 해당 Tensor의 파일과 Descriptor를 찾을 수 있는 Map을 만든다.
    
    ```
    각 Split의 Tensor Descriptor 배열
    → Tensor 이름 기준으로 등록
    → weights_map
    ```
    
    `weights_map`은 개념적으로 다음과 같다.
    
    ```
    weights_map
    
    "token_embd.weight"
    → {
        file_index = 0,
        tensor_info = {
            type,
            shape,
            offset,
            nbytes
        }
    }
    
    "blk.0.attn_q.weight"
    → {
        file_index = 1,
        tensor_info = {
            type,
            shape,
            offset,
            nbytes
        }
    }
    ```
    
    따라서 이후 코드에서는 전체 GGUF 파일을 다시 순차 탐색하지 않고 Tensor 이름으로 필요한 Weight의 위치를 찾을 수 있다.
    
    ```
    Tensor 이름
    → weights_map 조회
    → 해당 Split 파일 + Offset + Type + Shape
    ```
    
    ### 9. 모델 크기 통계 계산
    
    Tensor Descriptor를 순회하면서 전체 Tensor Element 수와 전체 Weight Byte 수도 계산된다.
    
    ```
    모든 Tensor의 element 수
    → 합산
    → n_elements
    
    모든 Tensor의 byte 수
    → 합산
    → n_bytes
    ```
    
    중간 결과는 다음과 같다.
    
    ```
    n_elements = 전체 Parameter Element 수
    n_bytes    = GGUF Tensor Data의 전체 Byte 크기
    ```
    
    이 값은 이후 `load_stats()`를 통해 `llama_model` 내부로 복사된다.
    
    ### 10. `llama_model_loader`의 최종 Output
    
    `llama_model_loader` 생성자는 별도의 반환값을 반환하지 않는다. 생성이 완료된 `llama_model_loader` 객체 자체가 이 단계의 최종 Output이다.
    
    ```
    Final Output 1
    
    **llama_model_loader
    = {
        metadata,
        files,
        gguf_context 배열,
        weights_map,
        split 정보,
        n_elements,
        n_bytes,
        mmap 및 파일 접근 설정,
        Metadata Override
    }**
    ```
    
    즉 Loader 단계만 기준으로 정리하면 다음과 같다.
    
    ```
    GGUF 파일 경로 + llama_model_params
    → llama_model_loader::llama_model_loader()
    → gguf_init_from_file()
    → gguf_init_from_file_ptr()
    → gguf_init_from_reader()
    → gguf_context
    → Split 파일 파싱
    → Metadata Override 적용
    → Tensor Descriptor 순회
    → weights_map 생성
    → n_elements·n_bytes 계산
    → 초기화가 완료된 llama_model_loader
    ```
    
    ### 11. Loader Output이 실제 모델 Runtime 정보로 변환되는 과정
    
    생성된 `llama_model_loader`는 이후 `llama_model`을 초기화하는 함수들의 Input으로 사용된다.
    
    ```
    llama_model_loader
    → load_hparams()
    → llama_hparams
    ```
    
    `load_hparams()`는 Loader의 Metadata Map에서 Architecture, Embedding Dimension, Layer 수, Attention Head 수, KV Head 수, FFN Dimension, RoPE 설정 등을 읽어 `llama_hparams`에 저장한다.
    
    ```
    Metadata Map
    → load_hparams()
    → llama_hparams
    ```
    
    결과는 개념적으로 다음과 같다.
    
    ```
    llama_hparams
    
    arch       = GEMMA
    n_embd     = 2048
    n_layer    = 26
    n_head     = 8
    n_head_kv  = 4
    n_ctx_train = 8192
    ```
    
    Tokenizer 관련 Metadata는 `load_vocab()`의 Input이 된다.
    
    ```
    Tokenizer Metadata
    → load_vocab()
    → llama_vocab
    ```
    
    결과는 다음과 같은 Runtime Tokenizer 구조다.
    
    ```
    llama_vocab
    
    token_id → token 문자열
    token 문자열 → token_id
    BOS token ID
    EOS token ID
    Special Token 정보
    Merge Rule
    Token Type
    ```
    
    Loader가 계산한 모델 크기 정보는 `load_stats()`를 통해 모델에 저장된다.
    
    ```
    loader.n_elements + loader.n_bytes
    → load_stats()
    → model 내부 모델 크기 통계
    ```
    
    따라서 모델 구조까지 포함한 최종 파이프라인은 다음과 같다.
    
    ```
    GGUF 파일 Byte + llama_model_params
    → llama_model_loader::llama_model_loader()
    → gguf_init_from_file()
    → gguf_init_from_reader()
    → gguf_context
    → Metadata Map + Tensor Descriptor 배열
    → Split 통합 + weights_map 생성
    → llama_model_loader
    → load_hparams()
    → llama_hparams
    → load_vocab()
    → llama_vocab
    → load_stats()
    → n_elements + n_bytes
    → Metadata·Hyperparameter·Vocabulary·Tensor 위치 정보가 연결된 llama_model
    ```
    
    이 단계의 최종 Output은 실제 Weight 계산 결과가 아니라, 이후 Tensor 객체 생성과 Weight Data 로드에 사용할 수 있도록 정리된 모델 Runtime 정보다.
    
    ```
    최종 Runtime 정보
    
    llama_model
    = {
        hparams,
        vocab,
        architecture,
        모델 크기 통계
    }
    
    llama_model_loader
    = {
        Tensor 이름별 파일 위치,
        Tensor Type,
        Tensor Shape,
        Tensor Offset,
        Split 파일 정보
    }
    ```
    
    실제 `ggml_tensor` 객체 생성, Weight Byte 연결, CPU·HTP Buffer 배치는 다음 단계인 `create_tensor()`와 `load_tensors()`에서 수행된다.
    

---

# 6. Metadata, Hyperparameter, Vocabulary 로드

**Loader가 읽은 Metadata는 모델 구조와 Tokenizer를 초기화하는 데 사용**된다. Architecture별 코드는 Embedding Dimension, Layer 수, Attention Head 수, KV Head 수, RoPE 설정, Norm Epsilon, Context 관련 값을 읽어 모델 Hyperparameter 구조에 저장한다.

이 값들은 단순 설명 정보가 아니다. 이후 Weight Tensor의 기대 Shape를 계산하고 Graph를 구성할 때 직접 사용된다. 예를 들어 Query Projection Weight의 Shape, KV Cache의 Head Dimension, Attention Score의 Shape는 이 Hyperparameter에서 결정된다.

Vocabulary 로드는 Token 문자열, Token Type, Merge Rule, BOS/EOS ID, 특수 Token 정보를 Runtime Tokenizer 구조로 옮기는 과정이다. 이 단계가 잘못되면 모델 Weight가 정상이어도 입력 문자열이 기대와 다른 Token ID로 변환되거나, EOS 처리와 Chat Template 동작이 달라질 수 있다.

Metadata 로드의 Input은 GGUF Metadata Key-Value이며, Output은 모델 Hyperparameter와 Vocabulary 객체다.

```
Input
GGUF metadata
tokenizer metadata
architecture key

Output
model hyperparameters
vocabulary / tokenizer state
special token mapping
```

Architecture가 다른 모델을 비교할 때는 이 단계부터 차이가 시작된다. Llama, Gemma, EXAONE 등은 Attention 구조, Norm 위치, FFN 구조, Tied Embedding 여부가 다를 수 있다. 그러나 그 차이를 처리하는 상위 파이프라인은 동일하다. llama.cpp는 Metadata를 읽고 Architecture별 Builder가 그 값에 맞는 Weight와 Graph를 구성하도록 연결한다.

---

# 7. `create_tensor`: Weight 값을 만드는 함수가 아니다

`create_tensor()`에 전달되는 값은 모델의 Hyperparameter, Tensor 이름, 기대 Shape, Optional·Duplicate 처리 Flag, Backend Buffer 후보 목록이다.

GGUF Tensor Descriptor와 **`ggml_context`가 함수 인자로 직접 전달되는 것은 아니다. 대신 `create_tensor()` 내부에서 Tensor 이름을 기준으로 `llama_model_loader`의 `weights_map`을 조회해, 앞 단계에서 읽어 둔 GGUF Tensor의 Type, Shape, 파일 번호, Offset 정보를 가져온다.**

```
Input
tensor name
expected dimensions
optional / duplicate policy
GGUF tensor descriptor
ggml context

Output
ggml_tensor*
name / type / shape / stride가 설정된 Weight Tensor 객체
```

Weight Tensor는 일반적으로 `op=NONE`에 가까운 Leaf Tensor다. 이 Tensor는 다른 Operation의 결과가 아니라 이미 GGUF에 존재하는 데이터를 가리킨다. 반면 `ggml_mul_mat`가 만드는 Tensor는 `op=MUL_MAT`이며 `src0`와 `src1`을 가지고, Backend가 Graph를 실행한 뒤에야 실제 결과 값이 채워진다.

| 구분 | Weight Tensor | Operation Result Tensor |
| --- | --- | --- |
| 대표 예시 | `blk.0.attn_q.weight` | `blk.0.attn_q` |
| `op` | `NONE` | `MUL_MAT` |
| 입력 `src[]` | 없음 | Weight와 Activation |
| 값의 출처 | GGUF Tensor Data | Backend Kernel 실행 결과 |
| 값이 확정되는 시점 | 모델 로드 | Graph Compute |

따라서 `create_tensor`에서 확인해야 할 것은 Weight Byte의 계산식이 아니라 이름 Mapping, 기대 Shape 검증, Type 확인, Optional Tensor 처리다. 실제 Byte 연결과 Backend Buffer 배치는 다음 단계에서 일어난다.

코드에서 호출한 `create_tensor()`는 `llama_model_base`의 멤버 함수입니다.

```cpp
ggml_tensor * llama_model_base::create_tensor(constLLM_TN_IMPL &tn,const std::initializer_list<int64_t> &ne,intflags) {GGML_ASSERT(ml!=nullptr);returncreate_tensor(*ml,tn,ne,flags);
}
```

이후 실제로 Loader의 `create_tensor()`를 호출합니다.

```cpp
ggml_tensor * llama_model_base::create_tensor(llama_model_loader &ml,constLLM_TN_IMPL &tn,const std::initializer_list<int64_t> &ne,intflags) {returnml.create_tensor(hparams,
        &pimpl->cpu_buft_list,pimpl->dev_input.buft_list,pimpl->dev_output.buft_list,buft_list_layer,tn,ne,flags
    );
}
```

따라서 Gemma4의 `create_tensor()`는 독립적으로 Tensor를 만드는 것이 아니라, **앞에서 GGUF를 읽은 동일한 Loader의 `create_tensor()`로 다시 들어갑니다.**

- `const buft_list_t * buft_list_layer = tn.bid == -1 ? nullptr : pimpl->dev_layer.at(tn.bid).buft_list;`
    
    `tn.bid`는 이 Tensor가 어느 Transformer Block 또는 Layer에 속하는지를 나타내는 인덱스로 사용됩니다.
    
    ```
    tn.bid = 0
    → 0번 Layer Tensor
    
    tn.bid = 15
    → 15번 Layer Tensor
    
    tn.bid = -1
    → 특정 Transformer Layer에 속하지 않는 Tensor
    ```
    
    예를 들어 다음 Tensor는 Layer Tensor입니다.
    
    ```
    blk.0.attn_q.weight
    → tn.bid = 0
    
    blk.15.ffn_up.weight
    → tn.bid = 15
    ```
    
    반면 다음 Tensor는 특정 반복 Layer 하나에 속하지 않습니다.
    
    ```
    token_embd.weight
    output_norm.weight
    output.weight
    ```
    
    그래서 `tn.bid == -1`이면 Layer 전용 Buffer 목록을 사용하지 않고 `nullptr`을 전달합니다.
    
    ```
    tn.bid == -1
    → buft_list_layer = nullptr
    ```
    
    Layer Tensor라면 해당 Layer에 설정된 Backend Buffer 후보 목록을 가져옵니다.
    
    ```
    tn.bid = 5
    → pimpl->dev_layer.at(5)
    → 5번 Layer의 Device 설정
    → buft_list
    ```
    
    `/src/llama-arc.cpp` 에서 정의되어있는 layer 확인 가능
    
- `return ml.create_tensor(hparams, &pimpl->cpu_buft_list, pimpl->dev_input.buft_list, pimpl->dev_output.buft_list, buft_list_layer, tn, ne, flags);`
    
    실제 작업은 `llama_model_loader::create_tensor()`에 위임합니다.
    
    각 인자는 다음 의미입니다.
    
    ```
    hparams
    = GGUF Metadata에서 읽은 모델 구조 정보
    
    cpu_buft_list
    = CPU에 Tensor를 만들 때 사용할 Buffer 후보
    
    dev_input.buft_list
    = Input Layer Tensor용 Buffer 후보
    
    dev_output.buft_list
    = Output Layer Tensor용 Buffer 후보
    
    buft_list_layer
    = 해당 Transformer Layer Tensor용 Buffer 후보
    
    tn
    = 생성할 Tensor 이름과 Layer 정보
    
    ne
    = 기대 Shape
    
    flags
    = Optional, Duplicate 등의 생성 조건
    ```
    
    전체 흐름은 다음과 같습니다.
    
    ```
    Gemma4 모델 코드
    → create_tensor("blk.0.attn_q.weight", {2304, 2048})
    → tn.bid 확인
    → 0번 Layer의 Buffer 후보 선택
    → ml.create_tensor() 호출
    → weights_map에서 GGUF Tensor 검색
    → Shape 검증
    → 적절한 Buffer Type 선택
    → Runtime ggml_tensor 생성
    ```
    

---

# 8. Tensor Data 연결과 Backend Buffer 배치

Tensor 객체가 만들어진 뒤에는 GGUF의 실제 Weight Byte를 Tensor와 연결해야 한다. mmap을 사용하는 경우 CPU가 파일의 해당 영역을 직접 참조할 수 있고, 별도 Backend Buffer가 필요한 경우에는 Source Byte를 Backend가 사용할 수 있는 Buffer로 복사하거나 변환한다.

이 단계의 Input은 `ggml_tensor*`, GGUF 파일과 Offset, Tensor Type과 Shape, Layer의 역할, Device 및 Buffer Type 후보 목록이다. Output은 실제 `tensor->data`, `tensor->buffer`, 선택된 Device와 Buffer Type이다.

```
Input
Weight Tensor 객체
GGUF file offset
Tensor Type / Shape
Layer category
Backend Device 및 Buffer 후보

Output
실제 Weight Byte와 연결된 Tensor
선택된 Buffer Type
필요한 경우 Backend 전용 Repack 결과
```

이 단계가 중요한 이유는 같은 Operation이라도 Weight가 어느 Buffer에 있느냐에 따라 Scheduler의 선택이 달라질 수 있기 때문이다. Backend가 `MUL_MAT` 자체를 지원해도 `src0` Weight가 Backend가 읽을 수 없는 Buffer에 있으면 해당 Node는 CPU에 남거나 별도 Copy가 필요하다.

Snapdragon HTP 사례에서는 Input Layer, Repeating Layer, Output Layer가 서로 다른 Device·Buffer 정책을 사용할 수 있다. 현재 관찰된 코드에서는 Input Layer를 CPU Device에 두는 정책이 존재한다. 그 결과 Input Embedding에서 사용하는 `token_embd.weight`는 CPU 또는 CPU_Mapped에 남고, Transformer Block의 Matmul Weight는 HTP0-REPACK 같은 가속기 전용 Buffer를 사용할 수 있다.

이 단계는 반드시 Tensor 단위 로그로 확인해야 한다.

```
[TENSOR-PLACEMENT]
name=token_embd.weight
layer_kind=INPUT
type=F32
device=CPU
buffer=CPU_Mapped
reason=input_layer_policy

[TENSOR-PLACEMENT]
name=blk.0.attn_q.weight
layer_kind=REPEATING
type=Q4_0
device=HTP0
buffer=HTP0-REPACK
repacked=true
```

“모델의 몇 개 Layer를 Offload했다”는 요약 로그만으로는 충분하지 않다. 실제 성능 분석에는 주요 Weight마다 Type, Device, Buffer, Repack 여부가 필요하다.

이 과정은 한 함수에서 한 번에 처리되지 않고 다음 3단계로 나뉩니다.

```
1. create_tensor()
Tensor Type·Shape 확인 + Buffer Type 결정
        ↓
2. Backend Buffer 할당
tensor->buffer / tensor->data 설정
        ↓
3. GGUF Weight Byte 로드
mmap 또는 파일 read
        ↓
ggml_backend_tensor_set()
CPU 복사 / GPU 업로드 / HTP Repack
        ↓
실제 Weight가 연결된 Runtime Tensor
```

여기서 **Layer 역할과 Buffer 후보 목록은 이미 `create_tensor()`에서 사용됩니다.** 이후 Weight 로드 단계는 “선택된 Buffer에 GGUF Byte를 어떻게 넣을 것인가”를 처리합니다.

- **`create_tensor()`에서 Buffer Type을 먼저 결정**
    
    모델별 `load_arch_tensors()`가 Tensor 생성을 요청합니다.
    
    ```
    layer.wq =create_tensor(tn(LLM_TENSOR_ATTN_Q,"weight",i),
        {n_embd,n_embd_head*n_head },0
    );
    ```
    
    이 요청은 다음 함수로 전달됩니다.
    
    ```
    llama_model_base::create_tensor(...)
        ↓
    llama_model_loader::create_tensor(...)
    ```
    
    **Loader는 `weights_map`에서 GGUF Tensor를 찾고, Shape를 확인한 뒤 CPU·Input·Output·Layer별 `buft_list` 중에서 사용할 Buffer Type을 선택**합니다.
    
    ```
    blk.0.attn_q.weight
    → weights_map에서 GGUF Tensor 검색
    → Type = Q4_0
    → Shape 일치 확인
    → Layer 0 Buffer 후보 검사
    **→ HTP0-REPACK 선택
    → HTP0-REPACK용 ggml_context에 Runtime Tensor 생성**
    ```
    
    이때 만들어진 Tensor는 대략 다음 상태입니다.
    
    ```
    tensor->name   = "blk.0.attn_q.weight"
    tensor->type   = Q4_0
    tensor->ne     = [2048, 2048]
    tensor->op     = GGML_OP_NONE
    tensor->buffer = 아직 할당 전
    tensor->data   = 아직 연결 전
    ```
    
    Loader의 `weights_map`에는 Tensor 이름을 기준으로 원본 GGUF Tensor 명세와 파일 위치가 보관됩니다.
    
- **선택된 Buffer Type별로 Backend Buffer를 할당**
    
    Tensor들이 Buffer Type별 `ggml_context`에 모두 만들어지면, `load_tensors()` 후반에서 각 Context에 필요한 Backend Buffer를 할당합니다.
    
    개념적인 호출 흐름은 다음과 같습니다.
    
    ```
    Buffer Type별 ggml_context
    → ggml_backend_alloc_ctx_tensors_from_buft()
    → ggml_backend_buft_alloc_buffer()
    → Context 안의 Tensor별 메모리 위치 계산
    → ggml_backend_tensor_alloc()
    → tensor->buffer 설정
    → tensor->data 설정
    ```
    
    Allocator는 Backend Buffer의 기준 주소에 Tensor별 Offset을 더해 Tensor 저장 위치를 계산한 뒤 `ggml_backend_tensor_alloc()`을 호출합니다.
    
    개념적으로는 다음과 같습니다.
    
    ```
    tensor_address =buffer_base+tensor_offset;ggml_backend_tensor_alloc(backend_buffer,tensor,tensor_address
    );
    ```
    
    할당 후에는 다음 상태가 됩니다.
    
    ```
    tensor->buffer
    = 이 Tensor를 소유하는 Backend Buffer
    
    tensor->data
    = 해당 Buffer 안에서 Tensor 데이터가 위치한 주소
    ```
    
    다만 `tensor->data`의 의미는 Backend에 따라 다릅니다.
    
    ```
    CPU Buffer
    → CPU가 읽을 수 있는 일반 메모리 주소
    
    CPU_Mapped Buffer
    → mmap된 GGUF 파일 영역의 주소
    
    GPU / HTP Buffer
    → Backend가 관리하는 Device Memory 위치
    → CPU에서 일반 포인터처럼 읽을 수 있다고 가정하면 안 됨
    ```
    
    Backend Buffer가 Tensor에 붙은 뒤에는 Backend별 `init_tensor` Hook도 호출될 수 있습니다. 이 Hook은 특정 Backend가 Tensor 초기화 시 별도 Descriptor나 내부 상태를 만들어야 할 때 사용합니다
    
- `weights_map`에서 GGUF 파일과 Offset을 찾음
    
    Runtime Tensor가 만들어지고 Buffer도 할당되면, Loader는 Tensor 이름으로 원본 Weight 위치를 다시 찾습니다.
    
    ```
    Runtime Tensor
    name = "blk.0.attn_q.weight"
    
    → weights_map["blk.0.attn_q.weight"]
    
    → llama_tensor_weight
    {
        file = 원본 GGUF 파일,
        idx  = Split 파일 번호,
        offs = Weight Byte 시작 위치,
        tensor = 원본 Tensor 명세
    }
    ```
    
    즉 로드에 필요한 Source 정보는 다음과 같이 만들어집니다.
    
    ```
    Source GGUF 파일
    = files[weight.idx]
    
    Source Offset
    = weight.offs
    
    복사 크기
    = ggml_nbytes(runtime_tensor)
    ```
    
    `weights_map` 조회와 Tensor Shape 검증은 `get_weight()`, `get_tensor_meta()`, `check_tensor_dims()` 경로에서 처리됩니다
    

---

# 9. CPU_Mapped, Backend Buffer, Repack의 차이

CPU_Mapped는 GGUF 파일의 Weight Byte를 mmap 형태로 CPU Address Space에 연결한 상태다. 파일 전체를 별도 Heap에 복사하지 않고 필요한 영역을 읽을 수 있다는 장점이 있다. CPU Backend는 이 데이터를 직접 사용할 수 있지만, 다른 Backend가 동일한 Memory를 직접 읽을 수 있는지는 Backend와 Buffer 구현에 따라 다르다.

Backend Buffer는 CPU, GPU, HTP 등 특정 Backend가 Tensor를 저장하고 접근하도록 정의된 Memory 영역이다. Runtime Activation은 Backend Buffer에 할당되는 경우가 많고, Weight도 Backend가 요구하는 형식에 맞춰 별도 Buffer로 옮길 수 있다.

Repack은 Quantization과 다르다. Quantization은 Weight의 숫자 표현 정밀도를 바꾸는 작업이다. Repack은 이미 정해진 Weight 값과 Quant Type을 유지하면서 Kernel이 읽기 좋은 순서로 Byte Layout을 바꾸는 작업이다.

예를 들어 Q4_0 Weight가 GGUF 안에서 Row 중심 순서로 저장되어 있더라도 HTP Kernel이 여러 Output Row와 K 구간을 Tile 단위로 읽는다면, 모델 로드 시 Weight를 Tile-major 형태로 재배치할 수 있다. 이 결과가 HTP0-REPACK 같은 Buffer다.

```
GGUF Q4_0 Byte
→ 값과 Scale은 유지
→ Tile 단위 순서 변경
→ Alignment와 Padding 적용
→ HTP0-REPACK Buffer
```

Repack의 정확한 Tile 크기와 Byte Order는 HTP Backend 구현을 직접 확인해야 한다. 문서에서 개념적인 32×32 Tile을 예로 들 수는 있지만, 실제 구현 확인 없이 고정된 규격으로 단정하면 안 된다.

---

# 10. Context와 KV Cache 초기화

모델 객체가 고정 Weight와 구조를 관리한다면 `llama_context`는 실제 추론 상태를 관리한다. Context 생성의 대표 진입점은 `llama_init_from_model` 계열이다. Input은 `llama_model*`과 `llama_context_params`이며, Output은 `llama_context*`다.

Context Parameter에는 Context Length, Batch 크기, Micro-batch 크기, Thread 수, KV Cache Type, Flash Attention 사용 여부, Backend 관련 실행 설정이 포함된다. Context 초기화 과정에서는 Runtime Activation을 위한 Memory 계획, Backend Scheduler, KV Cache가 만들어진다.

KV Cache는 각 Layer에서 과거 Token의 Key와 Value를 저장한다. Decode는 매 Step마다 현재 Token의 K/V만 계산하고 Cache에 Append한 뒤 과거 Cache를 재사용한다. 따라서 KV Cache는 모델 파일에 포함된 고정 Weight가 아니라 Context가 소유한 Runtime 상태다.

```
Input
llama_model*
context length
batch / ubatch
KV type
backend execution parameters

Output
llama_context*
KV cache
backend scheduler
runtime buffers
sequence state
```

Context Length를 2048로 설정했다고 해서 항상 2048개 Token이 모두 유효한 것은 아니다. 실제 유효 범위는 현재 Sequence의 Position과 Cache Slot 상태로 관리된다. 남은 Cache 공간은 할당되어 있을 수 있지만 Attention이 읽어야 하는 범위는 Mask와 Index Tensor가 제한한다.

`valid_length`, `seq_id`, Cache Slot 상태는 매 Step마다 별도 파일로 생성되는 정보가 아니다. Context 내부의 Runtime Metadata와 Graph Input Tensor로 유지된다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 11. 문자열이 Token ID가 되는 과정

사용자가 입력한 문자열은 먼저 Tokenizer를 거쳐 Token ID 배열이 된다. 이 과정은 Transformer Layer의 일부가 아니며 보통 CPU에서 처리된다.

```
"Hello."
→ Tokenizer
→ [token_0, token_1, token_2]
```

Tokenizer 함수의 Input은 UTF-8 문자열, BOS 추가 여부, Special Token 처리 옵션이다. Output은 정수 Token ID 배열이다. 이후 각 Token ID는 Position, Sequence ID와 함께 `llama_batch`에 들어간다.

Tokenizing과 Input Embedding은 구분해야 한다. Tokenizing은 문자열을 정수 ID로 바꾸는 규칙 기반 처리다. Input Embedding은 `token_embd.weight`에서 해당 ID의 Row를 읽어 Hidden Vector를 만드는 Graph Operation이다.

| 단계 | Input | Output | 대표 실행 위치 |
| --- | --- | --- | --- |
| Tokenize | 문자열 | Token ID 배열 | CPU |
| Input Embedding | Token ID + Embedding Weight | Hidden Vector | CPU 또는 Backend |
| Detokenize | Token ID | 문자열 조각 | CPU |

Tokenizer 차이가 있는 모델을 Benchmark할 때는 동일한 문자열을 넣었더라도 실제 Token 수가 다를 수 있다. 입력 길이를 255로 맞추는 실험에서는 단순히 같은 문자열을 사용하는 것이 아니라 각 모델에서 실제 Token Count가 255인지 기록해야 한다.

---

# 12. `llama_batch`와 `llama_ubatch`

`llama_batch`는 Application이 llama.cpp에 전달하는 Forward 입력 구조다. Token ID, 직접 제공하는 Embedding, Position, Sequence ID, Logits 출력 요청 여부 등을 담는다.

내부에서는 Batch를 한 번에 처리 가능한 크기로 나누기 위해 `llama_ubatch`가 사용될 수 있다. UBatch는 Micro-batch로 이해하면 된다. 큰 Prefill 입력을 여러 조각으로 나누거나, Backend가 처리하기 적합한 단위로 정리할 때 사용된다.

`llama_batch`의 Input은 Application이 준비한 Token과 Sequence 정보다. `llama_ubatch`의 Output은 Graph Builder와 Input Setter가 바로 사용할 수 있는 정리된 Runtime 입력이다.

```
Application token array
→ llama_batch
→ batch allocation / split
→ llama_ubatch
→ graph input tensor
```

UBatch에는 현재 Token 수, Embedding 사용 여부, Position, Sequence 관련 정보가 포함된다. Prefill에서는 `n_tokens`가 여러 개이고 Decode에서는 보통 1이다. 이 차이는 이후 모든 Matmul과 Attention Tensor의 Shape를 바꾼다.

---

# 13. `llama_decode`: 한 번의 Forward 요청이 시작되는 지점

`llama_decode`는 Decoder-only 모델에서 한 번의 Forward 처리를 요청하는 대표 API다. Input은 `llama_context*`와 `llama_batch`다. Output은 성공 또는 오류 상태이며, 계산된 Logits와 Embedding은 Context가 관리하는 Output Buffer를 통해 조회한다.

```
Input
llama_context*
llama_batch

Side Effect
KV Cache 업데이트
Runtime Output Buffer 갱신
Sequence 상태 변경

Output
status code
```

`llama_decode` 내부에서는 Batch Validation, UBatch 분할, KV Cache Slot 준비, Graph Build, Scheduler 실행, Output 수집이 이어진다. 버전에 따라 이 과정이 여러 내부 함수와 객체로 분리되어 있으므로, Source 분석에서는 `llama_decode` 한 함수만 보는 것이 아니라 내부 Call Chain을 끝까지 따라가야 한다.

성능 측정 시 `llama_decode` 전체 시간은 Graph Build, Scheduler, Backend Compute, Copy, Output 수집을 모두 포함할 수 있다. Backend Kernel 시간과 동일한 값이 아니다. 따라서 Decode가 느릴 때는 전체 API 시간만 비교하지 말고 내부 단계를 분리해야 한다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 14. Graph Input Tensor 준비

Graph Builder가 `inp_tokens`, `inp_pos`, Attention Mask 같은 Input Tensor 객체를 만들면, Input Setter가 현재 UBatch의 값을 실제 Tensor Data에 기록한다.

대표적인 Input Tensor는 Token ID, Position, Output ID, Attention Mask, KV Cache Write Index다. Tensor 이름과 정확한 Type은 Architecture와 llama.cpp 버전에 따라 달라질 수 있다.

| Input Tensor | 의미 | 주요 Source |
| --- | --- | --- |
| `inp_tokens` | 현재 처리할 Token ID | UBatch Token |
| `inp_pos` | 각 Token의 Position | UBatch Position |
| `out_ids` | Logits가 필요한 Token 위치 | Batch Output Flag |
| KQ Mask | Causal Attention 허용 범위 | Sequence와 KV 상태 |
| K/V Index | Cache에 쓸 Slot | KV Context |

이 단계에서 Function Input은 UBatch와 KV Cache 상태다. Function Output은 새 Tensor 객체가 아니라 이미 만들어진 Input Tensor의 Data가 채워지는 Side Effect인 경우가 많다.

```
Input
ubatch token / position / sequence
KV slot information

Output
inp_tokens data
inp_pos data
mask data
cache index data
```

Graph Dump에는 Input Tensor가 `op=NONE`인 Leaf로 나타난다. 계산 Node가 아니므로 Backend Kernel을 실행하지 않지만, 어느 Buffer에 놓이는지가 첫 번째 Compute Node의 Backend 배치에 영향을 준다.

---

# 15. Input Embedding과 `ggml_get_rows`

Input Embedding은 Token ID에 해당하는 Weight Row를 꺼내 Hidden Vector를 만드는 단계다. ggml Graph에서는 일반적으로 `GET_ROWS` Operation으로 표현된다.

```cpp
inp_embd = ggml_get_rows(ctx, token_embd_weight, inp_tokens);
```

`src0`는 `token_embd.weight`, `src1`은 정수 Token ID Tensor다. Output은 `[embedding_dimension, n_tokens]` 형태의 Activation Tensor다. 정확한 차원 순서는 ggml의 `ne[]` 기준으로 Graph Dump에서 확인해야 한다.

```
Input
src0 = token_embd.weight
src1 = inp_tokens

Output
dst = inp_embd
op = GET_ROWS
```

현재 Snapdragon HTP 사례에서는 Input Layer를 CPU에 유지하는 정책으로 인해 Input Embedding의 Weight와 Input Token이 CPU Buffer에 놓이고 `GET_ROWS`도 CPU Split에서 실행되는 구조가 관찰되었다. 이후 Transformer Block이 HTP에 배정되면 `inp_embd`가 CPU에서 HTP Buffer로 복사된다.

```
CPU GET_ROWS
→ inp_embd 생성
→ CPU→HTP Copy
→ HTP Transformer Block
```

여기서 주의할 점은 `GET_ROWS` Operation 자체가 HTP에서 전면 미지원이라는 뜻은 아니라는 것이다. HTP 내부의 다른 Tensor에 대한 `GET_ROWS`가 지원될 수 있다. 현재 Input Embedding이 CPU인 이유는 Operation 이름 하나보다 Input Layer 정책과 Tensor Buffer 위치의 영향이 크다.

또한 CPU `GET_ROWS`가 존재한다는 사실만으로 Decode 성능 저하의 전체 원인을 확정할 수 없다. Copy Byte와 시간, Split 수, HTP Kernel 시간까지 분리해 확인해야 한다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 16. `ggml_mul_mat`: 계산이 아니라 계산 Node 생성

Graph Build 중 다음과 같은 코드가 호출된다고 가정한다.

```cpp
q = ggml_mul_mat(ctx, wq, normed_hidden);
```

이 호출이 반환되는 시점에는 Q Projection의 숫자 계산이 완료되지 않았다. 함수는 `op=MUL_MAT`인 새 `ggml_tensor`를 만들고, `src0`에 Weight, `src1`에 Activation을 연결한다.

```
Input
src0 = Weight Tensor
src1 = Activation Tensor

Output
MUL_MAT Result Tensor
op = MUL_MAT
src[0] = Weight
src[1] = Activation
```

Operation Result Tensor는 “값을 담는 Buffer”이면서 동시에 “그 값을 어떻게 만들 것인지 기록한 Graph Node”다. 이 이중 역할이 ggml Graph를 처음 볼 때 가장 혼동되는 부분이다.

Backend 실행 전에는 Output Tensor의 Shape와 Type, Dependency는 정해져 있지만 실제 값은 계산되지 않는다. Scheduler가 Backend를 결정하고 해당 Backend의 `graph_compute`가 Node를 실행한 뒤에야 Destination Buffer에 결과가 기록된다.

따라서 Source Code에서 `ggml_mul_mat` 호출 위치를 찾았다고 해서 실제 CPU나 HTP Matmul Kernel을 찾은 것은 아니다. Graph 생성 경로와 Kernel 실행 경로는 별도로 추적해야 한다.

---

# 17. Attention Graph Build

Architecture별 Attention Builder는 Input Activation을 받아 여러 개의 작은 ggml Node를 연결한다. 함수 이름은 버전과 Architecture에 따라 `build_attn` 계열 Helper 또는 Architecture별 `build_graph` 내부 코드로 존재할 수 있다.

Attention Builder의 주요 Input은 Layer Index, 현재 Hidden State, Attention Norm Weight, Q/K/V/O Projection Weight, Position Tensor, Attention Mask, KV Cache Context다. Output은 Attention 결과와 Residual이 반영된 다음 Activation Tensor다.

```
Input Hidden
→ RMS Norm
→ Q Projection
→ K Projection
→ V Projection
→ RoPE 또는 Position 처리
→ KV Cache Write
→ Q와 Cache K의 Score 계산
→ Mask와 Softmax
→ Cache V 가중합
→ Output Projection
→ Residual Add
```

Q/K/V Projection에서 `src0`는 각각 `attn_q.weight`, `attn_k.weight`, `attn_v.weight`이고 `src1`은 Norm이 적용된 Activation이다. 각 Projection Output은 Runtime Activation이며 다음 RoPE와 Cache Node의 Input이 된다.

Attention Builder가 끝났을 때도 실제 Norm, Matmul, Softmax는 실행되지 않았다. 반환된 Tensor는 Attention 부분 Graph의 마지막 Node를 가리킨다. Backend 실행은 Scheduler 이후에 일어난다.

팀 문서에서 Attention을 설명할 때는 “Attention 함수의 Input/Output”만 적는 것보다 내부 Node별 `src0`, `src1`, Output을 함께 기록해야 한다. 그래야 Weight Type과 Backend 배치를 Layer별로 분석할 수 있다.

---

# 18. KV Cache Write와 Read

Decode에서 현재 Token의 K와 V는 Cache에 추가되고, Attention은 과거 Token의 K/V와 현재 값을 함께 읽는다. llama.cpp의 KV Cache 구현은 버전에 따라 구조가 바뀔 수 있지만, Graph 관점의 Input/Output은 동일하게 정리할 수 있다.

K Cache Write의 Input은 RoPE가 적용된 현재 K, Cache Write Index, Layer 정보다. Output은 새로운 독립 Tensor라기보다 Cache Tensor의 특정 위치가 갱신되는 Side Effect다. V Cache도 같은 방식이다.

```
K Write Input
current K
write slot/index
layer

K Write Output
K cache update

V Write Input
current V
write slot/index
layer

V Write Output
V cache update
```

Attention Read 단계에서는 Cache Tensor 전체를 무조건 사용하는 것이 아니라 현재 Sequence에서 유효한 범위를 View 또는 Index 형태로 참조한다. Attention Mask는 미래 Position과 다른 Sequence의 Slot을 읽지 않도록 제한한다.

Graph Dump에서 KV Cache 관련 Node는 `CPY`, `SET_ROWS`, `VIEW`, `PERMUTE`, `CONT` 같은 Operation 조합으로 나타날 수 있다. 어떤 Operation이 사용되는지는 Cache Layout과 Backend 구현에 따라 달라진다.

성능 분석에서는 KV Cache의 Data Type과 Buffer 위치를 반드시 기록해야 한다. K/V가 CPU Buffer에 있고 Attention이 HTP에서 실행되면 Backend 경계 Copy가 발생할 수 있다. 반대로 Cache가 HTP Buffer에 있어도 View, Transpose, Contiguous 변환이 비효율적이면 Decode 시간이 증가할 수 있다.

---

# 19. FFN Graph Build

Attention 이후에는 FFN이 실행된다. Llama 계열의 Gated FFN을 단순화하면 Norm, Gate Projection, Up Projection, Activation Function, Elementwise Multiply, Down Projection, Residual Add의 순서다.

```
Attention Output
→ RMS Norm
→ Gate MUL_MAT
→ Up MUL_MAT
→ SiLU 또는 Architecture별 Activation
→ Gate × Up
→ Down MUL_MAT
→ Residual Add
```

FFN Builder의 Input은 Attention 결과, FFN Norm Weight, Gate/Up/Down Weight와 Activation 종류다. Output은 다음 Layer로 전달할 Hidden State다.

Gate와 Up Projection은 동일한 Normed Activation을 `src1`으로 공유한다. `src0`에는 서로 다른 Weight Tensor가 들어간다. Activation Function을 통과한 Gate 결과와 Up 결과가 Elementwise Multiply로 결합되고, Down Projection이 Embedding Dimension으로 되돌린다.

FFN은 모델 Weight의 큰 비중을 차지하므로 Quant Type과 Backend Kernel이 Decode 성능에 큰 영향을 줄 수 있다. Layer별 Benchmark에서는 `ffn_gate`, `ffn_up`, `ffn_down`을 각각 분리해 Kernel 시간과 Fallback 여부를 기록하는 편이 좋다.

---

# 20. Final Norm과 Output Projection

마지막 Transformer Layer가 끝나면 Final Norm이 적용되고 Output Projection이 Hidden State를 Vocabulary 크기의 Logits로 바꾼다.

Tied Word Embedding 모델에서는 Input Embedding과 Output Projection이 같은 원본 Weight를 공유할 수 있다. 이 경우 Input에서는 `token_embd.weight`가 `GET_ROWS`의 `src0`로 사용되고, Output에서는 같은 Weight가 `MUL_MAT`의 `src0`로 사용된다.

```
Input Embedding
token ID
→ GET_ROWS(token_embd.weight)
→ Hidden Vector

Output Projection
Final Hidden
→ MUL_MAT(token_embd.weight)
→ Vocabulary Logits
```

같은 Weight를 공유하더라도 두 Operation의 Backend 배치는 같을 필요가 없다. Input `GET_ROWS`는 CPU에서 실행되고 Output `MUL_MAT`는 HTP에서 실행될 수 있다. Runtime이 동일한 원본 Weight를 서로 다른 Buffer Layout으로 보관하는지도 별도로 확인해야 한다.

Untied 모델에서는 `output.weight` 또는 Architecture별 LM Head Tensor가 별도로 존재한다. 확인 방법은 HF Config만 보는 것이 아니라 GGUF Tensor List와 Graph Dump에서 마지막 `MUL_MAT`의 `src0`를 직접 확인하는 것이다.

Output Projection은 Vocabulary가 클 경우 Decode 시간에서 상당한 비중을 차지할 수 있다. 따라서 “Transformer Block은 HTP에서 실행됐다”는 로그만으로 전체 Decode Path를 설명하면 안 되고, 마지막 Projection의 Backend와 Kernel 시간도 포함해야 한다.

---

# 21. `ggml_build_forward_expand`: 실행 Graph 확정

Graph Builder가 여러 Tensor Node를 만들었더라도 모두 실행 대상이 되는 것은 아니다. `ggml_build_forward_expand`는 최종 Output Tensor에서 시작해 `src[]` Dependency를 거꾸로 따라가 실제로 필요한 Node를 `ggml_cgraph`에 등록한다.

```cpp
ggml_build_forward_expand(graph, final_output);
```

Input은 Graph 객체와 최종 Tensor다. Output은 Dependency 순서가 정리된 Compute Graph다. 최종 Output과 연결되지 않은 Node는 Graph에 포함되지 않을 수 있다.

```
Input
final output tensor
partially built graph

Output
topologically ordered compute graph
node list
leaf list
```

Dependency 탐색은 Output에서 Input 방향으로 이루어지지만, Graph의 실행 순서는 먼저 계산해야 하는 Node가 앞에 오도록 정리된다. 이 단계에서도 실제 계산은 발생하지 않는다. Backend 배정도 아직 확정되지 않는다.

이 함수가 끝난 직후 Graph Dump를 남기면 Scheduler의 영향을 받기 전 “모델이 어떤 계산을 요청했는가”를 확인할 수 있다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 22. Graph Dump로 함수의 Input/Output 확인하기

함수 설명을 실제 Source와 연결하려면 Graph Node를 텍스트로 Dump하는 것이 가장 효과적이다. 각 Node에 대해 Index, Name, Operation, Type, Shape, `src0`, `src1`, `src2`, View Source를 기록한다.

```
[GRAPH]
#000 name=inp_tokens
     op=NONE
     type=I32
     shape=[1]

#001 name=inp_embd
     op=GET_ROWS
     src0=token_embd.weight
     src1=inp_tokens

#010 name=blk.0.attn_q
     op=MUL_MAT
     src0=blk.0.attn_q.weight
     src1=blk.0.attn_norm
```

이 로그는 함수의 의미를 코드 한 줄보다 명확하게 보여 준다. `blk.0.attn_q`의 Input이 실제로 어떤 Weight와 Activation인지, Weight Type이 무엇인지, Output Shape가 무엇인지 확인할 수 있기 때문이다.

Graph Dump에서 `MUL_MAT`이 보인다는 사실은 그 Operation이 실행 Graph에 포함됐다는 뜻이다. HTP에서 실행됐다는 뜻은 아니다. 실제 Backend는 Scheduler Dump에서 확인해야 한다.

팀에서 공유할 Graph Dump는 Prefill과 Decode를 분리해야 한다. Prefill은 Token 차원이 크고 Decode는 1이므로 Node 이름이 같더라도 Shape와 Kernel 선택이 달라질 수 있다.

---

# 23. Backend Scheduler와 Graph Split

Compute Graph가 완성되면 Backend Scheduler가 각 Node를 어느 Backend에서 실행할지 결정한다. Scheduler의 Output은 단순한 Node별 Label이 아니라 Backend별 Graph Split과 Split 사이의 Input Copy 계획이다.

```
전체 Graph
→ Node별 Backend 후보 판정
→ 연속된 Backend Node를 Split으로 묶음
→ Split 입력 확인
→ Backend 경계 Copy 준비
→ Backend별 Graph Compute
```

예를 들어 Input Embedding은 CPU, Transformer Block은 HTP, Sampling은 Graph 밖의 CPU 코드라면 Compute Graph는 다음과 같이 나뉠 수 있다.

```
Split 0: CPU
GET_ROWS

Copy
inp_embd CPU → HTP0

Split 1: HTP0
Transformer Blocks
Final Norm
Output Projection
```

Scheduler의 Input은 Compute Graph, Backend 목록, Tensor Buffer 상태다. Output은 Node별 Backend, Split 목록, Copy 대상 Tensor다.

성능 분석에서는 HTP Node 비율만 보는 것보다 Split 수와 경계 위치가 더 중요할 수 있다. HTP Node가 많더라도 Graph가 CPU와 HTP 사이를 여러 번 왕복하면 Copy와 Synchronization 비용이 누적된다.

---

# 24. `supports_op`와 `supports_buft`

Backend가 Node를 실행할 수 있는지 판단할 때 대표적으로 Operation 지원과 Buffer Type 지원을 확인한다.

`supports_op`는 Backend가 해당 Operation, Tensor Type, Shape, Stride 조건을 처리할 수 있는지를 판단한다. `supports_buft`는 Backend가 연결된 Buffer Type을 사용할 수 있는지 판단한다.

```
supports_op Input
backend device
operation node
src/dst type과 shape

supports_op Output
true / false
필요 시 거부 이유

supports_buft Input
backend device
buffer type

supports_buft Output
true / false
```

`supports_op=true`는 해당 Node가 반드시 그 Backend에서 실행된다는 뜻이 아니다. Weight가 다른 Backend Buffer에 있거나 전체 Split 구성상 Copy 비용이 필요하면 최종 배정은 달라질 수 있다.

실무에서 가장 필요한 로그는 Boolean 값만이 아니라 판정 이유다.

```
[BACKEND-SUPPORT]
node=blk.0.attn_q
op=MUL_MAT
backend=HTP0
supports_op=true
supports_src0_buft=true
supports_src1_buft=true
decision=HTP0

[BACKEND-SUPPORT]
node=inp_embd
op=GET_ROWS
backend=HTP0
supports_op=true
supports_src0_buft=false
src0_buft=CPU_Mapped
decision=CPU
reason=input weight buffer
```

이 형태의 로그가 있어야 “Operation 미지원”과 “Buffer 배치 때문에 CPU에 남음”을 구분할 수 있다.

---

# 25. Backend 경계의 Tensor Copy

서로 다른 Backend가 연속된 Node를 실행하면 앞 Split의 Output을 다음 Backend가 읽을 수 있는 Buffer로 전달해야 한다. Scheduler는 이 경계에 Copy Tensor 또는 Copy Operation을 준비한다.

Copy의 Input은 Source Tensor와 Destination Backend Buffer다. Output은 동일한 값을 가진 Destination Tensor다. 수학적 값은 변하지 않지만 Memory 위치와 Buffer Type이 달라진다.

```
Input
source tensor
source backend/buffer
destination backend/buffer

Output
copied tensor
copy completion / synchronization
```

Decode에서는 Token 하나를 생성할 때 같은 경계가 반복된다. Copy Byte가 작더라도 매 Step마다 Queue Submit과 Synchronization이 발생하면 비용이 누적될 수 있다. 반대로 경계가 한 번뿐이고 Copy가 매우 작다면 전체 Decode 지연의 주원인일 가능성은 낮다.

따라서 Copy 분석에는 횟수, Byte, 실제 Copy 시간, Synchronization 시간을 함께 기록해야 한다.

```
[COPY]
phase=decode
token_index=0
tensor=inp_embd
from=CPU
to=HTP0
bytes=...
copy_us=...
sync_us=...
```

“CPU와 HTP 사이를 왔다 갔다 한다”는 표현은 Graph Split을 실제로 확인한 뒤 사용해야 한다. CPU→HTP 한 번만 존재하는지, Layer 중간에 여러 번 왕복하는지는 성능 원인 해석이 완전히 다르다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 26. Backend `graph_compute`와 실제 계산

Scheduler가 Graph Split을 만들면 각 Backend의 `graph_compute`가 자신에게 배정된 Node를 실제로 실행한다. 여기서부터 `MUL_MAT`, `RMS_NORM`, `GET_ROWS` 같은 Operation이 실제 숫자 계산으로 바뀐다.

CPU Backend는 Operation Type과 Tensor Type에 맞는 CPU Kernel을 호출한다. GPU Backend는 Command Buffer 또는 Shader/Kernel 호출로 변환한다. HTP Backend는 Host 쪽에서 Graph를 장치 명령으로 변환한 뒤 DSP Queue에 전달할 수 있다.

```
Input
backend handle
split graph
allocated tensor buffers

Output
destination tensor values
execution completion
status
```

Backend 전체 시간을 하나로만 기록하면 병목을 구분하기 어렵다. 가속기 Backend는 Host-side 준비, Graph Cache 조회, Kernel Parameter 생성, Queue Enqueue, Device 실행, Host Wait가 분리될 수 있다.

```
[GRAPH-COMPUTE]
backend=HTP0
prepare_us=...
enqueue_us=...
device_us=...
wait_us=...
total_us=...
```

Decode에서 Device 계산은 짧지만 Enqueue와 Wait가 상대적으로 크다면 NPU 연산 성능이 나쁜 것이 아니라 작은 Graph를 매 Token 제출하는 Runtime Overhead가 문제일 수 있다.

---

# 27. Qualcomm HTP Backend 사례

Qualcomm Hexagon HTP Backend는 llama.cpp의 공통 ggml Graph를 Snapdragon NPU 계열에서 실행하기 위한 Backend 구현이다. 상위 Graph Builder는 HTP 전용 Graph를 직접 만들지 않는다. 먼저 `GET_ROWS`, `MUL_MAT`, `RMS_NORM`, `ROPE` 같은 공통 ggml Operation을 만들고, Scheduler가 HTP에 배정한 Node를 HTP Backend가 장치 전용 실행 형태로 변환한다.

HTP `graph_compute`의 Host-side 단계는 일반적으로 Graph 검증 또는 Cache 조회, Compute Node 추출, Operation Remap, Fusion 가능성 판단, Kernel Parameter 계산, VTCM 요구량 확인, DSP Queue Submit, Completion 대기의 순서로 이해할 수 있다.

```
ggml HTP Split
→ HTP graph_compute
→ op remap
→ fusion / parameter build
→ VTCM 및 alignment 검사
→ dspqueue submit
→ HMX 또는 HVX Kernel
→ completion
```

정확한 함수명, 내부 Opcode, Tile 규격은 사용하는 Hexagon Backend Fork에서 직접 확인해야 한다. 공개된 상위 개념과 실제 사내 또는 Vendor Fork의 구현을 섞어 단정하면 안 된다.

팀 문서에는 Source에서 확인된 함수명과 개념상 예상되는 단계를 구분해 기록해야 한다. 확인되지 않은 함수는 가칭으로 쓰지 말고 “HTP Graph Compute 내부 Repack 함수 확인 필요”처럼 남기는 편이 안전하다.

---

# 28. HTP0-REPACK과 Q4 Weight

Q4_0은 Weight 32개를 하나의 Block으로 묶고, Block Scale과 4bit Code로 저장하는 Quant Type이다. FP16 Weight 32개는 64 Byte지만 Q4_0 Block은 일반적으로 Scale 2 Byte와 Code 16 Byte로 구성되어 18 Byte가 된다.

```cpp
// 개념 설명용
struct block_q4_0 {
    fp16 scale;
    uint8_t qs[16];
};
```

Q4 Weight가 작아지면 Decode에서 Weight를 읽는 Memory Traffic을 줄일 수 있다. 그러나 GGUF의 Q4 Byte Layout이 HTP Matmul Kernel이 원하는 Tile 순서와 같다는 보장은 없다. HTP0-REPACK은 이 Weight를 장치 Kernel이 읽기 쉬운 Layout으로 바꾸는 Buffer다.

Repack의 Input은 Q4_0 Weight Block, Tensor Shape, Kernel Tile 규격, Alignment 조건이다. Output은 수치적으로 같은 Weight를 표현하는 HTP 전용 Byte Layout이다.

```
Input
GGUF Q4_0 blocks
tensor dimensions
tile/alignment rule

Output
HTP0-REPACK weight buffer
```

Repack은 Model Load 시 한 번 수행되거나 Buffer 초기화 단계에서 수행될 수 있다. 매 Decode Step마다 Weight 전체를 Repack한다면 성능상 문제가 되므로 실제 호출 시점을 확인해야 한다.

Q4 Matmul이 빠르려면 Weight를 전체 FP16 Buffer로 복원한 뒤 Matmul하는 방식이 아니라, Kernel 내부에서 Nibble Unpack과 Scale 적용을 Dot Product에 결합하는 경로를 사용해야 한다. 실제 HTP Backend가 어떤 방식으로 Activation을 처리하고 어떤 Kernel을 선택하는지는 Kernel Path 로그로 검증해야 한다.

---

# 29. HMX, HVX, Fallback 경로

Hexagon에서 Matrix 연산은 HMX 계열 경로, HVX Vector 경로, 덜 최적화된 Flat 경로 또는 CPU Fallback으로 나뉠 수 있다. 어떤 경로가 선택되는지는 Operation, Weight Type, Activation Type, Shape, Tile Alignment, VTCM 용량에 영향을 받는다.

HMX는 Matrix 연산에 특화된 경로로 이해할 수 있고, HVX는 Vector 연산을 수행하는 경로다. HVX Tiled는 미리 정렬된 Tile Layout을 활용하는 반면 HVX Flat은 Shape 또는 Layout 조건이 최적 경로에 맞지 않을 때 사용될 수 있다. 정확한 명칭과 선택 조건은 Backend Source와 Runtime Log로 확인해야 한다.

Kernel 선택의 Input은 Node Operation, Tensor Type, Shape, Buffer Layout, Scratch Memory 조건이다. Output은 선택된 Kernel과 실행 결과다.

```
Input
op=MUL_MAT
weight_type=Q4_0
activation_type=...
M/N/K shape
buffer=HTP0-REPACK
VTCM availability

Output
path=HMX / HVX_TILED / HVX_FLAT / fallback
kernel result
```

Decode의 `n_tokens=1`은 Prefill보다 Matrix의 한 차원이 작다. 이 Shape가 HMX Tile을 충분히 채우지 못하거나 Kernel Launch Overhead를 상쇄하지 못하면 HTP가 CPU보다 느릴 수 있다. 이 경우 “NPU가 Matmul을 지원하지 않는다”가 아니라 “지원은 하지만 해당 Shape에서 효율이 낮다”가 정확한 해석이다.

필수 Kernel 로그는 다음 형태가 적합하다.

```
[KERNEL]
phase=decode
layer=0
node=blk.0.attn_q
op=MUL_MAT
weight_type=Q4_0
activation_type=...
weight_buffer=HTP0-REPACK
path=HMX
tile_m=...
tile_n=...
tile_k=...
vtcm_bytes=...
kernel_us=...
fallback_reason=none
```

---

# 30. Logits, Sampling, Detokenize

Output Projection이 만든 Logits는 Vocabulary의 각 Token에 대한 점수다. Application은 Context에서 필요한 Token 위치의 Logits를 읽고 Sampler Chain에 전달한다.

Sampling의 Input은 Logits와 Temperature, Top-k, Top-p, Repeat Penalty 같은 Sampler 상태다. Output은 다음 Token ID다. Greedy Sampling을 사용하면 가장 큰 Logit의 Token을 선택하고, 확률 Sampling을 사용하면 Logits를 확률 분포로 변환해 선택한다.

```
Input
logits
sampler parameters
sampler history

Output
next token id
updated sampler state
```

선택된 Token ID는 `llama_token_to_piece` 계열의 Detokenize 함수를 거쳐 문자열 조각이 된다. 이 단계는 일반적으로 CPU에서 실행되며 ggml Compute Graph 밖에 있다.

Sampling 시간이 전체 Decode 시간에서 차지하는 비율은 모델과 Vocabulary 크기, Sampler 설정에 따라 달라질 수 있다. Backend Benchmark에서는 순수 `llama_decode` 시간과 Sampling·Detokenize를 포함한 End-to-End Token 시간을 구분해서 기록하는 것이 좋다.

---

# 31. Prefill과 Decode가 같은 모델인데도 다르게 동작하는 이유

Prefill과 Decode는 같은 Weight와 같은 Layer를 사용하지만 Tensor Shape와 Runtime 특성이 다르다.

Prefill은 Prompt Token 여러 개를 한 번에 처리해 KV Cache를 채운다. Matrix의 Token 차원이 크기 때문에 Parallelism이 높고 가속기 Tile을 채우기 쉽다. Backend Launch와 Synchronization 비용도 많은 연산에 나누어 부담된다.

Decode는 보통 Token 하나를 입력으로 받아 다음 Token 하나를 만든다. 각 Step마다 동일한 큰 Weight를 읽지만 Activation의 Token 차원은 1이다. 따라서 Weight Memory Bandwidth, Kernel Launch, Backend Synchronization, KV Cache Read가 상대적으로 중요해진다.

| 구분 | Prefill | Decode |
| --- | --- | --- |
| 입력 Token 수 | 여러 개 | 보통 1개 |
| Matmul 병렬성 | 큼 | 작음 |
| KV 동작 | 여러 Slot 작성 | 1 Slot Append |
| Backend Overhead | 여러 Token에 분산 | 매 Token 반복 |
| 주요 지표 | Prompt tok/s, total ms | ms/token, tok/s |

현재 Snapdragon HTP 실험에서 Prefill은 HTP가 CPU보다 빠르지만 Decode는 CPU가 더 빠른 경향이 관찰되었다. 이 결과는 Input Embedding Copy 하나만으로 설명해서는 안 된다. Small-shape Kernel 효율, HTP Prepare/Enqueue/Wait, Q4 Activation 변환, KV Cache Operation, Output Projection을 분리해 측정해야 한다.

↩︎ 전체 파이프라인으로 돌아가기

---

# 32. Decode 1 Step 전체 함수·Tensor 흐름

아래 흐름은 Decoder-only Transformer의 Decode 한 Step을 Graph Node 기준으로 단순화한 것이다. 실제 Node 이름과 세분화 정도는 Architecture와 llama.cpp 버전에 따라 달라질 수 있다.

```
inp_tokens
op=NONE
Token ID 1개
        ↓
ggml_get_rows
src0=token_embd.weight
src1=inp_tokens
dst=inp_embd
        ↓
RMS_NORM
src0=inp_embd
src1=attn_norm.weight
dst=normed_hidden
        ↓
ggml_mul_mat Q
src0=attn_q.weight
src1=normed_hidden
dst=q_cur

ggml_mul_mat K
src0=attn_k.weight
src1=normed_hidden
dst=k_cur

ggml_mul_mat V
src0=attn_v.weight
src1=normed_hidden
dst=v_cur
        ↓
ROPE
Q/K + Position
        ↓
KV Cache Write
K/V + Cache Index
        ↓
Attention Score
Q + Cached K
        ↓
Mask / Softmax
        ↓
Value Aggregate
Probability + Cached V
        ↓
Output Projection
src0=attn_output.weight
src1=attention_value
        ↓
Residual Add
        ↓
FFN Norm
        ↓
Gate / Up Projection
        ↓
Activation + Elementwise Multiply
        ↓
Down Projection
        ↓
Residual Add
        ↓
다음 Layer 반복
        ↓
Final Norm
        ↓
LM Head MUL_MAT
        ↓
Logits
        ↓
Sampling
        ↓
Next Token ID
```

각 Node를 분석할 때는 함수명, `src0`, `src1`, Output Tensor, Type, Shape, Buffer, Backend, Kernel Path, 시간을 하나의 Row로 기록해야 한다. 이 정보가 모이면 Layer별 CPU/HTP 실행 비율과 Quant Type별 시간을 Excel 또는 HTML에서 비교할 수 있다.

---

# 33. 성능 원인 분석 시 지켜야 할 증명 순서

성능 문제가 발생했을 때 가장 먼저 확인할 것은 모델의 Quant 이름이 아니라 실제 실행 경로다. 올바른 순서는 Graph에 Operation이 존재하는지, Scheduler가 어느 Backend에 배정했는지, Backend 경계 Copy가 어디에 생겼는지, 실제 Kernel이 무엇인지, 각 구간 시간이 얼마인지를 차례로 확인하는 것이다.

먼저 Graph Dump로 `GET_ROWS`, `MUL_MAT`, `RMS_NORM`, KV Cache Node가 기대한 Dependency로 만들어졌는지 확인한다. 그다음 Scheduler Dump로 각 Node의 Backend와 Split 경계를 확인한다. 이후 Copy 로그로 CPU와 가속기 사이에 어떤 Tensor가 몇 Byte 이동하는지 확인한다. 마지막으로 Backend Timing과 Kernel Path로 실제 장치 실행 시간을 확인한다.

이 순서를 건너뛰면 다음과 같은 잘못된 결론이 생긴다.

“Q4 모델이므로 Q4 HMX Kernel이 실행됐다.”

실제로는 Weight가 CPU Buffer에 남거나 Q4 Shape가 미지원되어 CPU Fallback일 수 있다.

“GET_ROWS가 CPU이므로 모든 Layer마다 CPU와 HTP를 왕복한다.”

실제로는 Graph 앞에서 CPU→HTP Copy 한 번만 존재할 수 있다.

“HTP Node가 90%이므로 성능이 좋아야 한다.”

실제로는 매 Token마다 Prepare와 Wait가 커서 CPU보다 느릴 수 있다.

“Prefill이 빠르므로 Decode도 같은 Kernel에서 빠르다.”

실제로는 Token 차원이 달라 다른 Kernel Path가 선택될 수 있다.

팀 보고에서는 확인된 사실, 로그 기반 추론, 아직 검증되지 않은 가설을 명확히 구분해야 한다.

---

# 34. 권장 Instrumentation 산출물

팀 단위로 결과를 비교하려면 실행할 때마다 동일한 형식의 산출물을 남겨야 한다. 다음 구조는 모델, Quant Type, Backend가 달라도 재사용할 수 있다.

```
experiments/
└── <model>_<quant>_<backend>_<timestamp>/
    ├── run_config.json
    ├── tensor_list.csv
    ├── tensor_placement.csv
    ├── graph_prefill_nodes.csv
    ├── graph_decode_nodes.csv
    ├── schedule_prefill.csv
    ├── schedule_decode.csv
    ├── split_timing.csv
    ├── copy_timing.csv
    ├── kernel_path.csv
    ├── benchmark.json
    ├── raw_log.txt
    └── report.md
```

`tensor_placement.csv`에는 Tensor 이름, Layer 종류, Type, Shape, Device, Buffer, Repack 여부를 기록한다. `graph_decode_nodes.csv`에는 Node Index, Layer, Name, Operation, `src0`, `src1`, Type과 Shape를 기록한다. `schedule_decode.csv`에는 Backend, Buffer Type, Support 판정 이유, Split ID를 기록한다.

`kernel_path.csv`는 실제 가속기 분석의 핵심이다.

```
phase
token_index
layer
node
op
weight_type
activation_type
path
tile_m
tile_n
tile_k
vtcm_bytes
prepare_us
kernel_us
wait_us
fallback_reason
```

이 산출물은 최종 HTML 문서의 데이터 Source로도 사용할 수 있다. HTML에서는 전체 Pipeline의 각 항목을 클릭하면 해당 함수 설명과 실제 측정 결과로 이동하게 하고, 상세 설명 아래에는 전체 Pipeline으로 돌아가는 버튼을 배치한다.

---

# 35. 코드 탐색 순서

Source를 처음부터 무작위로 검색하면 모델 로드 코드, Graph Builder, Backend Kernel이 섞여 이해하기 어렵다. 다음 순서로 Call Chain을 추적하는 편이 효율적이다.

먼저 Application에서 `llama_model_load_from_file`과 `llama_init_from_model`이 호출되는 위치를 찾는다. 여기서 Model Parameter와 Context Parameter가 어떻게 설정되는지 확인한다.

그다음 `llama_model_loader`, GGUF 초기화, Metadata 로드, Vocabulary 로드, `create_tensor`, Tensor Data Load를 따라간다. 이 구간에서 Weight 이름과 Type, Buffer 배치가 결정된다.

이후 `llama_decode`에서 Batch와 KV Cache Slot이 준비되는 경로를 따라간다. Architecture별 Graph Build 진입점을 찾고 `ggml_get_rows`, `ggml_mul_mat`, Norm, RoPE, KV Cache 관련 Node가 만들어지는 위치를 기록한다.

Graph가 완성되는 `ggml_build_forward_expand` 이후에는 Backend Scheduler로 이동한다. `supports_op`, `supports_buft`, Split 생성, Copy Tensor 생성 코드를 추적한다.

마지막으로 CPU 또는 HTP Backend의 `graph_compute`에서 Operation이 실제 Kernel로 연결되는 경로를 찾는다. HTP에서는 Repack, Opcode Remap, Fusion, DSP Queue, HMX/HVX 선택 조건을 확인한다.

검색 키워드는 다음과 같이 구간별로 사용할 수 있다.

```
Model Load
llama_model_load_from_file
llama_model_loader
gguf_init_from_file
load_hparams
load_vocab
create_tensor
load_tensors
load_all_data

Context
llama_init_from_model
llama_context
kv_cache
backend_sched

Inference
llama_decode
llama_batch
llama_ubatch
build_graph
ggml_get_rows
ggml_mul_mat
ggml_build_forward_expand

Scheduler
supports_op
supports_buft
sched
split
copy
graph_compute

HTP
hexagon
HTP0
REPACK
dspqueue
HMX
HVX
VTCM
tile
fallback
```

함수별 조사 결과는 함수명과 Signature만 기록하지 말고 Caller, Callee, 일반 인자, Tensor Input, Tensor Output 또는 Side Effect, Buffer 조건, Fallback 조건까지 함께 기록해야 한다.

---

# 36. 결론

llama.cpp는 모델 파일을 읽어 고정 Graph를 실행하는 단순 Runtime이 아니다. 모델 로드 시점에는 GGUF Metadata와 Weight Tensor를 Runtime 객체와 Backend Buffer에 연결하고, 추론 시점에는 현재 Batch와 KV Cache 상태에 맞는 Compute Graph를 동적으로 만든다. Scheduler는 그 Graph를 Backend별로 나누고, 실제 계산은 각 Backend의 `graph_compute`와 Kernel에서 수행된다.

이 구조를 이해하는 데 가장 중요한 구분은 네 가지다.

첫째, GGUF Tensor Type과 실제 Kernel 실행은 다른 층위다. Q4 Weight가 파일에 존재하는 것과 Q4 HTP Kernel이 실행되는 것은 별도로 증명해야 한다.

둘째, Graph Build와 Compute는 다르다. `ggml_mul_mat`는 Node를 만들고 Backend Kernel이 실제 값을 계산한다.

셋째, Operation 지원과 최종 Backend 배치는 다르다. `supports_op`가 참이어도 Buffer Type과 Split 조건 때문에 CPU에 남을 수 있다.

넷째, Backend에 배정된 Node 비율과 End-to-End 성능은 다르다. Copy, Queue Submit, Synchronization, KV Cache, Small-shape Kernel 효율을 함께 봐야 한다.

이 문서의 다음 단계는 현재 사용하는 llama.cpp와 Hexagon Backend Branch의 Commit을 고정하고, 각 절의 대표 함수에 실제 파일 경로와 Line Range를 연결하는 것이다. 이후 Prefill과 Decode의 Graph·Scheduler·Kernel 로그를 같은 Schema로 수집하면 특정 모델에 한정되지 않는 팀 공용 llama.cpp 분석 문서와 Benchmark 체계를 완성할 수 있다.

---

# 부록 A. HTML 전환 규칙

최종 HTML에서는 2장의 전체 Pipeline을 Navigation Hub로 사용한다. 각 Pipeline Box는 해당 상세 절로 이동하고, 각 절 마지막의 “전체 파이프라인으로 돌아가기” 문구는 상단 Pipeline으로 이동하는 Button으로 바꾼다.

권장 Anchor는 다음과 같다.

```
pipeline-root
pipeline-model-load
pipeline-tensor-load
pipeline-context
pipeline-tokenize
pipeline-decode
pipeline-graph-build
pipeline-scheduler
pipeline-backend-compute
pipeline-sampling
```

Decode 1 Step은 Node별 Anchor를 별도로 둔다.

```
decode-input
decode-embedding
decode-qkv
decode-kv-cache
decode-attention
decode-ffn
decode-output
decode-sampling
```

HTML은 Prefill과 Decode, CPU와 HTP를 전환할 수 있는 Toggle을 제공하는 것이 좋다. Node 상세 Card에는 Function, `src0`, `src1`, Output, Type, Shape, Buffer, Backend, Kernel Path, Timing을 표시한다.

---

# 부록 B. 문서 작성 기준

이 문서는 특정 모델의 변환 절차를 설명하는 문서가 아니다. Llama, Gemma, EXAONE 등 llama.cpp가 지원하는 Decoder 계열 모델에 공통으로 적용되는 Runtime 파이프라인을 설명한다.

Qualcomm HTP 내용은 전체 Pipeline을 이해하기 위한 Backend 사례다. HTP 전용 함수명, Tile Layout, HMX/HVX 선택 조건은 현재 사용 중인 Fork Source와 Runtime Log에서 확인된 내용으로 갱신해야 한다.

문서에서 “확인됨”이라고 표현할 수 있는 내용은 Source Code, Graph Dump, Scheduler Log, Kernel Log 또는 Benchmark Raw Data로 근거가 남아 있는 항목에 한정한다. 구현마다 달라질 수 있는 내용은 일반 구조와 실제 프로젝트 확인 결과를 분리해 작성한다.