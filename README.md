# NeuroBranch Intelligent Branching Demo

A high-performance demonstration project showcasing the **Intelligent Branching** programming paradigm in pure Go using [`github.com/gluedays-cyber/neurobranch`](https://github.com/gluedays-cyber/neurobranch).

It proves how embedded domain-specific neural networks replace fragile static branching (`if` / regex) and slow, expensive cloud LLMs with **deterministic in-memory neural routing in ~30 μs with 0 B/op heap allocation**.

---

## 1. Architectural Comparison Matrix

| Metric / Capability | Retro Branching (`if` / Regex) | Cloud LLMs (OpenAI / Claude) | **NeuroBranch v2.0 (Intelligent Branching)** |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | < 1 μs | 400 ms – 2,500 ms (Network bound) | **~30 μs (In-Memory Pure Math)** |
| **Permutations & Typos** | ❌ Fails on unseen phrasing | ✅ Supported via attention | ✅ **Absorbed into continuous latent vectors** |
| **Heap Allocations (GC)** | 0 B/op | High HTTP payload overhead | **0 B/op (Zero-Allocation stack evaluation)** |
| **Operational Cost** | $0.00 / binary | Per-token API billing ($$$) | **$0.00 / 100% Free & Self-Contained** |
| **External Dependencies** | Standard library | External API, Network, API Key | **Zero CGO, Zero Downloads, Pure Go (`CGO_ENABLED=0`)** |
| **Noise & OOD Isolation** | Exponential regex explosion | Hallucinated misclassification | **3-Tier Multi-Metric Guard (Entropy & Energy)** |
| **Dynamic Rule Updates** | Recompile & Redeploy service | Prompt re-engineering | **Lock-Free Atomic Pointer Hot-Swap (`0 ns`)** |

---

## 2. Intelligent Branching Primitives

### 2.1 Native Go `switch-case` Branching (`ai.Select`)
Transforms natural language, typos, and phrasing variations into switch labels. Unlearned vocabulary (OOV) or Out-of-Domain queries return `""`, cleanly falling back to native `default:` branch.

```go
switch branch := ai.Select(query); branch {
case "Refund":
    processRefundPipeline()
case "Delivery":
    trackCourierDelivery()
case "Account":
    resetAccountPassword()
default:
    // Safe isolation for OOD queries or unlearned noise
    transferToHumanAgent()
}
```

### 2.2 Native Go Guard Clauses (`ai.If`, `ai.Is`)
Replaces brittle regex or substring matching with confidence-calibrated neural guard assertions.

```go
if ai.If(query, "Refund") {
    // Verified Refund intent
}

if ai.Is(query, "Refund", 0.85) {
    // Confirmed Refund intent with >= 85% confidence
}
```

### 2.3 Native Go Comma-ok Idiom (`ai.Match`)
Follows Go's idiomatic `comma-ok` pattern to separate decisive matches from borderline ambiguous queries.

```go
if intent, confident := ai.Match(query); confident {
    // Decisive execution branch
} else if intent != "" {
    // Borderline ambiguous state: prompt user for clarification
} else {
    // Out-of-domain or degenerated input safely blocked
}
```

### 2.4 Declarative Fluent DSL & Interactive Confirmation (`ai.Switch`)
Provides fluent method chaining for automated execution (`Auto`), human-in-the-loop confirmation (`Confirm`), and fallback handling (`Default`).

```go
err := ai.Switch(query).
    Case("Account").
        Auto(func(ctx context.Context) error {
            return unlockAccount()
        }).
        Confirm("Would you like to send a 2FA verification SMS?", func(ctx context.Context, prompt string) error {
            return promptUser(prompt)
        }).
    Case("Delivery").
        Auto(func(ctx context.Context) error {
            return fetchDeliveryStatus()
        }).
    Default(func(ctx context.Context) error {
        return defaultFallbackHandler()
    }).
    Evaluate(ctx)
```

### 2.5 Multi-Metric Neural Inspection (`ai.Inspect`)
Decomposes queries into BPE subwords, single-character ratio (`SingleCharRatio`), Shannon entropy, and LogSumExp free energy for full observability and drift monitoring.

### 2.6 Zero-Downtime Atomic Hot-Swap (`ai.AppendDataMap`)
Appends new domain intents and retrains weights in-memory, replacing pointers atomically without dropping active requests or restarting the service.

---

## 3. Project Structure

```
neurobranch-demo/
├── data/
│   └── train.csv       # Labeled CSV dataset for domain neural network compilation
├── weights/
│   └── model.bin       # Compiled Little-Endian self-calibrating binary model
├── go.mod              # Go module definition and local neurobranch replace directive
├── main.go             # 7-step comprehensive intelligent branching benchmark suite
├── main_test.go        # Unit & integration test suite verifying all branching mechanics
└── README.md           # Technical documentation and architecture specification
```

---

## 4. Build & Execution

### 4.1 Run All Unit Tests
```bash
go test -v ./...
```

### 4.2 Run the Intelligent Branching Demo
```bash
go run main.go
```

### 4.3 Compile Single Static Binary
```bash
go build -v -o neurobranch-demo.exe .
```
