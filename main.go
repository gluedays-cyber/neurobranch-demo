package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gluedays-cyber/neurobranch"
)

func main() {
	printBanner()

	// =========================================================================
	// 1. Train & Load Domain Neural Network from CSV Dataset
	// =========================================================================
	fmt.Println("[Step 1] Compile Neural AI Model Directly from CSV Dataset")
	dataPath := filepath.Join("data", "train.csv")
	modelDir := "weights"
	modelPath := filepath.Join(modelDir, "model.bin")

	// Load training samples from data/train.csv
	samples, err := neurobranch.LoadCSVDataset(dataPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to load dataset from %s: %v", dataPath, err))
	}

	cfg := neurobranch.DefaultTrainConfig()
	cfg.Seed = 42
	cfg.Epochs = 100

	startTrain := time.Now()
	ai, err := neurobranch.TrainInMemory(samples, cfg)
	if err != nil {
		panic(fmt.Sprintf("In-memory training failed: %v", err))
	}
	trainDuration := time.Since(startTrain)

	// Persist compiled Little-Endian binary model for offline deployment
	_ = os.MkdirAll(modelDir, 0755)
	if err := neurobranch.SaveBinaryModel(modelPath, ai.Model()); err != nil {
		panic(fmt.Sprintf("Failed to serialize model: %v", err))
	}

	fmt.Printf(">> Domain AI compiled and loaded into RAM (Elapsed: %v)\n", trainDuration)
	fmt.Printf("   Architecture: In-Memory BPE Tokenizer + Positional Embeddings (D=64) + GELU Pooling\n")
	fmt.Printf("   Model Artifact: Serialized to %s (Format v3 Self-Calibrating)\n\n", modelPath)

	// =========================================================================
	// 2. Native Go switch-case Intelligent Branching (ai.Select)
	// =========================================================================
	fmt.Println("[Step 2] Native Go switch-case Intelligent Branching (ai.Select)")
	fmt.Println("Concept: Maps natural language, typos, slang, and permutations into continuous vector space.")
	fmt.Println("         Unlearned OOV or Out-of-Domain inputs return \"\", cleanly falling back to default: branch.")

	testQueries := []string{
		"reverse transaction charge refund",           // Refund intent
		"where is my package delivery tracking",       // Delivery intent
		"locked out of user account login",            // Account intent
		"billing credit card monthly receipt",         // Billing intent
		"zzzzzzzz xxxxxxxx yyyyyyyy",                  // Unlearned noise (Layer 1 Fail-Safe Cutoff)
		"quantum physics entangled photon spin",       // Out-of-Domain (Layer 2 Multi-Metric Guard)
	}

	for _, q := range testQueries {
		fmt.Printf("\n>> Input Query: %q\n", q)
		switch branch := ai.Select(q); branch {
		case "Refund":
			fmt.Printf("   [BRANCH: CASE Refund] Invoking refund & chargeback processing service\n")
		case "Delivery":
			fmt.Printf("   [BRANCH: CASE Delivery] Querying live shipment courier tracking API\n")
		case "Account":
			fmt.Printf("   [BRANCH: CASE Account] Launching user credential recovery workflow\n")
		case "Billing":
			fmt.Printf("   [BRANCH: CASE Billing] Fetching monthly invoices and tax receipts\n")
		default:
			fmt.Printf("   [BRANCH: DEFAULT Fallback] Isolated noise/OOD query -> Route to fallback handler\n")
		}
	}
	fmt.Println()

	// =========================================================================
	// 3. Native Go Guard Clauses (ai.If, ai.Is)
	// =========================================================================
	fmt.Println("[Step 3] Native Go Guard Clauses (ai.If, ai.Is)")
	fmt.Println("Concept: Execute deterministic neural guard assertions without brittle regex patterns.")

	guardQuery := "please process full refund immediately"
	fmt.Printf(">> Guard Target: %q\n", guardQuery)

	if ai.If(guardQuery, "Refund") {
		fmt.Println("   [PASSED] ai.If(query, \"Refund\") == true -> Validated refund request")
	}

	// Strict confidence threshold assertion (>= 80%)
	if ai.Is(guardQuery, "Refund", 0.80) {
		fmt.Println("   [PASSED] ai.Is(query, \"Refund\", 0.80) == true -> Confirmed high confidence (>= 80%)")
	} else {
		fmt.Println("   [REJECTED] ai.Is(query, \"Refund\", 0.80) == false -> Confidence below threshold")
	}
	fmt.Println()

	// =========================================================================
	// 4. Native Go Comma-ok Idiom (ai.Match)
	// =========================================================================
	fmt.Println("[Step 4] Native Go Comma-ok Idiom (ai.Match)")
	fmt.Println("Concept: (intent, confident) return signature distinguishes definite vs ambiguous states.")

	evalInputs := []string{
		"want my money back refund",
		"zzzzzzzz xxxxxxxx yyyyyyyy",
	}

	for _, input := range evalInputs {
		fmt.Printf(">> Query: %q\n", input)
		if intent, confident := ai.Match(input); confident {
			fmt.Printf("   [COMMA-OK] Decisive intent identified: %s (confident: true)\n", intent)
		} else if intent != "" {
			fmt.Printf("   [AMBIGUOUS] Borderline intent detected: %s (confident: false, human review recommended)\n", intent)
		} else {
			fmt.Println("   [OOD/REJECT] Out-of-domain or unlearned noise safely rejected")
		}
	}
	fmt.Println()

	// =========================================================================
	// 5. Declarative Fluent DSL & HITL (Human-In-The-Loop) Confirmation
	// =========================================================================
	fmt.Println("[Step 5] Declarative Fluent DSL & Interactive Confirmation (ai.Switch)")
	fmt.Println("Concept: Fluent chaining for automatic dispatch (Auto), confirmation prompts (Confirm), and Default.")

	ctx := context.Background()
	dslQueries := []struct {
		Text string
		Desc string
	}{
		{Text: "locked out of user account login", Desc: "High-Confidence Account Query"},
		{Text: "where is my package delivery tracking", Desc: "High-Confidence Delivery Query"},
		{Text: "zzzzzzzz xxxxxxxx yyyyyyyy", Desc: "Degenerated Noise Query"},
	}

	for _, item := range dslQueries {
		fmt.Printf(">> [%s] Query: %q\n", item.Desc, item.Text)
		err := ai.Switch(item.Text).
			Case("Account").
				Auto(func(ctx context.Context) error {
					fmt.Println("   [DSL AUTO] Account recovery pipeline executed automatically")
					return nil
				}).
				Confirm("Would you like to send a 2FA verification SMS?", func(ctx context.Context, prompt string) error {
					fmt.Printf("   [DSL CONFIRM] Prompting user: %q\n", prompt)
					return nil
				}).
			Case("Delivery").
				Auto(func(ctx context.Context) error {
					fmt.Println("   [DSL AUTO] Real-time courier dispatch status returned")
					return nil
				}).
			Default(func(ctx context.Context) error {
				fmt.Println("   [DSL DEFAULT] Fallback exception handler invoked")
				return nil
			}).
			Evaluate(ctx)

		if err != nil {
			fmt.Printf("   [DSL ERROR] %v\n", err)
		}
	}
	fmt.Println()

	// =========================================================================
	// 6. Multi-Metric Neural Inspection (ai.Inspect)
	// =========================================================================
	fmt.Println("[Step 6] Multi-Metric Neural Guard Inspection (ai.Inspect)")
	fmt.Println("Concept: BPE subword decomposition, SingleCharRatio, Shannon Entropy, LogSumExp Free Energy.")

	inspectQuery := "sent return parcel need refund"
	trace := ai.Inspect(inspectQuery)

	fmt.Printf(">> Inspected Query: %q\n", inspectQuery)
	fmt.Printf("   ├─ BPE Token Sequence: %v\n", trace.TokenIDs)
	fmt.Printf("   ├─ Subwords Extracted: %v\n", trace.Subwords)
	fmt.Printf("   ├─ Single Character Ratio: %.2f\n", trace.SingleCharRatio)
	fmt.Printf("   ├─ Predicted Intent: %s (Confidence: %.4f)\n", trace.PredictedLabel, trace.Confidence)
	fmt.Printf("   ├─ Secondary Intent: %s (Margin Gap: %.4f)\n", trace.SecondaryLabel, trace.Margin)
	fmt.Printf("   ├─ Shannon Prediction Entropy: %.4f\n", trace.Entropy)
	fmt.Printf("   ├─ Free Energy (LogSumExp): %.4f\n", trace.Energy)
	fmt.Printf("   └─ Microsecond Latency: %d μs\n\n", trace.LatencyMicros)

	// =========================================================================
	// 7. Lock-Free Atomic Hot-Swap & Telemetry Ring Buffer
	// =========================================================================
	fmt.Println("[Step 7] Zero-Downtime Atomic Hot-Swap & Active Learning Telemetry")
	fmt.Println("Concept: Dynamically inject new intent domain ('TechSupport') at runtime without service restart.")

	newQuery := "hardware device driver crash kernel panic"
	fmt.Printf(">> Before Hot-Swap: Query %q -> Branch: %q (Unrecognized / Default)\n", newQuery, ai.Select(newQuery))

	ai.EnableTelemetry(128)
	err = ai.AppendDataMap(map[string][]string{
		"TechSupport": {
			"hardware device driver crash kernel panic",
			"firmware update failed reboot loop",
			"blue screen memory diagnostic dump",
			"graphic card display driver error",
		},
	})
	if err != nil {
		panic(fmt.Sprintf("Hot-swap retraining failed: %v", err))
	}

	swappedBranch := ai.Select(newQuery)
	swappedTrace := ai.Inspect(newQuery)
	fmt.Printf(">> After Hot-Swap:  Query %q -> Branch: %q\n", newQuery, swappedBranch)
	fmt.Printf("   (Confidence: %.2f, 2nd: %s, Margin: %.2f, IsFallback=%v, IsAmbiguous=%v)\n",
		swappedTrace.Confidence, swappedTrace.SecondaryLabel, swappedTrace.Margin, swappedTrace.IsFallback, swappedTrace.IsAmbiguous)

	telemetryEvents := ai.DrainTelemetry()
	fmt.Printf(">> Telemetry Events Recorded: %d event(s)\n\n", len(telemetryEvents))

	fmt.Println("=========================================================================")
	fmt.Println("NeuroBranch Intelligent Branching Verification Completed Successfully.")
	fmt.Println("=========================================================================")
}

func printBanner() {
	border := strings.Repeat("=", 73)
	fmt.Println(border)
	fmt.Println("   NEUROBRANCH INTELLIGENT BRANCHING BENCHMARK SUITE")
	fmt.Println("   Pure Go In-Memory Neural Control Flow Routing Engine")
	fmt.Println(border)
	fmt.Println()
}
