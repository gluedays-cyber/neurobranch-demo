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
	// 1. Train & Load Domain Neural Network from CSV Dataset with Functional Options
	// =========================================================================
	fmt.Println("[Step 1] Compile Neural AI Model Directly from CSV Dataset with Functional Options")
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
	// v3.0 Functional Options API: externalize configuration thresholds
	ai, err := neurobranch.TrainAIWithOptions(
		samples,
		cfg,
		neurobranch.WithConfidenceThreshold(0.80, 0.45),
		neurobranch.WithEnergyThreshold(3.0),
		neurobranch.WithMarginCutoff(0.15),
		neurobranch.WithPatternGuard(true),
	)
	if err != nil {
		panic(fmt.Sprintf("In-memory training failed: %v", err))
	}
	trainDuration := time.Since(startTrain)

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

	testQueries := []struct {
		Query string
		Desc  string
	}{
		// Refund Intent
		{"reverse transaction charge refund", "Refund: Standard Charge Reversal"},
		{"give me my cash back right now", "Refund: Colloquial Demand"},
		{"refud please double charg", "Refund: Compound Typos"},
		{"package arrived broken want money back", "Refund: Cross-Domain Overlap Query"},

		// Delivery Intent
		{"where is my package delivery tracking", "Delivery: Standard Tracking Query"},
		{"still waiting for my box when does it arrive", "Delivery: Conversational ETA Inquiry"},
		{"traxking parcel not moving", "Delivery: Typo Tracking with Delay Issue"},
		{"courier left note but no box at door", "Delivery: Courier Exception Handling"},

		// Account Intent
		{"locked out of user account login", "Account: Standard Account Lockout"},
		{"cant log in pasword reset link please", "Account: Slang and Typo Combination"},
		{"2fa auth code not sending to phone", "Account: Two-Factor Auth Issue"},
		{"someone hacked my profile change password", "Account: Security Breach Incident"},

		// Billing Intent
		{"billing credit card monthly receipt", "Billing: Standard Monthly Receipt"},
		{"where can i download corporate tax invoice", "Billing: Tax Invoice Issuance"},
		{"send recipt to my email plz", "Billing: Slang and Incomplete Token"},

		// Unlearned Noise / Degenerate Input (Layer 1 Fail-Safe)
		{"zzzzzzzz xxxxxxxx yyyyyyyy", "Noise: Repetitive Character Burst"},
		{"!@#$%^&*()_+<>?:{}", "Noise: Pure Symbol Sequence"},
		{"asdfghjkl qwertyuiop", "Noise: Random Keyboard Smash"},

		// Out-of-Domain / Semantic Guard (Layer 2 OOD Guard)
		{"quantum physics entangled photon spin", "OOD: Quantum Mechanics Query"},
		{"how to bake sourdough bread with yeast", "OOD: Cooking Recipe Query"},
		{"tell me the capital city of France", "OOD: General Trivia Query"},
	}

	for _, item := range testQueries {
		fmt.Printf("\n>> [%s]\n   Query: %q\n", item.Desc, item.Query)
		switch branch := ai.Select(item.Query); branch {
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

	guardQueries := []struct {
		Query     string
		Expected  string
		Threshold float64
	}{
		{"please process full refund immediately", "Refund", 0.80},
		{"pls refnd my money asap", "Refund", 0.75},
		{"unlock user account credentials", "Account", 0.85},
		{"track shipment number status", "Delivery", 0.80},
		{"download monthly tax invoice receipt", "Billing", 0.80},
	}

	for _, g := range guardQueries {
		fmt.Printf(">> Guard Target: %q (Target: %s, Threshold: %.2f)\n", g.Query, g.Expected, g.Threshold)
		if ai.If(g.Query, g.Expected) {
			fmt.Printf("   [PASSED] ai.If(query, %q) == true\n", g.Expected)
		} else {
			fmt.Printf("   [FAILED] ai.If(query, %q) == false\n", g.Expected)
		}

		if ai.Is(g.Query, g.Expected, g.Threshold) {
			fmt.Printf("   [PASSED] ai.Is(query, %q, %.2f) == true -> High Confidence Confirmed\n", g.Expected, g.Threshold)
		} else {
			fmt.Printf("   [REJECTED] ai.Is(query, %q, %.2f) == false -> Low Confidence / Re-routing\n", g.Expected, g.Threshold)
		}
	}
	fmt.Println()

	// =========================================================================
	// 4. Native Go Comma-ok Idiom (ai.Match)
	// =========================================================================
	fmt.Println("[Step 4] Native Go Comma-ok Idiom (ai.Match)")
	fmt.Println("Concept: (intent, confident) return signature distinguishes definite vs ambiguous states.")

	evalInputs := []struct {
		Query string
		Desc  string
	}{
		{"want my money back refund", "Decisive Refund Request"},
		{"courier ETA package tracking inquiry", "Decisive Delivery Request"},
		{"reset two factor authentication credentials", "Decisive Account Request"},
		{"shipped returned parcel when do i get payment", "Boundary Overlap Query: Delivery vs Refund"},
		{"is account email tied to billing statements", "Boundary Overlap Query: Account vs Billing"},
		{"zzzzzzzz xxxxxxxx yyyyyyyy", "Degenerate Noise String"},
		{"the weather is nice today", "Out-of-Domain Casual Conversation"},
	}

	for _, input := range evalInputs {
		fmt.Printf(">> [%s] Query: %q\n", input.Desc, input.Query)
		if intent, confident := ai.Match(input.Query); confident {
			fmt.Printf("   [COMMA-OK] Decisive intent identified: %s (confident: true)\n", intent)
		} else if intent != "" {
			fmt.Printf("   [AMBIGUOUS] Borderline intent detected: %s (confident: false, human review recommended)\n", intent)
		} else {
			fmt.Println("   [OOD/REJECT] Out-of-domain or unlearned noise safely rejected")
		}
	}
	fmt.Println()

	// =========================================================================
	// 5. Declarative Fluent DSL & Interactive Confirmation (ai.Switch)
	// =========================================================================
	fmt.Println("[Step 5] Declarative Fluent DSL & Interactive Confirmation (ai.Switch)")
	fmt.Println("Concept: Fluent chaining for automatic dispatch (Auto), confirmation prompts (Confirm), and Default.")

	ctx := context.Background()
	dslQueries := []struct {
		Text string
		Desc string
	}{
		{"locked out of user account login", "High-Confidence Account Query"},
		{"where is my package delivery tracking", "High-Confidence Delivery Query"},
		{"need tax invoice receipt for corporate accounting", "High-Confidence Billing Query"},
		{"money back request for damaged shipment", "Refund Query with Ambiguity Guard"},
		{"zzzzzzzz xxxxxxxx yyyyyyyy", "Degenerated Noise Query"},
		{"superconductor resistance zero temperature", "Out-of-Domain Query"},
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
			Case("Billing").
			Auto(func(ctx context.Context) error {
				fmt.Println("   [DSL AUTO] Tax invoice PDF generated and dispatched")
				return nil
			}).
			Case("Refund").
			Confirm("Confirm refund authorization of this order?", func(ctx context.Context, prompt string) error {
				fmt.Printf("   [DSL CONFIRM] Processing refund validation prompt: %q\n", prompt)
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

	inspectQueries := []string{
		"sent return parcel need refund",
		"where is my package delivery tracking",
		"hardware driver crash blue screen",
		"asdfghjkl random noise string",
	}

	for _, iq := range inspectQueries {
		trace := ai.Inspect(iq)
		fmt.Printf(">> Inspected Query: %q\n", iq)
		fmt.Printf("   ├─ BPE Token Sequence: %v\n", trace.TokenIDs)
		fmt.Printf("   ├─ Subwords Extracted: %v\n", trace.Subwords)
		fmt.Printf("   ├─ Single Character Ratio: %.2f\n", trace.SingleCharRatio)
		fmt.Printf("   ├─ Predicted Intent: %s (Confidence: %.4f)\n", trace.PredictedLabel, trace.Confidence)
		fmt.Printf("   ├─ Secondary Intent: %s (Margin Gap: %.4f)\n", trace.SecondaryLabel, trace.Margin)
		fmt.Printf("   ├─ Shannon Prediction Entropy: %.4f\n", trace.Entropy)
		fmt.Printf("   ├─ Free Energy (LogSumExp): %.4f\n", trace.Energy)
		fmt.Printf("   └─ Microsecond Latency: %d μs\n\n", trace.LatencyMicros)
	}

	// =========================================================================
	// 7. Dynamic Runtime Reconfiguration & Temperature Scaling (v3.0)
	// =========================================================================
	fmt.Println("[Step 7] Dynamic Policy Reconfiguration & Temperature Scaling")
	fmt.Println("Concept: Adapt energy cutoffs and Softmax temperature scaling dynamically without retraining.")

	fmt.Printf(">> Default Temperature: %.2f\n", ai.Temperature())
	ai.SetTemperature(1.5)
	fmt.Printf("   Updated Temperature (Softened Distribution): %.2f\n", ai.Temperature())
	tempTrace := ai.Inspect("where is my package delivery tracking")
	fmt.Printf("   Inference with T=1.5 -> Predicted: %s (Confidence: %.4f, Entropy: %.4f)\n",
		tempTrace.PredictedLabel, tempTrace.Confidence, tempTrace.Entropy)

	ai.SetTemperature(1.0) // Reset to standard
	ai.SetEnergyThreshold(3.2)
	fmt.Printf("   Updated Energy Threshold: %.2f (Strict Out-of-Domain Rejection)\n\n", 3.2)

	// =========================================================================
	// 8. Lock-Free Atomic Hot-Swap & Telemetry Ring Buffer
	// =========================================================================
	fmt.Println("[Step 8] Zero-Downtime Atomic Hot-Swap & Active Learning Telemetry")
	fmt.Println("Concept: Dynamically inject new intent domain ('TechSupport') at runtime without service restart.")

	testHotSwapQueries := []string{
		"hardware device driver crash kernel panic",
		"blue screen of death error after driver installation",
		"gpu driver keeps crashing screen black",
	}

	fmt.Println(">> [Before Hot-Swap Evaluation]")
	for _, hq := range testHotSwapQueries {
		fmt.Printf("   Query: %q -> Branch: %q\n", hq, ai.Select(hq))
	}

	ai.EnableTelemetry(128)

	// Balanced training samples to avoid class imbalance against base dataset
	err = ai.AppendDataMap(map[string][]string{
		"TechSupport": {
			"hardware device driver crash kernel panic",
			"blue screen of death error after driver installation",
			"operating system kernel panic on boot",
			"graphics card driver failure display glitch",
			"device driver not recognized by operating system",
			"audio interface driver buffer underrun crackle",
			"firmware update failure causing system freeze",
			"hardware diagnostics reporting memory parity error",
			"usb host controller driver failed to initialize",
			"pci express bus error causing system reboot",
			"driver crashed help pc blue screen",
			"gpu driver keeps crashing screen black",
			"motherboard firmware corrupted boot loop",
			"bsod blue screen crash dump error",
			"hardwere driver crash and panic",
		},
	})
	if err != nil {
		panic(fmt.Sprintf("Hot-swap retraining failed: %v", err))
	}

	fmt.Println("\n>> [After Hot-Swap Evaluation]")
	for _, hq := range testHotSwapQueries {
		swappedBranch := ai.Select(hq)
		swappedTrace := ai.Inspect(hq)
		fmt.Printf("   Query: %q -> Branch: %q\n", hq, swappedBranch)
		fmt.Printf("   (Confidence: %.2f, 2nd: %s, Margin: %.2f, IsFallback=%v, IsAmbiguous=%v)\n",
			swappedTrace.Confidence, swappedTrace.SecondaryLabel, swappedTrace.Margin, swappedTrace.IsFallback, swappedTrace.IsAmbiguous)
	}

	telemetryEvents := ai.DrainTelemetry()
	fmt.Printf("\n>> Telemetry Events Recorded: %d event(s)\n\n", len(telemetryEvents))

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
