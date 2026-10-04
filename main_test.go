package main

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/gluedays-cyber/neurobranch"
)

func setupTestAI(t *testing.T) *neurobranch.AI {
	t.Helper()
	csvPath := filepath.Join("data", "train.csv")
	samples, err := neurobranch.LoadCSVDataset(csvPath)
	if err != nil {
		t.Fatalf("Failed to load dataset from %s: %v", csvPath, err)
	}

	cfg := neurobranch.DefaultTrainConfig()
	cfg.Seed = 42
	cfg.Epochs = 100

	ai, err := neurobranch.TrainInMemory(samples, cfg)
	if err != nil {
		t.Fatalf("TrainInMemory failed: %v", err)
	}
	return ai
}

func TestIntelligentSwitchBranching(t *testing.T) {
	ai := setupTestAI(t)

	// 1. Definite match for Refund
	refundBranch := ai.Select("cancel payment and request refund")
	if refundBranch != "Refund" {
		t.Errorf("expected 'Refund' branch, got %q", refundBranch)
	}

	// 2. Definite match for Delivery
	deliveryBranch := ai.Select("where is my package delivery tracking")
	if deliveryBranch != "Delivery" {
		t.Errorf("expected 'Delivery' branch, got %q", deliveryBranch)
	}

	// 3. Definite match for Account
	accountBranch := ai.Select("locked out of user account login")
	if accountBranch != "Account" {
		t.Errorf("expected 'Account' branch, got %q", accountBranch)
	}

	// 4. Definite match for Billing
	billingBranch := ai.Select("billing credit card monthly receipt")
	if billingBranch != "Billing" {
		t.Errorf("expected 'Billing' branch, got %q", billingBranch)
	}

	// 5. Safe Fallback on Out-of-Domain (OOD) / Degenerated Input
	oodBranch := ai.Select("zzzzzzzz xxxxxxxx yyyyyyyy")
	if oodBranch != "" {
		t.Errorf("expected empty string (default fallback) for noise input, got %q", oodBranch)
	}
}

func TestNativeGuardClauses(t *testing.T) {
	ai := setupTestAI(t)

	if !ai.If("where is my package delivery tracking", "Delivery") {
		t.Errorf("ai.If failed for valid delivery query")
	}

	if ai.If("where is my package delivery tracking", "Refund") {
		t.Errorf("ai.If should reject mismatching intent")
	}
}

func TestNativeCommaOkBranching(t *testing.T) {
	ai := setupTestAI(t)

	intent, confident := ai.Match("want my money back refund")
	if !confident || intent != "Refund" {
		t.Errorf("expected confident match for Refund, got %s (confident=%v)", intent, confident)
	}

	// Unlearned degenerated noise query should reject confident match
	_, noiseConfident := ai.Match("zzzzzzzz xxxxxxxx yyyyyyyy")
	if noiseConfident {
		t.Errorf("expected false confident for degenerated noise query")
	}
}

func TestDeclarativeDSLBranching(t *testing.T) {
	ai := setupTestAI(t)
	ctx := context.Background()

	var executedBranch string
	err := ai.Switch("where is my package delivery tracking").
		Case("Refund").
			Auto(func(ctx context.Context) error {
				executedBranch = "Refund"
				return nil
			}).
		Case("Delivery").
			Auto(func(ctx context.Context) error {
				executedBranch = "Delivery"
				return nil
			}).
		Default(func(ctx context.Context) error {
			executedBranch = "Default"
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("DSL Evaluate failed: %v", err)
	}
	if executedBranch != "Delivery" {
		t.Errorf("expected 'Delivery' branch executed, got %q", executedBranch)
	}

	// Default fallback handler test
	executedBranch = ""
	err = ai.Switch("zzzzzzzz xxxxxxxx yyyyyyyy").
		Case("Delivery").
			Auto(func(ctx context.Context) error {
				executedBranch = "Delivery"
				return nil
			}).
		Default(func(ctx context.Context) error {
			executedBranch = "Default"
			return nil
		}).
		Evaluate(ctx)

	if err != nil {
		t.Fatalf("DSL Evaluate fallback failed: %v", err)
	}
	if executedBranch != "Default" {
		t.Errorf("expected 'Default' branch on noise query, got %q", executedBranch)
	}
}

func TestAtomicHotSwap(t *testing.T) {
	ai := setupTestAI(t)

	hotSwapQuery := "hardware device driver crash kernel panic"
	// Before hot-swap: 'TechSupport' does not exist in dataset
	if ai.Select(hotSwapQuery) == "TechSupport" {
		t.Fatalf("TechSupport branch should not exist prior to hot-swap")
	}

	err := ai.AppendDataMap(map[string][]string{
		"TechSupport": {
			"hardware device driver crash kernel panic",
			"firmware update failed reboot loop",
			"blue screen memory diagnostic dump",
			"graphic card display driver error",
		},
	})
	if err != nil {
		t.Fatalf("AppendDataMap failed: %v", err)
	}

	// After hot-swap: routing immediately updates to TechSupport with zero downtime
	if ai.Select(hotSwapQuery) != "TechSupport" {
		t.Errorf("expected 'TechSupport' branch after hot-swap, got %q", ai.Select(hotSwapQuery))
	}
}
