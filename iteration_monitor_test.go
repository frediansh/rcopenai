package rcopenai

import (
	"strings"
	"sync"
	"testing"
)

func TestNewIterationMonitor_DefaultThreshold_Zero(t *testing.T) {
	m := NewIterationMonitor(10, 0)
	if m.warningThreshold != 0.75 {
		t.Errorf("warningThreshold = %f, want 0.75", m.warningThreshold)
	}
	if m.maxIterations != 10 {
		t.Errorf("maxIterations = %d, want 10", m.maxIterations)
	}
}

func TestNewIterationMonitor_DefaultThreshold_TooHigh(t *testing.T) {
	m := NewIterationMonitor(10, 1.0)
	if m.warningThreshold != 0.75 {
		t.Errorf("warningThreshold = %f, want 0.75 (clamped from 1.0)", m.warningThreshold)
	}
}

func TestNewIterationMonitor_DefaultThreshold_Negative(t *testing.T) {
	m := NewIterationMonitor(10, -0.5)
	if m.warningThreshold != 0.75 {
		t.Errorf("warningThreshold = %f, want 0.75 (clamped from -0.5)", m.warningThreshold)
	}
}

func TestNewIterationMonitor_CustomThreshold(t *testing.T) {
	m := NewIterationMonitor(10, 0.5)
	if m.warningThreshold != 0.5 {
		t.Errorf("warningThreshold = %f, want 0.5", m.warningThreshold)
	}
}

func TestNewIterationMonitor_DefaultMaxIterations(t *testing.T) {
	m := NewIterationMonitor(0, 0.5)
	if m.maxIterations != defaultMaxIterations {
		t.Errorf("maxIterations = %d, want %d", m.maxIterations, defaultMaxIterations)
	}
}

func TestNewIterationMonitor_NegativeMax(t *testing.T) {
	m := NewIterationMonitor(-5, 0.5)
	if m.maxIterations != defaultMaxIterations {
		t.Errorf("maxIterations = %d, want %d (clamped)", m.maxIterations, defaultMaxIterations)
	}
}

func TestIncrement_BasicFlow(t *testing.T) {
	// maxIterations=4, threshold=0.5 → warningPoint = int(4*0.5) = 2
	m := NewIterationMonitor(4, 0.5)

	cur, warn, max := m.Increment() // iteration 1
	if cur != 1 || warn || max {
		t.Errorf("iter 1: cur=%d warn=%v max=%v, want 1 false false", cur, warn, max)
	}

	cur, warn, max = m.Increment() // iteration 2 — triggers warning
	if cur != 2 || !warn || max {
		t.Errorf("iter 2: cur=%d warn=%v max=%v, want 2 true false", cur, warn, max)
	}

	cur, warn, max = m.Increment() // iteration 3 — warning already sent
	if cur != 3 || warn || max {
		t.Errorf("iter 3: cur=%d warn=%v max=%v, want 3 false false", cur, warn, max)
	}

	cur, warn, max = m.Increment() // iteration 4 — max reached
	if cur != 4 || warn || !max {
		t.Errorf("iter 4: cur=%d warn=%v max=%v, want 4 false true", cur, warn, max)
	}
}

func TestIncrement_WarningSentOnlyOnce(t *testing.T) {
	m := NewIterationMonitor(10, 0.5)
	warnCount := 0
	for i := 0; i < 10; i++ {
		_, warn, _ := m.Increment()
		if warn {
			warnCount++
		}
	}
	if warnCount != 1 {
		t.Errorf("warning fired %d times, want exactly 1", warnCount)
	}
}

func TestIncrement_MaxIterationsOne(t *testing.T) {
	m := NewIterationMonitor(1, 0.5)
	// warningPoint = int(1*0.5) = 0, clamped to 1
	cur, _, max := m.Increment()
	if cur != 1 || !max {
		t.Errorf("cur=%d max=%v, want 1 true", cur, max)
	}
}

func TestIncrement_WarningAtThresholdBoundary(t *testing.T) {
	// maxIterations=8, threshold=0.75 → warningPoint = int(8*0.75) = 6
	m := NewIterationMonitor(8, 0.75)
	var warnAt int
	for i := 1; i <= 8; i++ {
		_, warn, _ := m.Increment()
		if warn {
			warnAt = i
		}
	}
	if warnAt != 6 {
		t.Errorf("warning triggered at iteration %d, want 6", warnAt)
	}
}

func TestReset_ClearsState(t *testing.T) {
	m := NewIterationMonitor(4, 0.5)
	m.Increment()
	m.Increment()
	m.Increment()

	m.Reset()

	if m.currentIteration != 0 {
		t.Errorf("currentIteration = %d after reset, want 0", m.currentIteration)
	}
	if m.warningSent {
		t.Error("warningSent = true after reset, want false")
	}

	// After reset, warning should fire again at threshold
	_, warn, _ := m.Increment() // iter 1
	if warn {
		t.Error("unexpected warning at iteration 1 after reset")
	}
	_, warn, _ = m.Increment() // iter 2 — should trigger warning again
	if !warn {
		t.Error("expected warning at iteration 2 after reset, got false")
	}
}

func TestFormatIterationStatus(t *testing.T) {
	m := NewIterationMonitor(10, 0.5)
	m.Increment()
	m.Increment()
	m.Increment()

	s := m.FormatIterationStatus()
	if !strings.Contains(s, "3/10") {
		t.Errorf("FormatIterationStatus = %q, want to contain '3/10'", s)
	}
	if !strings.Contains(s, "30.0%") {
		t.Errorf("FormatIterationStatus = %q, want to contain '30.0%%'", s)
	}
}

func TestFormatIterationStatus_Zero(t *testing.T) {
	m := NewIterationMonitor(10, 0.5)
	s := m.FormatIterationStatus()
	if !strings.Contains(s, "0/10") {
		t.Errorf("FormatIterationStatus = %q, want to contain '0/10'", s)
	}
}

func TestIncrement_Concurrent(t *testing.T) {
	m := NewIterationMonitor(200, 0.5)
	var wg sync.WaitGroup
	warnCount := 0
	var mu sync.Mutex

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, warn, _ := m.Increment()
			if warn {
				mu.Lock()
				warnCount++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if warnCount != 1 {
		t.Errorf("warning fired %d times concurrently, want exactly 1", warnCount)
	}
	if m.currentIteration != 100 {
		t.Errorf("currentIteration = %d after 100 concurrent increments, want 100", m.currentIteration)
	}
}
