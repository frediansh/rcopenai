package glopenai

import (
	"fmt"
	"sync"
)

// IterationMonitor tracks model-call iteration progress.
type IterationMonitor struct {
	maxIterations    int
	warningThreshold float64

	mu               sync.RWMutex
	currentIteration int
	warningSent      bool
}

func NewIterationMonitor(maxIterations int, warningThreshold float64) *IterationMonitor {
	if maxIterations <= 0 {
		maxIterations = defaultMaxIterations
	}
	if warningThreshold <= 0 || warningThreshold >= 1 {
		warningThreshold = 0.75
	}
	return &IterationMonitor{
		maxIterations:    maxIterations,
		warningThreshold: warningThreshold,
	}
}

func (m *IterationMonitor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.currentIteration = 0
	m.warningSent = false
}

func (m *IterationMonitor) Increment() (current int, warning bool, maxReached bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.currentIteration++
	current = m.currentIteration

	warningPoint := int(float64(m.maxIterations) * m.warningThreshold)
	if warningPoint < 1 {
		warningPoint = 1
	}
	if !m.warningSent && current >= warningPoint {
		m.warningSent = true
		warning = true
	}
	maxReached = current >= m.maxIterations
	return
}

func (m *IterationMonitor) FormatIterationStatus() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pct := float64(m.currentIteration) / float64(m.maxIterations) * 100
	return fmt.Sprintf("Iteration %d/%d (%.1f%%)", m.currentIteration, m.maxIterations, pct)
}
