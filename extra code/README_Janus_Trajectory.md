# Φ-Dwell: The Trajectory & Janus Suite

**Experimental module for continuous-time analysis of the 40D eigenmode configuration space.**

While the main Φ-Dwell pipeline uses discrete tokenization (“words”) to find clinical biomarkers, this suite attempts to analyze the *physics of the transition itself*. It asks:

> **How does the brain move from one state to another, and which frequency band drives that movement?**

---

## 1. The Hypothesis: Holographic Multiplexing (“Janus”)

Inspired by our complex-valued neural network experiments (`janus_cabbage.py`), we hypothesized that the brain uses different frequency bands as **orthogonal phase axes** to store superpositioned information.

### The Theory

To switch cognitive tasks, the brain *rotates* its active processing layer.

**Prediction**
- **Delta / Theta** drive the trajectory during **Rest** (idling, scaffolding).
- **Beta** acts as the *Pilot* during **Motor Tasks**.

---

## 2. What Was Attempted

Three tools were built to test this idea:

- **`phidwell_janus_pilot.py`**  
  Calculates the *velocity of change* within each frequency band’s eigenmode projection to determine which band is changing fastest (“steering”) at any given millisecond.

- **`phidwell_pilot_comparison.py`**  
  Runs the steering analysis across **N = 109 subjects** (Rest vs. Task) to detect a consistent *Pilot Shift*.

- **`trajectory_explorer.py`**  
  Visualizes the raw **40-dimensional orbit** of the brain’s phase-field geometry.

---

## 3. The Findings (Signal vs. Artifact)

### ❌ The Artifact: The “Beta Pilot”

Across 109 subjects, **Beta band dominates steering (~40–43%) in BOTH Rest and Task**.

**Analysis**

This is most likely a **bandwidth confound**:

- Beta spans ~17 Hz (13–30 Hz)
- Delta spans ~3 Hz (1–4 Hz)

Higher bandwidth allows faster phase decorrelation, producing higher apparent “velocity” in eigenmode space regardless of cognitive state.

**Conclusion**

Raw steering velocity **cannot** distinguish Rest from Task *without normalizing for intrinsic bandwidth volatility*.

---

### ✅ The Signal: The “Flight Recorder”

While the *Pilot* metric was confounded, **Trajectory Geometry proved robust**.

The logic was consolidated into:

- **`fuse_trajectory_sets.py`**

Instead of asking *“Who is steering?”*, this approach asks:

> **“Are we moving — or staying?”**

#### Dwell Detection

Neural velocity is *not* constant. It **bursts and pauses**.

By filtering out high-velocity transit periods, we revealed stable **Dwell epochs** — moments where the brain locks into a geometric configuration.

#### The “Barcode” of Thought

These Dwells form a **state-machine barcode**:
- Stable plateaus (Dwells)
- Sharp transitions (State changes)

#### Modular Rotation

The brain does *not* jump randomly in 40D space.

Instead, it performs **Modular Rotations**:

> Example: Theta shifts from Mode 1 → 2  
> Alpha and Beta remain in Mode 1

This supports a refined **Janus hypothesis**:

- High frequencies provide **stable scaffolding**
- Low frequencies act as **rotating tumblers** that access new state addresses

---

## 4. File Guide

### The Trajectory Engine

- **`trajectory_explorer.py`**  
  Visualization engine. Produces 9-panel dashboards showing:
  - Orbit geometry
  - Speed
  - Curvature
  - Recurrence plots  
  Use this to *see* the geometry.

- **`fuse_trajectory_sets.py`** *(The Flight Recorder)*  
  Logic core. Merges velocity data with set extraction to produce a clean timeline of:
  - Stable brain states (Dwells)
  - Transition periods  
  Use this to extract the state-machine barcode.

---

### The “Janus” Experiments

- **`phidwell_janus_pilot.py`**  
  Per-band steering velocity for a single subject.

- **`phidwell_pilot_comparison.py`**  
  Batch comparison of Pilot profiles across conditions.

- **`janus_cabbage.py`**  
  Original proof-of-concept **complex-valued neural network** demonstrating how orthogonal phase axes can store two images in one weight set. Included for theoretical grounding.

---

## 5. Synthesis with the Main Repo

This suite provides the **micro-mechanistic explanation** for the findings in the main Φ-Dwell repository:

- **Main Repo Finding**  
  Alzheimer’s brains show *more vocabulary but less structure*.

- **Janus Suite Explanation**  
  The Flight Recorder reveals that stable Dwells require **viscosity** — friction against entropy.

In Alzheimer’s:
- Neural velocity never drops low enough to lock into a Dwell
- The trajectory **slips continuously**
- The system wanders instead of stepping cleanly between stations

This manifests clinically as rich local activity with poor global organization.

---

**Φ-Dwell · Janus Suite**  
*Intelligence is not a point — it is a trajectory.*
