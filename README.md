# Pulsar-Detection-Fast-Folding-Algorithm

# 🌌 Pulsar Detection & Universe Expansion Analysis System

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)

A comprehensive radio telescope data analysis system for pulsar detection using fast folding algorithms, dispersion measure analysis, and cosmological expansion studies through pulsar mapping.

## 🌟 Features

### Pulsar Detection System
- ✨ **Fast Folding Algorithm** - Efficient period search across wide parameter space
- 📡 **Multi-channel Dispersion Analysis** - Dedispersion across 128+ frequency channels
- 🔍 **Automated DM Search** - Intelligent dispersion measure optimization
- 📊 **Pulse Profile Extraction** - High-fidelity pulse shape recovery
- 🎯 **Signal-to-Noise Optimization** - Adaptive thresholding and detection

### Universe Expansion Analysis
- 🌌 **3D Pulsar Distribution Mapping** - Realistic spatial distribution modeling
- 📈 **Hubble Constant Measurement** - Statistical fitting of expansion rate
- 🔴 **Redshift Analysis** - Cosmological distance-velocity relationships
- ⚖️ **Expansion vs Contraction Detection** - Determine universe fate
- 🎨 **Multi-dimensional Visualization** - Comprehensive data presentation

### Advanced Methods
- ⏱️ **Pulsar Timing Analysis** - Precision period and period derivative measurement
- 🌍 **Binary Pulsar Effects** - Shapiro delay and orbital parameter extraction
- 🌊 **Gravitational Wave Signatures** - Nanohertz GW detection via timing arrays
- 📏 **Cosmological Distance Ladder** - Multiple distance measure calculations
- 🧮 **Bayesian Parameter Estimation** - MCMC-based uncertainty quantification

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Module Documentation](#module-documentation)
- [Scientific Background](#scientific-background)
- [Output & Visualization](#output--visualization)
- [Performance Notes](#performance-notes)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

```bash
pip install numpy>=1.21.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
```

Or install all at once:

```bash
pip install numpy scipy matplotlib
```

### Clone Repository

```bash
git clone https://github.com/yourusername/pulsar-detection-system.git
cd pulsar-detection-system
```

### Optional: Create Virtual Environment

```bash
# Create virtual environment
python -m venv pulsar_env

# Activate (Windows)
pulsar_env\Scripts\activate

# Activate (Mac/Linux)
source pulsar_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### Basic Pulsar Detection

```python
from pulsar_detector import PulsarDetector

# Initialize detector
detector = PulsarDetector(sampling_rate=2000.0, dm_max=50.0)

# Generate synthetic pulsar signal for testing
time_array, signal_2d = detector.generate_pulsar_signal(
    duration=10.0,        # 10 seconds of data
    period=0.1,           # 100ms pulsar period
    pulse_width=0.05,     # 5% duty cycle
    dm=25.0,              # Dispersion measure
    snr=15.0              # Signal-to-noise ratio
)

# Detect the pulsar
results = detector.detect_pulsar(
    signal_2d,
    period_range=(0.01, 1.0),
    n_periods=5000,
    dm_steps=30
)

# Display results
print(f"Detected Period: {results['best_period']:.6f} seconds")
print(f"Detected DM: {results['best_dm']:.2f} pc/cm³")
print(f"Detection Statistic: {results['detection_statistic']:.2f}")

# Plot results
detector.plot_results(results, signal_2d, time_array)
```

### Universe Expansion Analysis

```python
from universe_simulator import UniverseSimulator

# Create expanding universe
universe = UniverseSimulator(hubble_constant=70.0)

# Generate pulsar distribution
pulsar_data = universe.generate_pulsar_distribution(
    n_pulsars=800,
    max_distance=3000.0  # Mpc
)

# Observe pulsars (measure redshifts)
obs_data = universe.observe_pulsars(pulsar_data)

# Analyze expansion
analysis = universe.analyze_expansion(obs_data)

# Display results
print(f"Universe Status: {analysis['status']}")
print(f"Hubble Constant: {analysis['fitted_h0']:.2f} km/s/Mpc")
print(f"Universe Age: {analysis['age_universe_years']/1e9:.2f} billion years")

# Visualize
universe.plot_pulsar_map(obs_data, analysis)
```

### Advanced Timing Analysis

```python
from advanced_methods import AdvancedPulsarMethods

methods = AdvancedPulsarMethods()

# Pulsar timing analysis
timing_result = methods.pulsar_timing_analysis(times, toas, toa_errors)

# Calculate Shapiro delay in binary system
delay = methods.shapiro_delay(
    orbital_phase,
    companion_mass=1.4,      # Solar masses
    orbital_inclination=np.radians(60),
    semi_major_axis=2.3      # Light-seconds
)

# Detect gravitational wave signatures
gw_signal = methods.gravitational_wave_signature(
    times,
    gw_amplitude=1e-15,
    gw_frequency=1e-8        # Hz
)
```

## 📖 Usage Examples

### Example 1: Real Radio Telescope Data

```python
import numpy as np
from pulsar_detector import PulsarDetector

# Load your radio telescope data
# Expected format: 2D array [channels, time_samples]
data = np.load('radio_observation.npy')

# Initialize detector with your observation parameters
detector = PulsarDetector(
    sampling_rate=1000.0,   # Hz
    dm_max=100.0            # Maximum DM to search
)

# Search for pulsars
results = detector.detect_pulsar(
    data,
    period_range=(0.001, 10.0),  # 1ms to 10s
    n_periods=10000,
    dm_steps=50
)

# Check if detection is significant
if results['detection_statistic'] > 50:
    print("✅ Significant pulsar detection!")
    print(f"Period: {results['best_period']*1000:.2f} ms")
    print(f"DM: {results['best_dm']:.2f} pc/cm³")
else:
    print("❌ No significant pulsar detected")
```

### Example 2: Comparing Multiple Pulsars

```python
from pulsar_detector import PulsarDetector

detector = PulsarDetector(sampling_rate=2000.0)

# Simulate different pulsar types
pulsars = [
    ("Millisecond Pulsar", 0.003, 0.05, 50.0),
    ("Normal Pulsar", 0.5, 0.05, 30.0),
    ("Slow Pulsar", 2.0, 0.1, 15.0)
]

for name, period, width, dm in pulsars:
    print(f"\n🔭 Observing {name}...")
    
    time, signal = detector.generate_pulsar_signal(
        duration=20.0,
        period=period,
        pulse_width=width,
        dm=dm,
        snr=10.0
    )
    
    results = detector.detect_pulsar(signal)
    
    error = abs(results['best_period'] - period) / period * 100
    print(f"   True Period: {period:.6f} s")
    print(f"   Detected: {results['best_period']:.6f} s")
    print(f"   Error: {error:.2f}%")
```

### Example 3: Cosmological Parameter Study

```python
from universe_simulator import UniverseSimulator

# Test different Hubble constants
h0_values = [60, 70, 80]  # km/s/Mpc

for h0 in h0_values:
    print(f"\n📊 Testing H₀ = {h0} km/s/Mpc")
    
    universe = UniverseSimulator(hubble_constant=h0)
    pulsar_data = universe.generate_pulsar_distribution(n_pulsars=500)
    obs_data = universe.observe_pulsars(pulsar_data)
    analysis = universe.analyze_expansion(obs_data)
    
    print(f"   Measured H₀: {analysis['fitted_h0']:.2f} ± {analysis['h0_uncertainty']:.2f}")
    print(f"   Accuracy: {100 - abs(analysis['fitted_h0']-h0)/h0*100:.1f}%")
    print(f"   Universe Age: {analysis['age_universe_years']/1e9:.2f} Gyr")
```

## 📚 Module Documentation

### PulsarDetector Class

**Methods:**

- `generate_pulsar_signal()` - Create synthetic pulsar data with realistic dispersion
- `dedisperse_signal()` - Remove interstellar dispersion effects
- `search_dm_range()` - Search across dispersion measure parameter space
- `fast_folding_algorithm()` - Period search using fast folding technique
- `detect_pulsar()` - Complete detection pipeline
- `plot_results()` - Comprehensive visualization

**Parameters:**

- `sampling_rate`: Data sampling frequency in Hz (default: 1000.0)
- `dm_max`: Maximum dispersion measure to search in pc/cm³ (default: 100.0)

### UniverseSimulator Class

**Methods:**

- `generate_pulsar_distribution()` - Create 3D pulsar distribution
- `calculate_redshift()` - Compute cosmological redshift
- `observe_pulsars()` - Simulate observations with realistic errors
- `analyze_expansion()` - Determine expansion/contraction status
- `plot_pulsar_map()` - Multi-panel visualization

**Parameters:**

- `hubble_constant`: H₀ in km/s/Mpc (default: 70.0)
- `seed`: Random seed for reproducibility

### AdvancedPulsarMethods Class

**Methods:**

- `pulsar_timing_analysis()` - High-precision timing solution
- `shapiro_delay()` - General relativistic time delay
- `gravitational_wave_signature()` - GW-induced timing residuals
- `cosmological_distance_ladder()` - Multiple distance measures
- `dispersion_measure_distance()` - DM to distance conversion
- `pulsar_braking_index()` - Spin-down mechanism analysis
- `bayesian_parameter_estimation()` - MCMC parameter fitting

## 🔬 Scientific Background

### Fast Folding Algorithm

The fast folding algorithm tests a range of trial periods by "folding" the time series data at each period and calculating a detection statistic. When folded at the correct period, pulses align constructively, producing a strong peak in the statistic.

**Detection Statistic:**
```
χ² = Σ(profile_i - mean)² / mean
```

### Dispersion Measure

Radio waves from pulsars are dispersed by free electrons in the interstellar medium. Higher frequencies arrive earlier than lower frequencies:

**Dispersion Delay:**
```
Δt = 4.148808 × DM × (f₁⁻² - f₂⁻²) ms
```

where DM is in pc/cm³ and f is in MHz.

### Hubble's Law

The expansion of the universe is characterized by Hubble's law:

**Recession Velocity:**
```
v = H₀ × d
```

where:
- v = recession velocity (km/s)
- H₀ = Hubble constant (km/s/Mpc)
- d = distance (Mpc)

### Cosmological Redshift

The stretching of light due to universal expansion:

**Redshift:**
```
z = Δλ/λ₀ = (1 + v/c)/(1 - v/c) - 1
```

For nearby objects (z << 1):
```
z ≈ H₀d/c
```

## 📊 Output & Visualization

### Pulsar Detection Plots

The system generates 6-panel visualization:

1. **Raw Multi-channel Data** - Waterfall plot showing dispersion
2. **DM Search Results** - SNR vs dispersion measure
3. **Period Search** - Folding statistic vs trial period
4. **Dedispersed Time Series** - Signal after DM correction
5. **Folded Pulse Profile** - Final integrated pulse shape
6. **Detection Statistics** - Summary of key parameters

### Universe Expansion Plots

The system generates 6-panel cosmological analysis:

1. **3D Pulsar Distribution** - Spatial map of pulsars
2. **Hubble Diagram** - Distance vs velocity (classic!)
3. **Distance-Redshift Relation** - Observational data
4. **Sky Distribution** - Aitoff projection
5. **H₀ vs Distance** - Consistency check
6. **Analysis Summary** - Key results and interpretation

### Advanced Methods Plots

9-panel comprehensive visualization:

1. **Timing Residuals** - Precision measurement quality
2. **Period Evolution** - Long-term spin behavior
3. **Shapiro Delay** - GR test in binary systems
4. **GW Signature** - Gravitational wave signal
5. **Distance Ladder** - Multiple distance measures
6. **DM-Distance Relation** - Galactic electron density
7. **P-Pdot Diagram** - Pulsar population study
8. **Relativistic Hubble Law** - Extended analysis
9. **Summary Statistics** - Comprehensive overview

## ⚙️ Performance Notes

### Optimization Tips

**For Large Datasets:**
- Reduce `n_periods` for faster but coarser period search
- Decrease `dm_steps` if approximate DM is known
- Use known period range to narrow search space

**Memory Usage:**
- Multi-channel data: ~10 MB per 1000 channels × 100k samples
- Adjust based on available RAM

**Computation Time:**
- Single pulsar detection: 30-120 seconds (depends on parameters)
- Universe simulation (800 pulsars): 60-180 seconds
- Advanced methods demo: 20-60 seconds

### Recommended Parameters

**For Quick Testing:**
```python
n_periods=1000
dm_steps=20
duration=5.0
```

**For High Precision:**
```python
n_periods=10000
dm_steps=100
duration=30.0
```

**For Production Use:**
```python
n_periods=50000
dm_steps=200
duration=300.0  # 5 minutes
```

## 🎯 Applications

### Research Applications

- **Pulsar Surveys** - Discovery and characterization of new pulsars
- **Timing Arrays** - Gravitational wave detection (NANOGrav, EPTA, PPTA)
- **Binary Systems** - Tests of general relativity
- **Cosmology** - Independent Hubble constant measurements
- **Galactic Structure** - Free electron density mapping
- **Fundamental Physics** - Equation of state studies

### Educational Applications

- Radio astronomy laboratory exercises
- Signal processing demonstrations
- Cosmology parameter fitting
- Statistical analysis examples
- Data visualization techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New features
- Documentation improvements
- Example notebooks

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pulsar-detection-system.git
cd pulsar-detection-system

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{pulsar_detection_2024,
  author = {Your Name},
  title = {Pulsar Detection and Universe Expansion Analysis System},
  year = {2024},
  url = {https://github.com/yourusername/pulsar-detection-system},
  version = {1.0.0}
}
```

## 📖 References

**Pulsar Detection:**
- Lorimer, D. R., & Kramer, M. (2004). *Handbook of Pulsar Astronomy*. Cambridge University Press.
- Ransom, S. M. (2001). "New Search Techniques for Binary Pulsars." PhD thesis, Harvard University.

**Fast Folding:**
- Staelin, D. H. (1969). "Pulsating Radio Sources near the Crab Nebula." *Science*, 162(3861), 1481-1483.

**Cosmology:**
- Riess, A. G., et al. (1998). "Observational Evidence from Supernovae for an Accelerating Universe." *AJ*, 116, 1009.
- Perlmutter, S., et al. (1999). "Measurements of Ω and Λ from 42 High-Redshift Supernovae." *ApJ*, 517, 565.

**Gravitational Waves:**
- NANOGrav Collaboration (2023). "The NANOGrav 15-year Data Set: Evidence for a Gravitational-Wave Background." *ApJL*, 951, L8.

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/pulsar-detection-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pulsar-detection-system/discussions)
- **Email**: your.email@example.com

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 🌟 Acknowledgments

- Inspired by radio telescope data from Parkes, Arecibo, and Green Bank observatories
- Cosmological models based on Planck satellite observations
- Fast folding algorithm from pulsar astronomy literature
- Community feedback and contributions

## 🚀 Future Development

Planned features for upcoming releases:

- [ ] Real-time pulsar monitoring dashboard
- [ ] Machine learning-based candidate classification
- [ ] Integration with PRESTO and PSRCHIVE
- [ ] GPU acceleration for large-scale searches
- [ ] Interactive 3D universe visualization
- [ ] Web-based analysis interface
- [ ] Support for additional radio telescope formats
- [ ] Automated report generation

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the pulsar astronomy community

</div>
