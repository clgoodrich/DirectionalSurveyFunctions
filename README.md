# Directional Well Survey Processing System

## Overview
This Python package provides comprehensive tools for processing directional well surveys and calculating plat boundary clearances. It handles survey interpolation, magnetic field corrections, coordinate transformations, and spatial relationship analysis between wellbores and lease boundaries.

## Key Features
- Survey data processing and interpolation
- Magnetic field reference handling
- Coordinate system transformations (UTM, lat/lon)
- Plat boundary clearance calculations
- Multi-well spatial relationship analysis
- Integration with industry-standard well engineering tools

## Installation

### Requirements
- Python 3.7+
- numpy
- pandas
- welleng
- pyproj
- shapely
- pygeomag
- scipy
- sqlite3

### Install
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example
```python
from dx_surveys import DXSurvey
from clearance import ClearanceProcess

# Process survey data
survey = DXSurvey(
    df=survey_df,
    start_nev=[0,0,0],
    conv_angle=1.5,
    interpolate=True
)
results = survey.process_trajectory()

# Calculate clearances
clearance = ClearanceProcess(
    df_used=results,
    df_plat=plat_boundaries,
    adjacent_plats=adjacent_plats_df
)
clearance_data = clearance.clearance_data
```

## Module Structure

### DXSurveys.py
Handles all survey-related calculations including:
- Minimum curvature interpolation
- Magnetic field corrections
- Coordinate transformations
- Trajectory calculations

### Clearance.py
Manages spatial relationships including:
- Plat boundary clearances
- FEL, FWL, FNL, FSL calculations
- Concentration zone assignments
- Adjacent plat handling

### MainDX.py
Main executable script that:
- Initializes database connections
- Loads and transforms survey data
- Processes trajectories
- Calculates clearances
- Outputs results

## Data Requirements

### Survey Data Format
Required columns:
- MeasuredDepth
- Inclination
- Azimuth
- Additional metadata fields

### Plat Data Format
Required columns:
- geometry (WKT format)
- centroid
- Concentration values
- Boundary definitions

## Configuration
- Database connection settings in DX_sample.db
- Coordinate system specifications
- Magnetic declination parameters
- Survey interpolation settings

## Output
The system produces:
- Processed survey data with interpolated points
- Clearance calculations to all relevant boundaries
- Concentration assignments
- Spatial relationship analysis
- Quality control metrics

## Best Practices
1. Always validate input data format
2. Check coordinate system consistency
3. Verify magnetic declination values
4. Monitor interpolation quality
5. Review clearance calculations
6. Backup database before processing

## Known Limitations
- Large datasets may require performance optimization
- Some coordinate transformations may need manual verification
- Complex plat geometries might need special handling
- Memory usage scales with survey point density


