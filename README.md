```markdown
# Directional Well Survey Processing System

## Overview
A Python-based processing system for directional drilling data management, specializing in survey calculations, boundary analysis, and spatial processing. The system interfaces with SQLite databases to process well trajectory data and analyze spatial relationships with lease boundaries.

## Key Features
- Survey interpolation and trajectory calculations
- Magnetic field corrections and transformations
- Plat boundary clearance analysis
- Multi-well interference checking
- Coordinate system conversions
- Concentration zone management

## Technical Requirements

### Database
- SQLite database containing tables:
  * PlatDF: Plat definitions and geometries
  * PlatAdj: Adjacent plat relationships
  * DX_Original: Raw directional survey data
  * Depths: Depth intervals and drill data

### Python Dependencies
```bash
numpy>=1.21.0
pandas>=1.3.0
welleng>=0.4.0
pyproj>=3.0.0
shapely>=1.8.0
scipy>=1.7.0
rdp>=0.8.0
utm>=0.7.0
sqlite3
```

## Data Structure

### Required Database Tables
1. PlatDF:
   - geometry (WKT format)
   - centroid
   - Additional plat metadata

2. PlatAdj:
   - geometry (WKT format)
   - centroid
   - Concentration values

3. DX_Original:
   - Survey measurement data
   - Directional data
   - Well metadata

4. Depths:
   - Interval data (JSON format)
   - Depth measurements
   - Drilling parameters

## Usage Example

```python
# Initialize database connection
conn = sqlite3.connect('your_database.db')

# Load and process survey data
survey_process = SurveyProcess(
    df_referenced=survey_data,
    drilled_depths=depth_data,
    elevation=surface_elevation,
    coords_type='latlon'
)

# Process clearances
clearance = ClearanceProcess(
    survey_process.df_t,
    plat_data,
    adjacent_plats
)

results = clearance.clearance_data
footages = clearance.whole_df
```

## Data Processing Flow
1. Database Connection & Data Loading
2. Geometry Conversion (WKT to Shapely)
3. Survey Processing & Interpolation
4. Clearance Calculations
5. Results Generation

## Sample Data
The included DX_sample.db contains example data structured in the required format. Use this as a template for organizing your production data.

## Configuration
Default configuration includes:
- Full column display in pandas
- Disabled chained assignment warnings
- Standard coordinate transformations
- Default elevation reference of 5515

## Best Practices
1. Always validate input data formats
2. Maintain consistent coordinate systems
3. Regular database backups
4. Quality control of interpolation results
5. Verification of boundary calculations

## Error Handling
The system includes handling for:
- Invalid geometry conversions
- Missing data fields
- Coordinate transformation errors
- Database connection issues

## Performance Considerations
- Large datasets may require batch processing
- Complex geometries impact calculation speed
- Memory usage scales with survey density
- Consider spatial indexing for large plat sets

## Development Guidelines
- Maintain consistent string quotation style
- Follow local code formatting patterns
- Document all new functionality
- Include type hints for new functions
- Add comments for complex calculations

## Future Enhancements
1. Parallel processing support
2. Advanced spatial indexing
3. Real-time calculation capabilities
4. Enhanced error reporting
5. Performance optimization options
```
